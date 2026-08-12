// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// These tools make REAL network calls: web_search via the Tavily API and
// fetch_url via a plain HTTP GET. This is a local dev example — fetch_url will
// GET any http(s) URL the model chooses (no SSRF allowlist), so don't point it
// at untrusted inputs on a networked host.

// maxFetchBytes caps a fetched body so a huge page can't blow up the span it's
// recorded into (the tool result rides as a gen_ai.tool.call.result attribute).
const maxFetchBytes = 256 << 10 // 256 KiB

func mustSchema[T any]() json.RawMessage {
	schema, err := jsonschema.For[T](nil)
	if err != nil {
		panic(fmt.Sprintf("failed to generate schema: %v", err))
	}
	schema.AdditionalProperties = &jsonschema.Schema{}
	b, _ := json.Marshal(schema)
	return b
}

// ---- web_search (Tavily) ---------------------------------------------------

// WebSearchTool searches the web via the Tavily API (https://tavily.com).
type WebSearchTool struct {
	apiKey string
	client *http.Client
}

func NewWebSearchTool(apiKey string) *WebSearchTool {
	return &WebSearchTool{apiKey: apiKey, client: &http.Client{Timeout: 20 * time.Second}}
}

type WebSearchInput struct {
	Query string `json:"query" jsonschema:"The search query. Keep each query focused on ONE sub-topic."`
}

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

type WebSearchOutput struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
}

func (t *WebSearchTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "web_search",
		Description: "Search the web for a focused query and return ranked results (title, url, snippet). Issue several web_search calls in parallel for distinct sub-topics rather than one broad query.",
		Parameters:  mustSchema[WebSearchInput](),
	}
}

func (t *WebSearchTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in WebSearchInput
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}
	if strings.TrimSpace(in.Query) == "" {
		return nil, fmt.Errorf("query must not be empty")
	}

	reqBody, _ := json.Marshal(map[string]any{
		"api_key":      t.apiKey,
		"query":        in.Query,
		"max_results":  5,
		"search_depth": "basic",
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.tavily.com/search", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tavily search failed: %w", err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("tavily search returned %s: %s", resp.Status, strings.TrimSpace(string(body)))
	}

	var tav struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(body, &tav); err != nil {
		return nil, fmt.Errorf("decoding tavily response: %w", err)
	}
	out := WebSearchOutput{Query: in.Query}
	for _, r := range tav.Results {
		out.Results = append(out.Results, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Content})
	}
	return json.Marshal(out)
}

// ---- fetch_url (real HTTP GET) --------------------------------------------

// FetchURLTool fetches a URL and returns its (readable) content.
type FetchURLTool struct {
	client *http.Client
}

func NewFetchURLTool() *FetchURLTool {
	return &FetchURLTool{client: &http.Client{Timeout: 20 * time.Second}}
}

type FetchURLInput struct {
	URL string `json:"url" jsonschema:"The absolute http(s) URL to fetch, typically one returned by web_search."`
}

type FetchURLOutput struct {
	URL         string `json:"url"`
	Status      int    `json:"status"`
	ContentType string `json:"content_type"`
	Truncated   bool   `json:"truncated"`
	Content     string `json:"content"`
}

func (t *FetchURLTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "fetch_url",
		Description: "Fetch an http(s) URL and return its readable text (HTML is stripped to text; JSON is returned verbatim). Use after web_search to read promising results or to hit a known JSON API. Multiple fetch_url calls may be issued in parallel.",
		Parameters:  mustSchema[FetchURLInput](),
	}
}

func (t *FetchURLTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in FetchURLInput
	if err := json.Unmarshal(args, &in); err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}
	if !strings.HasPrefix(in.URL, "http://") && !strings.HasPrefix(in.URL, "https://") {
		return nil, fmt.Errorf("url %q must be absolute (http/https)", in.URL)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, in.URL, nil)
	if err != nil {
		return nil, err
	}
	// Some sites/APIs reject the default Go user agent.
	req.Header.Set("User-Agent", "ai-sdk-go-research-agent/0.1 (+https://redpanda.com)")
	req.Header.Set("Accept", "application/json, text/html;q=0.9, */*;q=0.8")

	resp, err := t.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch %s: %w", in.URL, err)
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes+1))
	truncated := len(raw) > maxFetchBytes
	if truncated {
		raw = raw[:maxFetchBytes]
	}

	ct := resp.Header.Get("Content-Type")
	content := string(raw)
	if strings.Contains(ct, "html") {
		content = htmlToText(content)
	}

	return json.Marshal(FetchURLOutput{
		URL:         in.URL,
		Status:      resp.StatusCode,
		ContentType: ct,
		Truncated:   truncated,
		Content:     content,
	})
}

var (
	reScriptStyle = regexp.MustCompile(`(?is)<(script|style)[^>]*>.*?</(script|style)>`)
	reTags        = regexp.MustCompile(`(?s)<[^>]+>`)
	reWhitespace  = regexp.MustCompile(`\s+`)
)

// htmlToText does a dependency-free, best-effort HTML→text reduction: drop
// script/style, strip tags, unescape entities, collapse whitespace. Good enough
// to feed a model without pulling in a full HTML parser.
func htmlToText(s string) string {
	s = reScriptStyle.ReplaceAllString(s, " ")
	s = reTags.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	s = reWhitespace.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

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
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// web_search makes REAL network calls via the Tavily API.

// maxFetchBytes caps the Tavily response body so a huge payload can't blow up
// the span it's recorded into (the tool result rides as a
// gen_ai.tool.call.result attribute).
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

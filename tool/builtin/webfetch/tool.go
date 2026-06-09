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

package webfetch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	md "github.com/JohannesKaufmann/html-to-markdown"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Tool implements the webfetch tool for making secure web requests.
type Tool struct {
	cfg Config
}

var _ tool.Tool = (*Tool)(nil)

// New creates a new webfetch tool with default configuration.
func New(opts ...Option) *Tool {
	cfg := DefaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &Tool{cfg: cfg}
}

// Name implements tool.Tool.
func (*Tool) Name() string { return "webfetch" }

// Description implements tool.Tool.
func (*Tool) Description() string {
	return "Fetch a HTTPS URL (GET/HEAD) with SSRF protection and size limits. Text/JSON/XML only."
}

// InputSchema implements tool.Tool.
func (t *Tool) InputSchema() json.RawMessage {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{
				"type":        "string",
				"description": "HTTPS URL to fetch",
			},
			"method": map[string]any{
				"type":        "string",
				"enum":        []string{"GET", "HEAD"},
				"default":     "GET",
				"description": "HTTP method to use",
			},
			"convert_to_markdown": map[string]any{
				"type":        "boolean",
				"default":     t.cfg.ConvertToMarkdown,
				"description": "Convert HTML content to markdown for better readability",
			},
		},
		"required":             []string{"url"},
		"additionalProperties": false,
	}

	schemaBytes, _ := json.Marshal(schema) //nolint:errchkjson // We know that this will succeed

	return schemaBytes
}

// Execute performs the webfetch operation. Per webfetch's prompt
// contract, request failures are encoded as `{"error":true,...}`
// payloads inside Output (not as IsError responses), so the model can
// reason about them in the same shape regardless of which leg fails.
func (t *Tool) Execute(ctx context.Context, call tool.Call) (tool.Execution, error) {
	var params struct {
		URL               string `json:"url"`
		Method            string `json:"method,omitempty"`
		ConvertToMarkdown *bool  `json:"convert_to_markdown,omitempty"`
	}

	if err := json.Unmarshal(call.Request.Arguments, &params); err != nil {
		return errorExecution(fmt.Errorf("invalid arguments: %w", err)), nil
	}

	if params.URL == "" {
		return errorExecution(errors.New("url is required")), nil
	}

	// Default method
	if params.Method == "" {
		params.Method = http.MethodGet
	}

	// Validate method
	if params.Method != http.MethodGet && params.Method != http.MethodHead {
		return errorExecution(fmt.Errorf("unsupported method %q", params.Method)), nil
	}

	// Determine if we should convert to markdown
	convertToMarkdown := t.cfg.ConvertToMarkdown
	if params.ConvertToMarkdown != nil {
		convertToMarkdown = *params.ConvertToMarkdown
	}

	// Perform request
	resp, err := doRequest(ctx, t.cfg, params.Method, params.URL)
	if err != nil {
		return errorExecution(err), nil
	}

	// Build response
	result := map[string]any{
		"url":          params.URL,
		"final_url":    resp.FinalURL,
		"status_code":  resp.StatusCode,
		"status":       resp.Status,
		"media_type":   resp.MediaType,
		"encoding":     resp.Encoding,
		"retrieved_at": resp.RetrievedAt.UTC().Format(time.RFC3339),
		"truncated":    resp.Truncated,
		"redirected":   resp.Redirected,
	}

	// Add body for GET requests with successful responses
	if params.Method == http.MethodGet && resp.StatusCode >= 200 && resp.StatusCode < 300 && len(resp.Body) > 0 {
		body := string(resp.Body)

		// Try to convert to markdown if requested
		if convertToMarkdown {
			if markdown, err := convertHTMLToMarkdown(body, resp.FinalURL); err == nil {
				result["body"] = t.fenceBodyContent(markdown)
				result["converted_to_markdown"] = true
			} else {
				// If conversion fails, return original content with error info
				result["body"] = t.fenceBodyContent(body)
				result["markdown_conversion_error"] = err.Error()
				result["converted_to_markdown"] = false
			}
		} else {
			result["body"] = t.fenceBodyContent(body)
			result["converted_to_markdown"] = false
		}
	}

	output, err := json.Marshal(result)
	if err != nil {
		return tool.Execution{}, fmt.Errorf("marshal webfetch result: %w", err)
	}

	return tool.Execution{Output: output}, nil
}

// fenceBodyContent wraps body content in fence delimiters to protect against
// prompt injection attacks. If fencing is disabled, returns the content unchanged.
func (t *Tool) fenceBodyContent(body string) string {
	if !t.cfg.Fencing {
		return body
	}

	return fence(body, t.cfg.FenceConfig)
}

// convertHTMLToMarkdown converts HTML content to markdown using the html-to-markdown library.
func convertHTMLToMarkdown(content, baseURL string) (string, error) {
	// Create converter with base URL for resolving relative links
	converter := md.NewConverter(baseURL, true, nil)

	// Remove unwanted elements
	converter.Remove("script", "style", "nav", "footer", "header", "noscript")

	// Convert to markdown
	markdown, err := converter.ConvertString(content)
	if err != nil {
		return "", fmt.Errorf("failed to convert to markdown: %w", err)
	}

	return markdown, nil
}

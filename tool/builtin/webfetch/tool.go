package webfetch

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	md "github.com/JohannesKaufmann/html-to-markdown"

	"github.com/redpanda-data/ai-sdk-go/llm"
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

// Definition returns the tool definition for LLM consumption.
func (t *Tool) Definition() llm.ToolDefinition {
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

	return llm.ToolDefinition{
		Name:        "webfetch",
		Description: "Fetch a HTTPS URL (GET/HEAD) with SSRF protection and size limits. Text/JSON/XML only.",
		Parameters:  schemaBytes,
		Metadata: map[string]any{
			"category": "web",
			"security": "high",
		},
	}
}

// Execute performs the webfetch operation.
func (t *Tool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var params struct {
		URL               string `json:"url"`
		Method            string `json:"method,omitempty"`
		ConvertToMarkdown *bool  `json:"convert_to_markdown,omitempty"`
	}

	if err := json.Unmarshal(args, &params); err != nil {
		return marshalErr(fmt.Errorf("invalid arguments: %w", err))
	}

	if params.URL == "" {
		return marshalErr(errors.New("url is required"))
	}

	// Default method
	if params.Method == "" {
		params.Method = http.MethodGet
	}

	// Validate method
	if params.Method != http.MethodGet && params.Method != http.MethodHead {
		return marshalErr(fmt.Errorf("unsupported method %q", params.Method))
	}

	// Determine if we should convert to markdown
	convertToMarkdown := t.cfg.ConvertToMarkdown
	if params.ConvertToMarkdown != nil {
		convertToMarkdown = *params.ConvertToMarkdown
	}

	// Perform request
	resp, err := doRequest(ctx, t.cfg, params.Method, params.URL)
	if err != nil {
		return marshalErr(err)
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

	return json.Marshal(result)
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

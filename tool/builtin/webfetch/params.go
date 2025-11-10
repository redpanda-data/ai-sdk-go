package webfetch

import (
	"encoding/json"
	"net/http"
)

// Parameters represents the input parameters for the webfetch tool.
// This struct provides type safety when creating webfetch requests.
type Parameters struct {
	// URL is the HTTPS URL to fetch (required)
	URL string `json:"url"`

	// Method is the HTTP method to use (optional, defaults to GET)
	Method string `json:"method,omitempty"`

	// ConvertToMarkdown indicates whether to convert HTML content to markdown (optional)
	ConvertToMarkdown *bool `json:"convert_to_markdown,omitempty"`
}

// NewParameters creates a new Parameters struct with the given URL and default values.
func NewParameters(url string) *Parameters {
	return &Parameters{
		URL:    url,
		Method: http.MethodGet,
	}
}

// WithMethod sets the HTTP method for the request.
func (p *Parameters) WithMethod(method string) *Parameters {
	p.Method = method
	return p
}

// WithMarkdownConversion sets whether to convert HTML to markdown.
func (p *Parameters) WithMarkdownConversion(convert bool) *Parameters {
	p.ConvertToMarkdown = &convert
	return p
}

// ToJSONRawMessage serializes the parameters to json.RawMessage for use with llm.ToolRequest.
func (p *Parameters) ToJSONRawMessage() (json.RawMessage, error) {
	return json.Marshal(p)
}

// MustToJSONRawMessage serializes the parameters to json.RawMessage and panics on error.
// This is useful in tests where you want to keep the code concise.
func (p *Parameters) MustToJSONRawMessage() json.RawMessage {
	data, err := p.ToJSONRawMessage()
	if err != nil {
		panic(err)
	}

	return data
}

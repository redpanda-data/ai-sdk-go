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

package openaicompat

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// normalizeBaseURL ensures the base URL ends with /v1 for OpenAI API compatibility.
//
// The OpenAI SDK expects the base URL to include the /v1 path segment
// (e.g., "https://api.openai.com/v1"), while other providers like Anthropic
// expect it without (e.g., "https://api.anthropic.com"). This normalization
// allows users to provide URLs in either format consistently across providers,
// bridging the gap between different SDK expectations.
func normalizeBaseURL(url string) string {
	url = strings.TrimSuffix(url, "/")
	if !strings.HasSuffix(url, "/v1") {
		url += "/v1"
	}

	return url
}

// Provider implements the OpenAI-compatible model provider using Chat Completion API.
type Provider struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	name       string // Provider name for observability (e.g., "deepseek", "together")
	client     *openai.Client
}

// Name returns the provider identifier.
// Returns the configured name, or "openaicompat" if not set.
func (p *Provider) Name() string {
	if p.name != "" {
		return p.name
	}

	return "openaicompat"
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new OpenAI-compatible provider with the required API key and optional configuration.
func NewProvider(apiKey string, opts ...ProviderOption) (*Provider, error) {
	if apiKey == "" {
		return nil, errors.New("API key is required")
	}

	timeout := 10 * time.Minute
	p := &Provider{
		APIKey:  apiKey,
		BaseURL: "https://api.openai.com/v1",
		HTTPClient: &http.Client{
			Timeout: timeout,
		},
		Timeout: timeout,
	}

	for _, opt := range opts {
		err := opt(p)
		if err != nil {
			return nil, fmt.Errorf("provider configuration error: %w", err)
		}
	}

	// Initialize OpenAI client with provider configuration
	clientOpts := []option.RequestOption{
		option.WithAPIKey(p.APIKey),
		option.WithBaseURL(p.BaseURL),
		option.WithHTTPClient(p.HTTPClient),
	}
	client := openai.NewClient(clientOpts...)
	p.client = &client

	return p, nil
}

// WithBaseURL sets a custom API endpoint for OpenAI-compatible providers.
// The URL is normalized to ensure it ends with /v1 for API compatibility.
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) error {
		if url == "" {
			return errors.New("base URL cannot be empty")
		}

		p.BaseURL = normalizeBaseURL(url)

		return nil
	}
}

// WithHTTPClient sets a custom HTTP client for API requests.
func WithHTTPClient(client *http.Client) ProviderOption {
	return func(p *Provider) error {
		if client == nil {
			return errors.New("HTTP client cannot be nil")
		}

		p.HTTPClient = client

		return nil
	}
}

// WithProviderName sets the provider name for observability and identification.
// This is useful when using openaicompat with different providers (e.g., "deepseek", "together").
// If not set, defaults to "openaicompat".
func WithProviderName(name string) ProviderOption {
	return func(p *Provider) error {
		p.name = name
		return nil
	}
}

// WithTimeout sets the request timeout for API calls.
// If a custom http.Client has been provided, the client is shallow-copied
// to avoid mutating caller state.
func WithTimeout(timeout time.Duration) ProviderOption {
	return func(p *Provider) error {
		if timeout <= 0 {
			return fmt.Errorf("timeout must be positive, got %v", timeout)
		}

		p.Timeout = timeout

		// Clone the existing client to avoid mutating caller-owned instances.
		clone := *p.HTTPClient // shallow copy preserves Transport, Jar, etc.
		clone.Timeout = timeout
		p.HTTPClient = &clone

		return nil
	}
}

// NewModel creates a new OpenAI-compatible model instance with the specified configuration.
// Supports any model name string for maximum compatibility with OpenAI-compatible services.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	if modelName == "" {
		return nil, errors.New("model name cannot be empty")
	}

	// Use permissive constraints for dynamic model support
	cfg := &Config{
		ModelName:   modelName,
		Constraints: getDefaultConstraints(),
		setOptions:  make(map[string]bool),
	}

	// Apply all options with validation
	for _, opt := range opts {
		err := opt(cfg)
		if err != nil {
			return nil, fmt.Errorf("invalid option for %s: %w", modelName, err)
		}
	}

	// Validate configuration
	err := cfg.Validate()
	if err != nil {
		return nil, fmt.Errorf("configuration validation failed for %s: %w", modelName, err)
	}

	// Use custom capabilities if provided, otherwise use defaults
	var capabilities llm.ModelCapabilities
	if cfg.CustomCapabilities != nil {
		capabilities = *cfg.CustomCapabilities
	} else {
		capabilities = getDefaultCapabilities()
	}

	// Override reasoning capability if configured (for backward compatibility)
	if cfg.SupportsReasoning {
		capabilities.Reasoning = true
	}

	return &Model{
		provider:       p,
		config:         cfg,
		capabilities:   capabilities,
		client:         p.client,
		requestMapper:  NewRequestMapper(cfg),
		responseMapper: NewResponseMapper(cfg.Constraints),
	}, nil
}

// Models returns an empty list since openaicompat supports dynamic model names.
// The actual models available depend on the API endpoint being used.
func (*Provider) Models() []llm.ModelDiscoveryInfo {
	return []llm.ModelDiscoveryInfo{}
}

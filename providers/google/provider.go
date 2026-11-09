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

package google

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the Google Gemini model provider.
type Provider struct {
	APIKey  string
	BaseURL string
	// HTTPClient is the client the caller passed to WithHTTPClient. Requests
	// go through a copy made when the provider is built, so a later change to
	// this client doesn't reach them.
	HTTPClient *http.Client
	Timeout    time.Duration
	client     *genai.Client
}

// NewProviderWithClient creates a provider over a genai client the caller
// built. It takes no API key: whatever authenticates the client is already
// on it. The provider makes its own copy of client so it can read
// x-should-retry from error responses and report stream error events that
// genai would drop. It doesn't change the caller's client.
func NewProviderWithClient(ctx context.Context, client *genai.Client) (*Provider, error) {
	if client == nil {
		return nil, errors.New("genai client is required")
	}

	if ctx == nil {
		return nil, errors.New("context is required")
	}

	// genai drops response headers from its errors, so the header has to be
	// read in the HTTP client. The same client rewrites data: {"error": ...}
	// stream events, which genai would otherwise drop. A genai client's HTTP
	// client can't be replaced after the client is built, so build a new genai
	// client from the old one's settings with the HTTP client wrapped. The
	// settings already include an HTTP client, so genai doesn't look for
	// credentials again.
	cfg := client.ClientConfig()
	cfg.HTTPClient = wrapHTTPClient(cfg.HTTPClient)

	// For the Gemini API, genai fills Project and Location from the
	// GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables.
	// genai refuses to build a client that has those and an API key, so clear
	// them. The Gemini API doesn't use them or Credentials.
	if cfg.Backend != genai.BackendVertexAI {
		cfg.Project, cfg.Location, cfg.Credentials = "", "", nil
	}

	wrapped, err := genai.NewClient(ctx, &cfg)
	if err != nil {
		return nil, fmt.Errorf("rebuild genai client: %w", err)
	}

	return &Provider{client: wrapped}, nil
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new Google Gemini provider with the required API key and optional configuration.
func NewProvider(ctx context.Context, apiKey string, opts ...ProviderOption) (*Provider, error) {
	if apiKey == "" {
		return nil, errors.New("API key is required")
	}

	if ctx == nil {
		return nil, errors.New("context is required")
	}

	timeout := 10 * time.Minute
	p := &Provider{
		APIKey:  apiKey,
		BaseURL: "",
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

	// Initialize Gemini client with provider configuration
	clientConfig := &genai.ClientConfig{
		APIKey: apiKey,
	}

	// Set HTTPClient if provided
	if p.HTTPClient != nil {
		clientConfig.HTTPClient = wrapHTTPClient(p.HTTPClient)
	}

	// Set BaseURL via HTTPOptions if provided
	if p.BaseURL != "" {
		clientConfig.HTTPOptions.BaseURL = p.BaseURL
	}

	client, err := genai.NewClient(ctx, clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	p.client = client

	return p, nil
}

// Name returns the provider identifier.
func (*Provider) Name() llm.ProviderID {
	return ProviderName
}

// WithBaseURL sets a custom API endpoint for the Google provider.
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) error {
		if url == "" {
			return errors.New("base URL cannot be empty")
		}

		p.BaseURL = url

		return nil
	}
}

// WithHTTPClient sets a custom HTTP client for API requests. The provider
// sends requests through a copy of client made at construction, so later
// changes to client (Timeout, Transport, Jar) don't affect it.
func WithHTTPClient(client *http.Client) ProviderOption {
	return func(p *Provider) error {
		if client == nil {
			return errors.New("HTTP client cannot be nil")
		}

		p.HTTPClient = client

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

// Close closes the Gemini client and releases resources.
// Note: The current version of the Gemini SDK doesn't require explicit closing.
func (p *Provider) Close() error {
	// Gemini client doesn't have a Close method in the current SDK version
	return nil
}

// NewModel creates a new Google Gemini model instance with the specified
// configuration.
//
// modelName is resolved against the catalog (exact ID or versioned
// variants such as "gemini-2.5-flash-001"); unresolvable names are
// rejected rather than guessed at.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	offering, ok := Catalog().Resolve(strings.TrimPrefix(modelName, "models/"))
	if !ok {
		return nil, fmt.Errorf("unsupported model: %s", modelName)
	}

	return p.NewModelFromOffering(modelName, offering, opts...)
}

// NewModelFromOffering creates a model from an offering the caller resolved
// from its own catalog. Requests name modelName, or the WithCustomModelName
// value when set. The model reports the offering's provider. It reports the
// offering ID as InvokedModelID when the offering is not Google's.
func (p *Provider) NewModelFromOffering(modelName string, offering catalog.Offering, opts ...Option) (llm.Model, error) {
	cfg := &Config{
		ModelName:   modelName,
		Constraints: offering.Constraints,
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

	if cfg.ReasoningEffort != nil && !slices.Contains(offering.Reasoning.Efforts, *cfg.ReasoningEffort) {
		return nil, fmt.Errorf("model %s does not support reasoning effort %q (supported: %v)", modelName, *cfg.ReasoningEffort, offering.Reasoning.Efforts)
	}

	if cfg.ThinkingBudget != nil && !thinkingBudgets[offering.ID].supports(*cfg.ThinkingBudget) {
		return nil, fmt.Errorf("model %s does not support thinking budget %d", modelName, *cfg.ThinkingBudget)
	}

	responseMapper := NewResponseMapper(offering.ID)
	responseMapper.reportOffering = offering.Provider() != ProviderName

	return &Model{
		config:         cfg,
		offering:       offering,
		client:         p.client,
		requestMapper:  NewRequestMapper(cfg),
		responseMapper: responseMapper,
	}, nil
}

// Catalog implements catalog.Provider: the validated Google model
// catalog, including pricing and lifecycle metadata.
func (*Provider) Catalog() *catalog.Catalog {
	return Catalog()
}

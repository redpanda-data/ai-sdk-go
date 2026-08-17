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

package openai

import (
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the OpenAI model provider.
type Provider struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	client     *openai.Client
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "openai"
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new OpenAI provider with the required API key and optional configuration.
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
// The URL is used as-is (trailing slash stripped). Callers talking to
// OpenAI-compatible APIs must include /v1 in the URL themselves; callers
// pointing at a gateway or proxy pass whatever URL that proxy expects.
func WithBaseURL(url string) ProviderOption {
	return func(p *Provider) error {
		if url == "" {
			return errors.New("base URL cannot be empty")
		}

		p.BaseURL = strings.TrimRight(url, "/")

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

// NewModel creates a new OpenAI model instance with the specified configuration.
// It accepts family names (e.g. "o3"), official aliases (e.g. "gpt-5.6"),
// and timestamped snapshot IDs (e.g. "o3-2025-04-16"), resolving them
// against the catalog for capability/constraint lookup.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	offering, ok := Catalog().Resolve(modelName)
	if !ok {
		return nil, fmt.Errorf("unsupported OpenAI model: %s", modelName)
	}

	return p.NewCompatModel(modelName, CompatModelDefinition{
		Capabilities: offering.Capabilities,
		Constraints:  offering.Constraints,
		Reasoning:    offering.Reasoning,
	}, opts...)
}

// CompatModelDefinition is the transport configuration NewCompatModel
// consumes: exactly the three pieces the OpenAI request path uses.
// It is deliberately not a catalog shape — models served through
// OpenAI-compatible endpoints (bedrock-mantle, self-hosted gateways)
// carry their catalog metadata elsewhere.
type CompatModelDefinition struct {
	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	Reasoning    catalog.ReasoningSupport
}

// NewCompatModel creates a model for an OpenAI-compatible endpoint using an
// explicit CompatModelDefinition, bypassing the built-in catalog and its
// name resolver. It is the constructor for OpenAI-shaped models that are
// not OpenAI's own — e.g. the Google Gemma 4 and OpenAI gpt-5.x models on
// the AWS bedrock-mantle Responses endpoint, which the Bedrock provider
// reaches by pointing this provider at a mantle base URL with a
// SigV4-signing HTTP client (see providers/bedrock/mantle.go).
//
// modelName is sent verbatim as the API model ID. All request/response/
// streaming behaviour is identical to NewModel — only catalog lookup
// differs.
func (p *Provider) NewCompatModel(modelName string, def CompatModelDefinition, opts ...Option) (llm.Model, error) {
	cfg := &Config{
		ModelName:   modelName,
		Constraints: def.Constraints,
		setOptions:  make(map[string]bool),
	}

	for _, opt := range opts {
		if err := opt(cfg); err != nil {
			return nil, fmt.Errorf("invalid option for %s: %w", modelName, err)
		}
	}

	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("configuration validation failed for %s: %w", modelName, err)
	}

	// A requested effort is validated against the declared list; an empty
	// list means the model has no effort control, so any requested effort
	// is rejected (previously an empty list silently accepted every
	// value, unlike the other providers).
	if cfg.ReasoningEffort != nil {
		if !slices.Contains(def.Reasoning.Efforts, *cfg.ReasoningEffort) {
			return nil, fmt.Errorf("model %s does not support reasoning effort %q (supported: %v)", modelName, *cfg.ReasoningEffort, def.Reasoning.Efforts)
		}
	}

	return &Model{
		provider:       p,
		config:         cfg,
		definition:     def,
		client:         p.client,
		requestMapper:  NewRequestMapper(cfg),
		responseMapper: NewResponseMapper(),
	}, nil
}

// Catalog implements catalog.Provider: the validated OpenAI model
// catalog, including pricing and lifecycle metadata.
func (*Provider) Catalog() *catalog.Catalog {
	return Catalog()
}


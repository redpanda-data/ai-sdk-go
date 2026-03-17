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
	"time"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the Google Gemini model provider.
type Provider struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	client     *genai.Client
	context    context.Context //nolint:containedctx // Context required for Gemini client lifetime management
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "gcp.gemini"
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new Google Gemini provider with the required API key and optional configuration.
//
//nolint:contextcheck // Context is intentionally stored for Gemini client operations
func NewProvider(ctx context.Context, apiKey string, opts ...ProviderOption) (*Provider, error) {
	if apiKey == "" {
		return nil, errors.New("API key is required")
	}

	if ctx == nil {
		ctx = context.Background()
	}

	timeout := 10 * time.Minute
	p := &Provider{
		APIKey:  apiKey,
		BaseURL: "",
		HTTPClient: &http.Client{
			Timeout: timeout,
		},
		Timeout: timeout,
		context: ctx,
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
		clientConfig.HTTPClient = p.HTTPClient
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

// Close closes the Gemini client and releases resources.
// Note: The current version of the Gemini SDK doesn't require explicit closing.
func (p *Provider) Close() error {
	// Gemini client doesn't have a Close method in the current SDK version
	return nil
}

// NewModel creates a new Google Gemini model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	modelDef, ok := supportedModels[modelName]
	if !ok {
		return nil, fmt.Errorf("unsupported model: %s", modelName)
	}

	cfg := &Config{
		ModelName:   modelName,
		Constraints: modelDef.Constraints,
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

	return &Model{
		provider:       p,
		config:         cfg,
		definition:     modelDef,
		client:         p.client,
		requestMapper:  NewRequestMapper(cfg),
		responseMapper: NewResponseMapper(modelDef),
	}, nil
}

// Models returns all Gemini models with their capabilities.
func (p *Provider) Models() []llm.ModelDiscoveryInfo {
	models := make([]llm.ModelDiscoveryInfo, 0, len(supportedModels))
	for _, def := range supportedModels {
		models = append(models, llm.ModelDiscoveryInfo{
			Name:         def.Name,
			Label:        def.Label,
			Capabilities: def.Capabilities,
			Provider:     p.Name(),
		})
	}

	return models
}

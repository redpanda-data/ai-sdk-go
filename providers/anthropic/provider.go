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

package anthropic

import (
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// normalizeBaseURL ensures the base URL does not end with /v1 for Anthropic API compatibility.
//
// The Anthropic SDK expects the base URL without the /v1 path segment
// (e.g., "https://api.anthropic.com"), while other providers like OpenAI
// expect it with /v1 (e.g., "https://api.openai.com/v1"). This normalization
// allows users to provide URLs in either format consistently across providers,
// bridging the gap between different SDK expectations.
func normalizeBaseURL(url string) string {
	url = strings.TrimSuffix(url, "/")
	url = strings.TrimSuffix(url, "/v1")

	return url
}

// Provider implements the Anthropic model provider.
type Provider struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	// EnableCaching enables prompt caching by setting cache_control markers
	EnableCaching bool
	client        *anthropic.Client
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "anthropic"
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new Anthropic provider with the required API key and optional configuration.
func NewProvider(apiKey string, opts ...ProviderOption) (*Provider, error) {
	if apiKey == "" {
		return nil, errors.New("API key is required")
	}

	timeout := 10 * time.Minute
	p := &Provider{
		APIKey:  apiKey,
		BaseURL: "https://api.anthropic.com",
		HTTPClient: &http.Client{
			Timeout: timeout,
		},
		Timeout: timeout,
		// Caching is on by default. Opt-in caching just means callers forget
		// to enable it; use WithCachingDisabled to turn it off.
		EnableCaching: true,
	}

	for _, opt := range opts {
		err := opt(p)
		if err != nil {
			return nil, fmt.Errorf("provider configuration error: %w", err)
		}
	}

	// Initialize Anthropic client with provider configuration
	clientOpts := []option.RequestOption{
		option.WithAPIKey(p.APIKey),
		option.WithBaseURL(p.BaseURL),
		option.WithHTTPClient(p.HTTPClient),
	}
	client := anthropic.NewClient(clientOpts...)
	p.client = &client

	return p, nil
}

// WithBaseURL sets a custom API endpoint for Anthropic-compatible providers.
// The URL is normalized to ensure it does not end with /v1 for API compatibility.
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

// WithCaching is retained for backward compatibility and is now a no-op:
// prompt caching is enabled by default. Use WithCachingDisabled to turn it off.
func WithCaching() ProviderOption {
	return func(p *Provider) error {
		p.EnableCaching = true
		return nil
	}
}

// WithCachingDisabled turns off prompt caching, so no cache_control markers are
// emitted. Caching is on by default because an agent re-sending a stable system
// prompt and toolset every turn gets much cheaper cache reads for a one-time
// write premium, and below the model's token minimum the marker is ignored
// server-side. Use this only when you specifically do not want caching.
func WithCachingDisabled() ProviderOption {
	return func(p *Provider) error {
		p.EnableCaching = false
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

// defaultMaxTokens is a last-resort fallback, NOT a recommended budget.
//
// Anthropic requires max_tokens on every request, so the SDK must send some
// value when neither WithMaxTokens nor a per-request override (RequestOptions)
// is set. The budget *policy* belongs to the caller, not the SDK — so this value
// only needs to be a safe default that works out of the box, not the "right"
// number for any given workload.
//
// It is deliberately bounded rather than the model's output max: max_tokens is a
// reservation against the context window (input + max_tokens must fit), so
// defaulting to the model max would 400 long conversations. 16K is generous but
// bounded: enough for long answers, small enough to stay clear of context-window
// rejections on typical conversations.
const defaultMaxTokens = 16384

// NewModel creates a new Anthropic model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	family := resolveModelFamily(modelName)

	modelDef, ok := supportedModels[family]
	if !ok {
		return nil, fmt.Errorf("unsupported Anthropic model: %s", modelName)
	}

	cfg := &Config{
		ModelName:        modelName,
		Constraints:      modelDef.Constraints,
		MaxTokens:        defaultMaxTokens, // Required by Anthropic API; see const.
		EnableCaching:    p.EnableCaching,
		AdaptiveThinking: modelDef.AdaptiveThinking,
		setOptions:       make(map[string]bool),
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

	// Validate effort against model's supported values
	if cfg.ReasoningEffort != nil {
		if !slices.Contains(modelDef.SupportedReasoningEfforts, *cfg.ReasoningEffort) {
			return nil, fmt.Errorf("model %s does not support reasoning effort %q (supported: %v)", modelName, *cfg.ReasoningEffort, modelDef.SupportedReasoningEfforts)
		}
	}

	// Validate speed against model's supported values
	if cfg.Speed != nil {
		if len(modelDef.SupportedSpeeds) == 0 || !slices.Contains(modelDef.SupportedSpeeds, *cfg.Speed) {
			return nil, fmt.Errorf("model %s does not support speed '%s'", modelName, *cfg.Speed)
		}
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

// Models returns all Anthropic models with their capabilities.
func (*Provider) Models() []llm.ModelDiscoveryInfo {
	models := make([]llm.ModelDiscoveryInfo, 0, len(supportedModels))
	for _, def := range supportedModels {
		models = append(models, llm.ModelDiscoveryInfo{
			Name:         def.Name,
			Label:        def.Label,
			Capabilities: def.Capabilities,
			Constraints:  def.Constraints,
			Provider:     "anthropic",
		})
	}

	return models
}

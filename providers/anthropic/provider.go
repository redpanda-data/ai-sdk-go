package anthropic

import (
	"errors"
	"fmt"
	"net/http"
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

// WithCaching enables prompt caching by setting cache_control markers on requests.
// When enabled, the SDK automatically marks the last message content block for caching,
// allowing Anthropic to cache the conversation prefix for cost savings.
func WithCaching() ProviderOption {
	return func(p *Provider) error {
		p.EnableCaching = true
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

// NewModel creates a new Anthropic model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	// Resolve alias to canonical name if it exists
	if canonical, ok := modelAliases[modelName]; ok {
		modelName = canonical
	}

	modelDef, ok := supportedModels[modelName]
	if !ok {
		return nil, fmt.Errorf("unsupported Anthropic model: %s", modelName)
	}

	cfg := &Config{
		ModelName:     modelName,
		Constraints:   modelDef.Constraints,
		MaxTokens:     4096, // Default required by Anthropic API
		EnableCaching: p.EnableCaching,
		setOptions:    make(map[string]bool),
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

// Models returns all Anthropic models with their capabilities.
func (*Provider) Models() []llm.ModelDiscoveryInfo {
	models := make([]llm.ModelDiscoveryInfo, 0, len(supportedModels))
	for _, def := range supportedModels {
		models = append(models, llm.ModelDiscoveryInfo{
			Name:         def.Name,
			Label:        def.Label,
			Capabilities: def.Capabilities,
			Provider:     "anthropic",
		})
	}

	return models
}

package openaicompat

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/option"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the OpenAI-compatible model provider using Chat Completion API.
type Provider struct {
	APIKey     string
	BaseURL    string
	HTTPClient *http.Client
	Timeout    time.Duration
	client     *openai.Client
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

	// Use default capabilities for dynamic models
	capabilities := getDefaultCapabilities()
	// Override reasoning capability if configured
	if cfg.SupportsReasoning {
		capabilities.Reasoning = true
	}

	return &Model{
		provider:       p,
		config:         cfg,
		capabilities:   capabilities,
		client:         p.client,
		requestMapper:  NewRequestMapper(cfg),
		responseMapper: NewResponseMapper(),
	}, nil
}

// Models returns an empty list since openaicompat supports dynamic model names.
// The actual models available depend on the API endpoint being used.
func (*Provider) Models() []llm.ModelDiscoveryInfo {
	return []llm.ModelDiscoveryInfo{}
}

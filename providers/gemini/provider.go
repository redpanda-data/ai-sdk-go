package gemini

import (
	"context"
	"errors"
	"fmt"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the Gemini model provider.
type Provider struct {
	APIKey  string
	client  *genai.Client
	context context.Context //nolint:containedctx // Context required for Gemini client lifetime management
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*Provider) error

// NewProvider creates a new Gemini provider with the required API key and optional configuration.
//
//nolint:contextcheck // Context is intentionally stored for Gemini client operations
func NewProvider(ctx context.Context, apiKey string, opts ...ProviderOption) (*Provider, error) {
	if apiKey == "" {
		return nil, errors.New("API key is required")
	}

	if ctx == nil {
		ctx = context.Background()
	}

	p := &Provider{
		APIKey:  apiKey,
		context: ctx,
	}

	for _, opt := range opts {
		err := opt(p)
		if err != nil {
			return nil, fmt.Errorf("provider configuration error: %w", err)
		}
	}

	// Initialize Gemini client
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey: apiKey,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create Gemini client: %w", err)
	}

	p.client = client

	return p, nil
}

// Close closes the Gemini client and releases resources.
// Note: The current version of the Gemini SDK doesn't require explicit closing.
func (p *Provider) Close() error {
	// Gemini client doesn't have a Close method in the current SDK version
	return nil
}

// NewModel creates a new Gemini model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	// Resolve alias to canonical name if it exists
	if canonical, ok := modelAliases[modelName]; ok {
		modelName = canonical
	}

	modelDef, ok := supportedModels[modelName]
	if !ok {
		return nil, fmt.Errorf("unsupported Gemini model: %s", modelName)
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
		responseMapper: NewResponseMapper(),
	}, nil
}

// Models returns all Gemini models with their capabilities.
func (*Provider) Models() []llm.ModelDiscoveryInfo {
	models := make([]llm.ModelDiscoveryInfo, 0, len(supportedModels))
	for _, def := range supportedModels {
		models = append(models, llm.ModelDiscoveryInfo{
			Name:         def.Name,
			Label:        def.Label,
			Capabilities: def.Capabilities,
			Provider:     "gemini",
		})
	}

	return models
}

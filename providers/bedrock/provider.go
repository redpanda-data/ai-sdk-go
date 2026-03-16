package bedrock

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the Bedrock model provider using the Converse API.
type Provider struct {
	client        *bedrockruntime.Client
	region        string
	enableCaching bool
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*providerConfig) error

// providerConfig holds intermediate configuration before creating the provider.
type providerConfig struct {
	awsCfg     *aws.Config
	httpClient *http.Client
	region     string
	caching    bool
}

// NewProvider creates a new Bedrock provider.
// It uses the AWS SDK default credential chain (env vars, IAM roles, SSO, etc.).
func NewProvider(ctx context.Context, opts ...ProviderOption) (*Provider, error) {
	cfg := &providerConfig{}

	for _, opt := range opts {
		if err := opt(cfg); err != nil {
			return nil, fmt.Errorf("provider configuration error: %w", err)
		}
	}

	// Load AWS config if not provided
	var awsCfg aws.Config
	if cfg.awsCfg != nil {
		awsCfg = *cfg.awsCfg
	} else {
		var loadOpts []func(*awsconfig.LoadOptions) error
		if cfg.region != "" {
			loadOpts = append(loadOpts, awsconfig.WithRegion(cfg.region))
		}

		var err error

		awsCfg, err = awsconfig.LoadDefaultConfig(ctx, loadOpts...)
		if err != nil {
			return nil, fmt.Errorf("load AWS config: %w", err)
		}
	}

	// Build bedrockruntime client options
	var clientOpts []func(*bedrockruntime.Options)
	if cfg.httpClient != nil {
		clientOpts = append(clientOpts, func(o *bedrockruntime.Options) {
			o.HTTPClient = cfg.httpClient
		})
	}

	client := bedrockruntime.NewFromConfig(awsCfg, clientOpts...)

	return &Provider{
		client:        client,
		region:        awsCfg.Region,
		enableCaching: cfg.caching,
	}, nil
}

// Name returns the provider identifier.
func (*Provider) Name() string {
	return "aws.bedrock"
}

// WithAWSConfig sets a pre-loaded AWS configuration.
func WithAWSConfig(awsCfg aws.Config) ProviderOption {
	return func(cfg *providerConfig) error {
		cfg.awsCfg = &awsCfg
		return nil
	}
}

// WithHTTPClient sets a custom HTTP client for API requests.
func WithHTTPClient(client *http.Client) ProviderOption {
	return func(cfg *providerConfig) error {
		if client == nil {
			return errors.New("HTTP client cannot be nil")
		}

		cfg.httpClient = client

		return nil
	}
}

// WithRegion sets the AWS region for the Bedrock endpoint.
func WithRegion(region string) ProviderOption {
	return func(cfg *providerConfig) error {
		if region == "" {
			return errors.New("region cannot be empty")
		}

		cfg.region = region

		return nil
	}
}

// WithCaching enables prompt caching on Bedrock.
func WithCaching() ProviderOption {
	return func(cfg *providerConfig) error {
		cfg.caching = true
		return nil
	}
}

// NewModel creates a new Bedrock model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	family := resolveModelFamily(modelName)

	modelDef, ok := supportedModels[family]
	if !ok {
		return nil, fmt.Errorf("unsupported Bedrock model: %s", modelName)
	}

	// Resolve the model ID to send to the Bedrock API.
	// If the user passed a short name (matching the family key exactly),
	// build the inference profile ID: {regionPrefix}.{modelID}
	// Otherwise, the user passed a qualified ID — use it as-is.
	apiModelID := modelName
	if modelName == family && modelDef.DefaultModelID != "" {
		apiModelID = inferenceProfileRegion(p.region) + "." + modelDef.DefaultModelID
	}

	cfg := &Config{
		ModelName:     modelName,
		APIModelID:    apiModelID,
		Constraints:   modelDef.Constraints,
		EnableCaching: p.enableCaching,
		setOptions:    make(map[string]bool),
	}

	for _, opt := range opts {
		if err := opt(cfg); err != nil {
			return nil, fmt.Errorf("invalid option for %s: %w", modelName, err)
		}
	}

	if err := cfg.Validate(); err != nil {
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

// Models returns all supported Bedrock models with their capabilities.
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

	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models
}

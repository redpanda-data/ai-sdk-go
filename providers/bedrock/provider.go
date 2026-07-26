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

package bedrock

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sort"

	"github.com/aws/aws-sdk-go-v2/aws"
	signerv4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider implements the Bedrock model provider. Standard models use the
// bedrock-runtime Converse API; mantle-only models (Mantle: true in the
// catalog) use the OpenAI-compatible Responses API on bedrock-mantle, signed by
// the SigV4 transport in mantle.go — which is why the provider also retains the
// credentials, region, and a shared signer.
type Provider struct {
	client        *bedrockruntime.Client
	region        string
	enableCaching bool

	// Retained for the mantle Responses transport (see mantle.go).
	credentials  aws.CredentialsProvider
	signer       *signerv4.Signer
	httpClient   *http.Client // caller-supplied client, if any (base transport)
	baseEndpoint string       // custom base endpoint (proxy/gateway mode), if any
}

// ProviderOption configures a Provider instance using functional options.
type ProviderOption func(*providerConfig) error

// providerConfig holds intermediate configuration before creating the provider.
type providerConfig struct {
	awsCfg     *aws.Config
	httpClient *http.Client
	region     string
	caching    bool
	noAuth     bool // skip SigV4 signing (for proxied/gateway mode)
}

// NewProvider creates a new Bedrock provider.
// It uses the AWS SDK default credential chain (env vars, IAM roles, SSO, etc.).
func NewProvider(ctx context.Context, opts ...ProviderOption) (*Provider, error) {
	// Caching is on by default. Bedrock does no automatic prefix caching, so
	// these explicit markers are the only way to cache at all; opt-in just means
	// callers forget. Use WithCachingDisabled to turn it off.
	cfg := &providerConfig{caching: true}

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

		if cfg.noAuth {
			// Skip SigV4 signing — use anonymous credentials so the AWS SDK
			// sends requests without an Authorization header. Intended for
			// proxied/gateway mode where the proxy handles authentication.
			loadOpts = append(loadOpts, awsconfig.WithCredentialsProvider(aws.AnonymousCredentials{}))
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

	var baseEndpoint string
	if awsCfg.BaseEndpoint != nil {
		baseEndpoint = *awsCfg.BaseEndpoint
	}

	// Resolve the HTTP client the mantle transport wraps. WithHTTPClient wins;
	// otherwise honor a client supplied via WithAWSConfig (awsCfg.HTTPClient) so
	// mantle requests use the same transport as the Converse client does, rather
	// than silently falling back to the default. Anything else leaves it nil and
	// mantle.go uses http.DefaultTransport.
	httpClient := cfg.httpClient
	if httpClient == nil {
		if hc, ok := awsCfg.HTTPClient.(*http.Client); ok {
			httpClient = hc
		}
	}

	return &Provider{
		client:        client,
		region:        awsCfg.Region,
		enableCaching: cfg.caching,
		credentials:   awsCfg.Credentials,
		signer:        signerv4.NewSigner(),
		httpClient:    httpClient,
		baseEndpoint:  baseEndpoint,
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

// WithNoAuth disables AWS SigV4 request signing. Use this when routing
// requests through a proxy (such as the Redpanda AI Gateway) that handles
// authentication on behalf of the client.
func WithNoAuth() ProviderOption {
	return func(cfg *providerConfig) error {
		cfg.noAuth = true
		return nil
	}
}

// WithCaching is retained for backward compatibility and is now a no-op:
// prompt caching is enabled by default. Use WithCachingDisabled to turn it off.
func WithCaching() ProviderOption {
	return func(cfg *providerConfig) error {
		cfg.caching = true
		return nil
	}
}

// WithCachingDisabled turns off prompt caching, so no CachePoint markers are
// emitted. Caching is on by default because Bedrock does no automatic prefix
// caching: the explicit CachePoint markers are the only way to cache, and an
// agent re-sending a stable prefix every turn benefits immediately. Use this
// only when you specifically do not want caching.
func WithCachingDisabled() ProviderOption {
	return func(cfg *providerConfig) error {
		cfg.caching = false
		return nil
	}
}

// NewModel creates a new Bedrock model instance with the specified configuration.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	// Build the API model ID. Most Bedrock models are cross-region inference
	// profiles, so an un-prefixed name gets the source region's geo prefix
	// (e.g. "anthropic.claude-sonnet-4-6" -> "us.anthropic.claude-sonnet-4-6").
	// Two cases skip prefixing: a name that already carries a region prefix
	// (e.g. "eu.anthropic.…"), and a name that is itself a catalog entry — a
	// few third-party models (e.g. "mistral.mistral-large-3-675b-instruct") are
	// on-demand / in-region and are invoked by their bare ID, so an exact
	// catalog match is used as-is.
	apiModelID := modelName

	if _, registered := lookupModel(apiModelID); !registered && !hasRegionPrefix(apiModelID) {
		geo := InferenceProfileRegion(p.region)
		if geo != "" {
			geoModelID := geo + "." + apiModelID
			if _, ok := lookupModel(geoModelID); ok {
				apiModelID = geoModelID
			} else {
				globalModelID := "global." + apiModelID
				if _, ok := lookupModel(globalModelID); ok {
					apiModelID = globalModelID
				}
			}
		}
	}

	// Look up by the prefixed ID — each inference profile variant is a
	// separate catalog entry with its own pricing (geo profiles carry a
	// cross-region premium over the base/global rate).
	modelDef, ok := lookupModel(apiModelID)
	if !ok {
		return nil, fmt.Errorf("unsupported Bedrock model: %s", modelName)
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

	// Mantle-only models (Gemma 4, gpt-5.x) are not served by the Converse API;
	// route them through the SigV4-signed OpenAI Responses transport instead.
	if modelDef.Mantle {
		return newMantleModel(p, cfg, modelDef)
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
			Constraints:  def.Constraints,
			Provider:     p.Name(),
			Metadata:     def.discoveryMetadata(),
		})
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].Name < models[j].Name
	})

	return models
}

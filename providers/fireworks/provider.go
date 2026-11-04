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

// Package fireworks provides Fireworks serverless models and their catalog.
package fireworks

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"time"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/option"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

const (
	defaultBaseURL   = "https://api.fireworks.ai/inference/v1"
	defaultMaxTokens = 4096
)

// Provider supplies Fireworks models. Its zero value supports catalog discovery;
// constructing a model requires WithAPIKey.
type Provider struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

var _ catalog.Provider = (*Provider)(nil)

// ProviderOption configures the endpoint and bearer credential.
type ProviderOption func(*Provider) error

// NewProvider creates a provider without reading credentials or contacting the API.
func NewProvider(_ context.Context, opts ...ProviderOption) (*Provider, error) {
	p := &Provider{baseURL: defaultBaseURL, httpClient: &http.Client{Timeout: 10 * time.Minute}}

	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("nil Fireworks provider option")
		}

		if err := opt(p); err != nil {
			return nil, fmt.Errorf("fireworks provider configuration: %w", err)
		}
	}

	return p, nil
}

// WithAPIKey sets the bearer credential: a Fireworks key or a gateway token.
func WithAPIKey(key string) ProviderOption {
	return func(p *Provider) error {
		if strings.TrimSpace(key) == "" || strings.TrimSpace(key) != key || strings.ContainsAny(key, "\r\n") {
			return errors.New("a nonempty bearer credential without surrounding whitespace is required")
		}

		p.apiKey = key

		return nil
	}
}

// WithBaseURL sets the inference API root, including its /v1 path segment.
func WithBaseURL(baseURL string) ProviderOption {
	return func(p *Provider) error {
		u, err := url.Parse(baseURL)
		if err != nil || u == nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.ForceQuery || !strings.HasSuffix(strings.TrimRight(u.Path, "/"), "/v1") {
			return errors.New("base URL must be an HTTP(S) API root ending in /v1, without credentials, query or fragment")
		}

		p.baseURL = strings.TrimRight(baseURL, "/")

		return nil
	}
}

// WithHTTPClient sets the client used by the selected wire implementation.
func WithHTTPClient(client *http.Client) ProviderOption {
	return func(p *Provider) error {
		if client == nil {
			return errors.New("HTTP client cannot be nil")
		}

		p.httpClient = client

		return nil
	}
}

// Name returns the provider key used in discovery and pricing.
func (*Provider) Name() llm.ProviderID { return ProviderName }

// Catalog returns the immutable provider catalog.
func (*Provider) Catalog() *catalog.Catalog { return Catalog() }

// NewModel resolves an exact Fireworks offering and uses its first normalized
// WireAPI. An unsupported first binding is an error, never a fallback request.
func (p *Provider) NewModel(modelID string, opts ...Option) (llm.Model, error) {
	offering, ok := p.Catalog().Resolve(modelID)
	if !ok {
		return nil, fmt.Errorf("unsupported Fireworks model: %s", modelID)
	}

	return p.newModelFromOffering(modelID, offering, opts...)
}

func (p *Provider) newModelFromOffering(modelID string, offering catalog.Offering, opts ...Option) (llm.Model, error) {
	if len(offering.WireAPIs) == 0 {
		return nil, errors.New("fireworks offering has no wire API")
	}

	api := offering.WireAPIs[0]
	if api.Transport != catalog.WireTransportDirect || !slices.Contains([]catalog.WireProtocol{catalog.WireProtocolAnthropicMessages, catalog.WireProtocolOpenAIResponses, catalog.WireProtocolOpenAIChatCompletions}, api.Protocol) {
		return nil, fmt.Errorf("unsupported first Fireworks wire API: %s/%s", api.Protocol, api.Transport)
	}

	support, ok := offering.SupportForWireAPI(api)
	if !ok {
		return nil, fmt.Errorf("unknown Fireworks support for first wire API: %s/%s", api.Protocol, api.Transport)
	}

	if p.apiKey == "" {
		return nil, errors.New("fireworks model requires WithAPIKey")
	}

	selected := offering
	selected.Capabilities = support.Capabilities
	// llm.Part cannot represent images; native catalog support remains intact.
	selected.Capabilities.Vision = false
	selected.Reasoning = support.Reasoning

	selected.Constraints.SupportedParams = []string{"max_tokens", "temperature"}
	if len(support.Reasoning.Efforts) > 0 {
		selected.Constraints.SupportedParams = append(selected.Constraints.SupportedParams, "reasoning_effort")
	}

	if support.Reasoning.Adaptive {
		selected.Constraints.SupportedParams = append(selected.Constraints.SupportedParams, "thinking")
	}

	cfg := modelConfig{maxTokens: min(defaultMaxTokens, selected.Constraints.MaxOutputTokens)}

	for _, opt := range opts {
		if opt == nil {
			return nil, errors.New("nil Fireworks model option")
		}

		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	if err := cfg.validate(selected); err != nil {
		return nil, fmt.Errorf("invalid Fireworks model %s: %w", modelID, err)
	}

	if offering.ID == ModelMiniMaxM3 && api.Protocol == catalog.WireProtocolAnthropicMessages && cfg.thinking != nil && !*cfg.thinking {
		selected.Capabilities.StructuredOutput = false
	}

	baseURL, client := p.baseURL, p.httpClient
	if baseURL == "" {
		baseURL = defaultBaseURL
	}

	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	var runtime llm.Model
	var err error

	switch api.Protocol {
	case catalog.WireProtocolAnthropicMessages:
		runtime, err = p.messagesModel(modelID, selected, cfg, baseURL, client)
	case catalog.WireProtocolOpenAIResponses:
		runtime, err = p.responsesModel(modelID, selected, cfg, baseURL, client)
	case catalog.WireProtocolOpenAIChatCompletions:
		runtime, err = p.chatModel(modelID, selected, cfg, baseURL, client)
	case catalog.WireProtocolGeminiGenerateContent, catalog.WireProtocolBedrockConverse:
		return nil, fmt.Errorf("unsupported first Fireworks wire API: %s/%s", api.Protocol, api.Transport)
	default:
		return nil, fmt.Errorf("unsupported first Fireworks wire API: %s/%s", api.Protocol, api.Transport)
	}

	if err != nil {
		return nil, err
	}

	return &model{Model: runtime, name: modelID, offering: selected}, nil
}

func (p *Provider) messagesModel(modelID string, offering catalog.Offering, cfg modelConfig, baseURL string, client *http.Client) (llm.Model, error) {
	clientOpts := []anthropicoption.RequestOption{
		anthropicoption.WithAPIKey(""), anthropicoption.WithAuthToken(p.apiKey),
		anthropicoption.WithBaseURL(strings.TrimSuffix(baseURL, "/v1")), anthropicoption.WithHTTPClient(client),
	}

	if cfg.thinking != nil || (cfg.effort != nil && *cfg.effort == ReasoningEffortNone) {
		mode := "disabled"
		if cfg.thinking != nil && *cfg.thinking {
			mode = "adaptive"
		}

		clientOpts = append(clientOpts, anthropicoption.WithJSONSet("thinking", map[string]string{"type": mode}))
	}

	vendor := anthropicsdk.NewClient(clientOpts...)

	provider, err := anthropic.NewProviderWithClient(&vendor)
	if err != nil {
		return nil, err
	}

	provider.EnableCaching = false

	opts := []anthropic.Option{anthropic.WithMaxTokens(cfg.maxTokens)}
	if cfg.temperature != nil {
		opts = append(opts, anthropic.WithTemperature(*cfg.temperature))
	}

	if cfg.effort != nil && *cfg.effort != ReasoningEffortNone {
		opts = append(opts, anthropic.WithReasoningEffort(*cfg.effort))
	}

	return provider.NewModelFromOffering(modelID, offering, opts...)
}

func (p *Provider) responsesModel(modelID string, offering catalog.Offering, cfg modelConfig, baseURL string, client *http.Client) (llm.Model, error) {
	provider, err := openai.NewProvider(p.apiKey, openai.WithBaseURL(baseURL), openai.WithHTTPClient(client))
	if err != nil {
		return nil, err
	}

	opts := []openai.Option{openai.WithMaxTokens(cfg.maxTokens), openai.WithStore(false)}
	if cfg.temperature != nil {
		opts = append(opts, openai.WithTemperature(*cfg.temperature))
	}

	if cfg.effort != nil {
		opts = append(opts, openai.WithReasoningEffort(*cfg.effort))
	}

	return provider.NewCompatModel(modelID, openai.CompatModelDefinition{Capabilities: offering.Capabilities, Constraints: offering.Constraints, Reasoning: offering.Reasoning}, opts...)
}

func (p *Provider) chatModel(modelID string, offering catalog.Offering, cfg modelConfig, baseURL string, client *http.Client) (llm.Model, error) {
	provider, err := openaicompat.NewProvider(p.apiKey, openaicompat.WithBaseURL(baseURL), openaicompat.WithHTTPClient(client), openaicompat.WithProviderName(ProviderName))
	if err != nil {
		return nil, err
	}

	opts := []openaicompat.Option{openaicompat.WithConstraints(offering.Constraints), openaicompat.WithCapabilities(offering.Capabilities), openaicompat.WithMaxTokens(cfg.maxTokens)}
	if offering.Capabilities.Reasoning {
		opts = append(opts, openaicompat.WithReasoning())
	}

	if cfg.temperature != nil {
		opts = append(opts, openaicompat.WithTemperature(*cfg.temperature))
	}

	if cfg.effort != nil {
		if *cfg.effort == ReasoningEffortNone {
			opts = append(opts, openaicompat.WithThinking(false))
		} else {
			opts = append(opts, openaicompat.WithReasoningEffort(*cfg.effort))
		}
	}

	if cfg.thinking != nil {
		if *cfg.thinking {
			opts = append(opts, openaicompat.WithAdaptiveThinking())
		} else {
			opts = append(opts, openaicompat.WithThinking(false))
		}
	}

	return provider.NewModel(modelID, opts...)
}

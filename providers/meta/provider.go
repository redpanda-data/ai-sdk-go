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

// Package meta provides Muse Spark through Meta's OpenAI-compatible Responses API.
package meta

import (
	"errors"
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// Provider serves the Meta Model API catalog using the shared Responses transport.
// Configure HTTP clients, timeouts, and proxy URLs with openai.ProviderOption.
type Provider struct{ transport *openai.Provider }

// NewProvider creates a Meta provider using a Meta Model API key.
// Options may override the default endpoint for a proxy or test server.
func NewProvider(apiKey string, opts ...openai.ProviderOption) (*Provider, error) {
	options := append([]openai.ProviderOption{openai.WithBaseURL("https://api.meta.ai/v1")}, opts...)

	transport, err := openai.NewProvider(apiKey, options...)
	if err != nil {
		return nil, fmt.Errorf("meta: %w", err)
	}

	return &Provider{transport: transport}, nil
}

// Name returns the provider identity used by catalog and billing consumers.
func (*Provider) Name() string { return "meta" }

// Catalog returns the Meta model catalog.
func (*Provider) Catalog() *catalog.Catalog { return Catalog() }

// NewModel creates a registered Meta model. Options reuse the OpenAI Responses
// transport's temperature, max-token, reasoning-effort, and summary controls.
// Catalog modalities describe the vendor model; binary input is not yet mapped
// by the shared transport.
func (p *Provider) NewModel(modelName string, opts ...openai.Option) (llm.Model, error) {
	offering, ok := Catalog().Lookup(modelName)
	if !ok {
		return nil, fmt.Errorf("unsupported Meta model: %s", modelName)
	}

	options := append(slices.Clone(opts), func(cfg *openai.Config) error {
		if cfg.MaxTokens == nil {
			budget := maxOutputTokens
			cfg.MaxTokens = &budget
		}
		// https://dev.meta.ai/docs/protocols/responses — minimum output budget.
		if cfg.MaxTokens != nil && (*cfg.MaxTokens < 16 || *cfg.MaxTokens > maxOutputTokens) {
			return fmt.Errorf("meta: max_tokens must be between 16 and %d (SDK limit)", maxOutputTokens)
		}
		// These vendor parameters are not wired by the shared Responses mapper.
		// Reject rather than silently ignoring explicitly requested controls.
		if cfg.TopP != nil || cfg.FrequencyPenalty != nil || cfg.PresencePenalty != nil || cfg.Seed != nil {
			return errors.New("meta: top_p, frequency_penalty, presence_penalty, and seed are not supported by the Responses transport")
		}

		return nil
	})

	model, err := p.transport.NewCompatModel(modelName, openai.CompatModelDefinition{
		Capabilities: offering.Capabilities, Constraints: offering.Constraints, Reasoning: offering.Reasoning,
	}, options...)
	if err != nil {
		return nil, fmt.Errorf("meta: %w", err)
	}

	return &metaModel{Model: model, efforts: offering.Reasoning.Efforts}, nil
}

type metaModel struct {
	llm.Model

	efforts []llm.ReasoningEffort
}

func (*metaModel) Provider() string { return "meta" }

func (m *metaModel) SupportedReasoningEfforts() []llm.ReasoningEffort {
	return slices.Clone(m.efforts)
}

var (
	_ catalog.Provider          = (*Provider)(nil)
	_ llm.Model                 = (*metaModel)(nil)
	_ llm.ReasoningEffortLister = (*metaModel)(nil)
)

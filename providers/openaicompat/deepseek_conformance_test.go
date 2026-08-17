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

package openaicompat_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/pricing"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// DeepSeekFixture implements the conformance.Fixture interface for DeepSeek API.
// This tests the openaicompat provider against DeepSeek's reasoning models.
type DeepSeekFixture struct {
	provider     *openaicompat.Provider
	capabilities llm.ModelCapabilities
	constraints  llm.ModelConstraints
}

// NewDeepSeekFixture creates a new DeepSeek test fixture.
func NewDeepSeekFixture(t *testing.T) *DeepSeekFixture {
	t.Helper()

	apiKey := openaicompattest.GetDeepSeekAPIKeyOrSkipTest(t)
	baseURL := openaicompattest.GetDeepSeekBaseURL()

	// Create provider with DeepSeek base URL and extended timeout for reasoning
	provider, err := openaicompat.NewProvider(
		apiKey,
		openaicompat.WithBaseURL(baseURL),
		openaicompat.WithTimeout(3*time.Minute),
	)
	if err != nil {
		t.Fatalf("Failed to create DeepSeek provider: %v", err)
	}

	return newDeepSeekFixture(provider)
}

func newDeepSeekFixture(provider *openaicompat.Provider) *DeepSeekFixture {
	return &DeepSeekFixture{
		provider: provider,
		capabilities: llm.ModelCapabilities{
			Streaming:     true,
			Tools:         true,
			JSONMode:      true,
			MultiTurn:     true,
			SystemPrompts: true,
			Reasoning:     true,
		},
		constraints: llm.ModelConstraints{
			TemperatureRange:  [2]float64{0.0, 2.0},
			MaxInputTokens:    1_000_000,
			MaxOutputTokens:   384_000,
			SupportedParams:   []string{"temperature", "top_p", "max_tokens", "logprobs", "stop"},
			MutuallyExclusive: [][]string{{"temperature", "top_p"}},
		},
	}
}

func (f *DeepSeekFixture) Name() string {
	return "DeepSeek"
}

func (f *DeepSeekFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.newModel(
		openaicompattest.DeepSeekDefaultStandardModel,
		openaicompat.WithThinking(false),
	)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *DeepSeekFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.newModel(
		openaicompattest.DeepSeekDefaultReasoningModel,
		openaicompat.WithThinking(true),
	)
	if err != nil {
		t.Fatalf("Failed to create reasoning model: %v", err)
	}

	return retry.WrapModel(model)
}

// Catalog builds a test-local catalog for the two DeepSeek presets: the
// openaicompat provider has no static catalog, so the fixture authors its
// own entries (with a fixture-local facts registry) purely for discovery
// testing.
func (f *DeepSeekFixture) Catalog() *catalog.Catalog {
	registry := catalog.Registry{
		"deepseek/v4-flash": {
			Name: "DeepSeek V4 Flash", Series: "deepseek-flash",
			Released: catalog.MustDate("2026-05-01"),
		},
		"deepseek/v4-pro": {
			Name: "DeepSeek V4 Pro", Series: "deepseek-pro",
			Released: catalog.MustDate("2026-05-01"),
		},
	}

	return catalog.MustNew("deepseek", []catalog.Entry{
		{
			ID:           openaicompattest.DeepSeekDefaultStandardModel,
			Model:        "deepseek/v4-flash",
			Label:        "DeepSeek V4 Flash",
			Capabilities: f.capabilities,
			Constraints:  f.constraints,
			Pricing:      pricing.FlatInfo(0.28, 0.42, 0.028),
		},
		{
			ID:           openaicompattest.DeepSeekDefaultReasoningModel,
			Model:        "deepseek/v4-pro",
			Label:        "DeepSeek V4 Pro",
			Capabilities: f.capabilities,
			Constraints:  f.constraints,
			Pricing:      pricing.FlatInfo(0.56, 1.68, 0.056),
		},
	}, catalog.WithRegistry(registry))
}

func (f *DeepSeekFixture) NewModel(modelName string) (llm.Model, error) {
	return f.newModel(modelName)
}

func (f *DeepSeekFixture) newModel(modelName string, opts ...openaicompat.Option) (llm.Model, error) {
	opts = append([]openaicompat.Option{
		openaicompat.WithConstraints(f.constraints),
		openaicompat.WithCapabilities(f.capabilities),
	}, opts...)

	return f.provider.NewModel(
		modelName,
		opts...,
	)
}

// TestDeepSeekConformance_Integration runs the generic conformance test suite against DeepSeek API.
//
// Set DEEPSEEK_API_KEY to run these tests:
//
//	DEEPSEEK_API_KEY=sk-xxx go test -v -run TestDeepSeekConformance_Integration
//
// Optional environment variables:
//
//	DEEPSEEK_BASE_URL - API base URL (default: https://api.deepseek.com)
func TestDeepSeekConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewDeepSeekFixture(t)
	testsuite.Run(t, conformance.NewSuite(fixture))
}

func TestDeepSeekV4ConformancePresets(t *testing.T) {
	t.Parallel()

	provider, err := openaicompat.NewProvider(
		"sk-test-key",
		openaicompat.WithBaseURL(openaicompattest.DeepSeekDefaultBaseURL),
	)
	require.NoError(t, err)

	fixture := newDeepSeekFixture(provider)
	wantConstraints := llm.ModelConstraints{
		TemperatureRange:  [2]float64{0.0, 2.0},
		MaxInputTokens:    1_000_000,
		MaxOutputTokens:   384_000,
		SupportedParams:   []string{"temperature", "top_p", "max_tokens", "logprobs", "stop"},
		MutuallyExclusive: [][]string{{"temperature", "top_p"}},
	}
	wantCapabilities := llm.ModelCapabilities{
		Streaming:     true,
		Tools:         true,
		JSONMode:      true,
		MultiTurn:     true,
		SystemPrompts: true,
		Reasoning:     true,
	}

	models := fixture.Catalog().All()
	require.Len(t, models, 2)
	require.Equal(t, []string{"deepseek-v4-flash", "deepseek-v4-pro"}, []string{models[0].ID, models[1].ID})

	for _, info := range models {
		assert.Equal(t, wantConstraints, info.Constraints)
		assert.Equal(t, wantCapabilities, info.Capabilities)

		model, err := fixture.NewModel(info.ID)
		require.NoError(t, err)
		assert.Equal(t, info.Constraints, model.Constraints())
		assert.Equal(t, info.Capabilities, model.Capabilities())
	}

	assert.Equal(t, "deepseek-v4-flash", fixture.NewStandardModel(t).Name())
	assert.Equal(t, "deepseek-v4-pro", fixture.NewReasoningModel(t).Name())
}

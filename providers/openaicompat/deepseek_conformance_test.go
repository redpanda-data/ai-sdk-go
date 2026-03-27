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

	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// DeepSeekFixture implements the conformance.Fixture interface for DeepSeek API.
// This tests the openaicompat provider against DeepSeek's reasoning models.
type DeepSeekFixture struct {
	provider      *openaicompat.Provider
	standardCaps  llm.ModelCapabilities
	reasoningCaps llm.ModelCapabilities
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

	// DeepSeek-specific capabilities
	// DeepSeek supports JSON mode (json_object) but not Structured Outputs (json_schema)
	deepseekCaps := llm.ModelCapabilities{
		Streaming:        true,
		Tools:            true,
		JSONMode:         true,  // Supports json_object
		StructuredOutput: false, // Does NOT support json_schema
		Vision:           true,
		Audio:            false,
		MultiTurn:        true,
		SystemPrompts:    true,
		Reasoning:        false, // Set per-model below
	}

	reasoningCaps := deepseekCaps
	reasoningCaps.Reasoning = true

	return &DeepSeekFixture{
		provider:      provider,
		standardCaps:  deepseekCaps,
		reasoningCaps: reasoningCaps,
	}
}

func (f *DeepSeekFixture) Name() string {
	return "DeepSeek"
}

func (f *DeepSeekFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(
		openaicompattest.DeepSeekDefaultStandardModel,
		openaicompat.WithCapabilities(f.standardCaps),
	)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *DeepSeekFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(
		openaicompattest.DeepSeekDefaultReasoningModel,
		openaicompat.WithCapabilities(f.reasoningCaps),
	)
	if err != nil {
		t.Fatalf("Failed to create reasoning model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *DeepSeekFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *DeepSeekFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
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

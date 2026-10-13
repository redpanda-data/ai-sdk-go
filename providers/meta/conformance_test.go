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

package meta_test

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	agentconformance "github.com/redpanda-data/ai-sdk-go/agent/conformance"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

type metaFixture struct {
	provider  *meta.Provider
	modelName string
}

func TestMetaFixtureModels(t *testing.T) {
	t.Parallel()

	provider, err := meta.NewProvider("test-key")
	require.NoError(t, err)

	for _, offering := range meta.Catalog().All() {
		t.Run(offering.ID, func(t *testing.T) {
			t.Parallel()

			fixture := &metaFixture{provider: provider, modelName: offering.ID}
			assert.Equal(t, offering.ID, fixture.NewStandardModel(t).Name())
			assert.Equal(t, offering.ID, fixture.NewReasoningModel(t).Name())

			standardAgent, err := fixture.StandardAgent(nil)
			require.NoError(t, err)
			assert.Equal(t, offering.ID, standardAgent.Info().ModelName)

			reasoningAgent, err := fixture.ReasoningAgent(nil)
			require.NoError(t, err)
			assert.Equal(t, offering.ID, reasoningAgent.Info().ModelName)
		})
	}
}

func newMetaFixture(t *testing.T, modelName string) *metaFixture {
	t.Helper()

	if testing.Short() {
		t.Skip("live Meta conformance")
	}

	key := os.Getenv("MODEL_API_KEY")
	if key == "" {
		t.Skip("MODEL_API_KEY is not set")
	}

	provider, err := meta.NewProvider(key, openai.WithTimeout(2*time.Minute))
	require.NoError(t, err)

	return &metaFixture{provider: provider, modelName: modelName}
}
func (f *metaFixture) Name() string                            { return "Meta" }
func (f *metaFixture) Catalog() *catalog.Catalog               { return f.provider.Catalog() }
func (f *metaFixture) NewModel(name string) (llm.Model, error) { return f.provider.NewModel(name) }
func (f *metaFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(f.modelName, openai.WithReasoningEffort(openai.ReasoningEffortMinimal))
	require.NoError(t, err)

	return retry.WrapModel(model)
}

func (f *metaFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(f.modelName, openai.WithReasoningEffort(openai.ReasoningEffortHigh))
	require.NoError(t, err)

	return retry.WrapModel(model)
}

func (f *metaFixture) StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(f.modelName, openai.WithReasoningEffort(openai.ReasoningEffortMinimal))
	if err != nil {
		return nil, err
	}

	return llmagent.New("test-agent", "You are a helpful assistant. Use available tools to answer questions.", model, llmagent.WithTools(tools), llmagent.WithMaxTurns(10))
}

func (f *metaFixture) ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error) {
	model, err := f.provider.NewModel(f.modelName, openai.WithReasoningEffort(openai.ReasoningEffortHigh))
	if err != nil {
		return nil, err
	}

	return llmagent.New("reasoning-agent", "You are a helpful assistant with reasoning capabilities.", model, llmagent.WithTools(tools), llmagent.WithMaxTurns(10))
}

func TestMetaConformance_Integration(t *testing.T) {
	t.Parallel()

	for _, offering := range meta.Catalog().All() {
		t.Run(offering.ID, func(t *testing.T) {
			t.Parallel()
			testsuite.Run(t, conformance.NewSuite(newMetaFixture(t, offering.ID)))
		})
	}
}

func TestMetaAgentConformance_Integration(t *testing.T) {
	t.Parallel()

	for _, offering := range meta.Catalog().All() {
		t.Run(offering.ID, func(t *testing.T) {
			t.Parallel()
			testsuite.Run(t, agentconformance.NewSuite(newMetaFixture(t, offering.ID)))
		})
	}
}

var (
	_ conformance.Fixture      = (*metaFixture)(nil)
	_ agentconformance.Fixture = (*metaFixture)(nil)
)

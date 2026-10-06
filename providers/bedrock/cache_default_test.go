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
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestDefaultProviderEnablesCaching pins the default: Bedrock runs Claude, and
// prompt caching being opt-in means consumers forget to opt in. A provider
// built the normal way (no caching option) must have caching ON.
func TestDefaultProviderEnablesCaching(t *testing.T) {
	t.Parallel()

	// WithAWSConfig bypasses the AWS credential/config-file lookup so this runs
	// fully offline.
	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)

	assert.True(t, p.enableCaching,
		"Bedrock caching must default to ON; consumers never call WithCaching()")
}

// TestDefaultCachingInsertsCachePoint proves the default flows through to the
// wire: a model built from a default provider must insert a CachePoint after
// the system block without anyone calling WithCaching().
func TestDefaultCachingInsertsCachePoint(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleSystem,
				Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
			},
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart("Hello")},
			},
		},
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)

	assert.True(t, hasSystemCachePoint(input.System),
		"system blocks must carry a CachePoint by default")
}

// TestWithCachingIsNoOp pins the backward-compatibility contract: WithCaching()
// stays callable with no arguments and leaves caching on.
func TestWithCachingIsNoOp(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(),
		WithAWSConfig(aws.Config{Region: "us-east-1"}), WithCaching())
	require.NoError(t, err)

	assert.True(t, p.enableCaching, "WithCaching() must remain a valid no-op with caching on")
}

// TestWithCachingDisabledSuppressesCachePoint proves the opt-out works end to
// end: no CachePoint inserted into the system blocks.
func TestWithCachingDisabledSuppressesCachePoint(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(),
		WithAWSConfig(aws.Config{Region: "us-east-1"}), WithCachingDisabled())
	require.NoError(t, err)
	require.False(t, p.enableCaching, "WithCachingDisabled must turn caching off")

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleSystem,
				Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
			},
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart("Hello")},
			},
		},
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)

	assert.False(t, hasSystemCachePoint(input.System),
		"WithCachingDisabled must suppress the CachePoint")
}

// hasSystemCachePoint reports whether the system blocks contain a cache point.
func hasSystemCachePoint(blocks []types.SystemContentBlock) bool {
	for _, b := range blocks {
		if _, ok := b.(*types.SystemContentBlockMemberCachePoint); ok {
			return true
		}
	}

	return false
}

// TestNewModel_NoCachePointFamiliesDisableCaching checks that caching-by-default
// is overridden for families whose Converse endpoint rejects CachePoint blocks.
func TestNewModel_NoCachePointFamiliesDisableCaching(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)

	for id, wantCaching := range map[string]bool{
		ModelMistralLarge3:    false,
		ModelGPT6AstraUS:      false,
		ModelGPT6AstraGlobal:  false,
		ModelGPT61SolUS:       false,
		ModelGPT61SolGlobal:   false,
		ModelGPT6SolUS:        false,
		ModelGPT6SolGlobal:    false,
		ModelClaudeSonnet45US: true,
		ModelClaudeOpus55US:   true,
	} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			m, err := provider.NewModel(id)
			require.NoError(t, err)

			bm, ok := m.(*Model)
			require.True(t, ok)
			assert.Equal(t, wantCaching, bm.config.EnableCaching)
		})
	}
}

// TestNoCachePointModelIDs checks that a NoCachePoints family opts out every
// profile ID but its bare ID only when invokable, so the flag does not leak
// onto a mantle family that shares the bare ID.
func TestNoCachePointModelIDs(t *testing.T) {
	t.Parallel()

	for id, want := range map[string]bool{
		ModelMistralLarge3:   true,
		ModelGPT6AstraUS:     true,
		ModelGPT6AstraGlobal: true,
		ModelGPT61SolUS:      true,
		ModelGPT61SolGlobal:  true,
		ModelGPT6Astra:       false,
		ModelGPT61Sol:        false,
	} {
		assert.Equal(t, want, noCachePointModelIDs[id], id)
	}
}

// TestGPT6RoutesByID checks the two surfaces of each GPT-6 model on
// Bedrock: the bare ID is a mantle (Responses) model with effort control,
// while the profiles run on Converse, which offers none.
func TestGPT6RoutesByID(t *testing.T) {
	t.Parallel()

	lowToMax := []ReasoningEffort{ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh, ReasoningEffortMax}

	for _, tt := range []struct {
		bare, us, global string
		efforts          []ReasoningEffort
	}{
		{ModelGPT6Astra, ModelGPT6AstraUS, ModelGPT6AstraGlobal, lowToMax},
		{ModelGPT61Sol, ModelGPT61SolUS, ModelGPT61SolGlobal, lowToMax},
		{ModelGPT6Sol, ModelGPT6SolUS, ModelGPT6SolGlobal, append([]ReasoningEffort{ReasoningEffortNone}, lowToMax...)},
	} {
		t.Run(tt.bare, func(t *testing.T) {
			t.Parallel()

			assert.True(t, IsMantleModel(tt.bare))
			assert.False(t, IsMantleModel(tt.us))
			assert.False(t, IsMantleModel(tt.global))

			bare, ok := Catalog().Lookup(tt.bare)
			require.True(t, ok)
			assert.Equal(t, tt.efforts, bare.Reasoning.Efforts)

			for _, id := range []string{tt.us, tt.global} {
				profile, ok := Catalog().Lookup(id)
				require.True(t, ok)
				assert.Empty(t, profile.Reasoning.Efforts)
			}
		})
	}
}

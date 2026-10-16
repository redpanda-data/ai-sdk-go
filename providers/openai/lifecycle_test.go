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

package openai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func TestDatedSnapshotLifecycle(t *testing.T) {
	t.Parallel()

	const (
		deprecated = "2026-10-14"
		retires    = "2027-01-14"
	)
	for _, tt := range []struct{ id, deprecated, retires, replacement string }{
		{ModelGPT4O20240513, "2026-04-22", "2026-10-23", ModelGPT5_6Sol},
		{ModelGPT4O20240806, deprecated, retires, ModelGPT6Luna},
		{ModelGPT4O20241120, deprecated, retires, ModelGPT6Luna},
		{ModelGPT4OMini20240718, deprecated, retires, ModelGPT6Luna},
		{ModelGPT41Snapshot20250414, deprecated, retires, ModelGPT6Luna},
		{ModelGPT41MiniSnapshot20250414, deprecated, retires, ModelGPT6Luna},
	} {
		t.Run(tt.id, func(t *testing.T) {
			t.Parallel()

			o, ok := Catalog().Resolve(tt.id)
			require.True(t, ok)
			assert.Equal(t, tt.id, o.ID, "snapshot-specific metadata must not resolve to a movable alias")
			assert.Equal(t, catalog.MustDate(tt.deprecated), o.Life.Deprecated)
			assert.Equal(t, catalog.MustDate(tt.retires), o.Life.Retires)
			assert.Equal(t, tt.replacement, o.Life.ReplacedBy)
			stage, ok := Catalog().At(catalog.MustDate(tt.retires)).Stage(tt.id)
			require.True(t, ok)
			assert.Equal(t, catalog.StageRetired, stage)
		})
	}

	for _, id := range []string{ModelGPT4O, ModelGPT4OMini, ModelGPT41, ModelGPT41Mini} {
		o, ok := Catalog().Lookup(id)
		require.True(t, ok)
		assert.True(t, o.Life.Retires.IsZero(), "alias retirement is not derived from registered snapshots")
	}
}

func TestOriginalGPT4oSnapshotDoesNotInheritAliasPricing(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)
	cost, err := prices.Calculate(ProviderName, ModelGPT4O20240513, &llm.TokenUsage{InputTokens: 1_000_000, OutputTokens: 1_000_000}, pricing.CalcRequest{})
	require.NoError(t, err)
	assert.Equal(t, int64(2_000_000_000), cost.Total) // $5 input + $15 output

	o, ok := Catalog().Lookup(ModelGPT4O20240513)
	require.True(t, ok)
	assert.False(t, o.Capabilities.StructuredOutput)
	assert.Equal(t, 4096, o.Constraints.MaxOutputTokens)
}

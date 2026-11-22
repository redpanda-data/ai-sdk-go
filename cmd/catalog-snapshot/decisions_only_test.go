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

package main

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// TestEveryDecisionsOnlyOfferingIsRefused walks every catalogue: NewModel
// must refuse each decisions-only offering with catalog.ErrDecisionsOnly,
// which agent admission relies on. Each provider package also feeds its
// constructor a synthetic decisions-only offering.
func TestEveryDecisionsOnlyOfferingIsRefused(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	anthropicProvider, err := anthropic.NewProvider("placeholder")
	require.NoError(t, err)
	bedrockProvider, err := bedrock.NewProvider(ctx, bedrock.WithRegion("us-east-1"))
	require.NoError(t, err)
	fireworksProvider, err := fireworks.NewProvider(ctx, fireworks.WithAPIKey("placeholder"))
	require.NoError(t, err)
	googleProvider, err := google.NewProvider(ctx, "placeholder")
	require.NoError(t, err)
	metaProvider, err := meta.NewProvider("placeholder")
	require.NoError(t, err)
	openaiProvider, err := openai.NewProvider("placeholder")
	require.NoError(t, err)
	vertexProvider, err := vertex.NewProvider(ctx)
	require.NoError(t, err)

	newModel := map[llm.ProviderID]func(string) error{
		anthropic.ProviderName: func(id string) error { _, err := anthropicProvider.NewModel(id); return err },
		bedrock.ProviderName:   func(id string) error { _, err := bedrockProvider.NewModel(id); return err },
		fireworks.ProviderName: func(id string) error { _, err := fireworksProvider.NewModel(id); return err },
		google.ProviderName:    func(id string) error { _, err := googleProvider.NewModel(id); return err },
		meta.ProviderName:      func(id string) error { _, err := metaProvider.NewModel(id); return err },
		openai.ProviderName:    func(id string) error { _, err := openaiProvider.NewModel(id); return err },
		vertex.ProviderName:    func(id string) error { _, err := vertexProvider.NewModel(id); return err },
	}

	var refused []string

	for _, cat := range allCatalogs() {
		build, ok := newModel[cat.Provider()]
		require.True(t, ok, "no constructor for %s", cat.Provider())

		for _, o := range cat.All() {
			if !o.DecisionsOnly() {
				continue
			}

			require.ErrorIs(t, build(o.ID), catalog.ErrDecisionsOnly, "%s/%s", cat.Provider(), o.ID)
			refused = append(refused, string(cat.Provider())+"/"+o.ID)
		}
	}

	assert.ElementsMatch(t, []string{"gcp.gemini/" + google.ModelGeminiDecision1, "fireworks/" + fireworks.ModelRune}, refused)
}

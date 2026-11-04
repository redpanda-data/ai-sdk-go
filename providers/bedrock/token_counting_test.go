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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func TestCatalogTokenCounting(t *testing.T) {
	t.Parallel()

	invoke := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockInvokeModel}
	converse := catalog.WireAPI{Protocol: catalog.WireProtocolBedrockConverse, Transport: catalog.WireTransportBedrockRuntime}
	mantle := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockMantle}

	for _, tt := range []struct {
		name string
		ids  []string
		want []catalog.WireAPI
	}{
		{
			name: "cards that count only on bedrock-mantle",
			ids: []string{
				ModelClaudeFable51US, ModelClaudeFable51Global, ModelClaudeFable5US, ModelClaudeFable5Global,
				ModelClaudeOpus55US, ModelClaudeOpus55JP, ModelClaudeOpus5AU, ModelClaudeOpus5Global,
				ModelClaudeOpus48EU, ModelClaudeOpus47AU, ModelClaudeSonnet55EU, ModelClaudeSonnet5Global,
				ModelClaudeHaiku55JP,
			},
			want: []catalog.WireAPI{mantle},
		},
		{
			name: "cards that count only on bedrock-runtime",
			ids: []string{
				ModelClaudeOpus46US, ModelClaudeOpus46Global, ModelClaudeOpus45EU,
				ModelClaudeSonnet46JP, ModelClaudeSonnet45AU,
			},
			want: []catalog.WireAPI{invoke, converse},
		},
		{
			name: "Haiku 4.5 counts on both endpoints",
			ids:  []string{ModelClaudeHaiku45US, ModelClaudeHaiku45Global, ModelClaudeHaiku45JP},
			want: []catalog.WireAPI{invoke, mantle, converse},
		},
		{
			name: "Fable 5 EU is unknown",
			ids:  []string{ModelClaudeFable5EU},
			want: nil,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			for _, id := range tt.ids {
				o, ok := Catalog().Lookup(id)
				require.True(t, ok, id)
				assert.ElementsMatch(t, tt.want, o.TokenCounting, id)
			}
		})
	}
}

func TestCatalogTokenCountingCoversEveryClaudeOffering(t *testing.T) {
	t.Parallel()

	var unknownClaude []string

	for _, o := range Catalog().All() {
		isClaude := strings.Contains(o.ID, "anthropic.claude-")
		if !isClaude {
			assert.Empty(t, o.TokenCounting, "%s: only Claude is populated in this change", o.ID)

			continue
		}

		if len(o.TokenCounting) == 0 {
			unknownClaude = append(unknownClaude, o.ID)
		}
	}

	assert.Equal(t, []string{ModelClaudeFable5EU}, unknownClaude)
}

func TestExpandFamiliesTokenCountingOverrides(t *testing.T) {
	t.Parallel()

	mantle := catalog.WireAPI{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportBedrockMantle}
	apis := []catalog.WireAPI{mantle}
	entries, _ := expandFamilies([]family{{
		BareID:            "anthropic.example",
		Profiles:          []string{"us", "eu"},
		TokenCounting:     apis,
		TokenCountingByID: map[string][]catalog.WireAPI{"eu.anthropic.example": nil},
	}})
	require.Len(t, entries, 2)
	assert.Equal(t, apis, entries[0].TokenCounting)
	assert.Empty(t, entries[1].TokenCounting)

	entries[0].TokenCounting[0].Transport = catalog.WireTransportBedrockRuntime

	assert.Equal(t, []catalog.WireAPI{mantle}, apis, "expansion must copy the family's slice")
}

func TestExpandFamiliesRejectsUnregisteredTokenCountingOverrides(t *testing.T) {
	t.Parallel()

	for _, id := range []string{"anthropic.example", "jp.anthropic.example", "eu.anthropic.other", ""} {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			f := family{
				BareID:            "anthropic.example",
				Profiles:          []string{"us", "eu"},
				TokenCountingByID: map[string][]catalog.WireAPI{id: nil},
			}
			assert.PanicsWithValue(t,
				"bedrock: family anthropic.example overrides TokenCounting for unregistered offering "+id,
				func() { expandFamilies([]family{f}) })
		})
	}
}

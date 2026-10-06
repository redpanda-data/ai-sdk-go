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
	"encoding/json"
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

var (
	batch = pricing.Selector{ServiceTier: llm.ServiceTierBatch}
	flex  = pricing.Selector{ServiceTier: llm.ServiceTierFlex}
	fast  = pricing.Selector{ServiceTier: llm.ServiceTierPriority}
)

// TestWithServiceTierMapsToRequest checks that service_tier reaches the
// Responses request only when the option is set, normalized to OpenAI's
// wire value.
func TestWithServiceTierMapsToRequest(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	for _, tt := range []struct {
		name string
		opts []Option
		want string // "" means the field is absent from the request body
	}{
		{"unset", nil, ""},
		{"default", []Option{WithServiceTier(llm.ServiceTierDefault)}, "default"},
		{"flex", []Option{WithServiceTier(llm.ServiceTierFlex)}, "flex"},
		{"priority", []Option{WithServiceTier(llm.ServiceTierPriority)}, "priority"},
		{"fast normalizes to priority", []Option{WithServiceTier("fast")}, "priority"},
		{"scale", []Option{WithServiceTier(llm.ServiceTierScale)}, "scale"},
		{"casing and spacing", []Option{WithServiceTier(" FLEX ")}, "flex"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, err := provider.NewModel(ModelGPT6_1Sol, tt.opts...)
			require.NoError(t, err)

			concrete, ok := model.(*Model)
			require.True(t, ok)

			apiReq, err := concrete.requestMapper.ToProvider(&llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}})
			require.NoError(t, err)
			assert.Equal(t, tt.want, string(apiReq.ServiceTier))

			body, err := json.Marshal(apiReq)
			require.NoError(t, err)

			var fields map[string]any
			require.NoError(t, json.Unmarshal(body, &fields))

			got, present := fields["service_tier"]
			if tt.want == "" {
				assert.False(t, present, "service_tier must be omitted when unset, got %v", got)
				return
			}

			assert.Equal(t, tt.want, got)
		})
	}
}

// TestWithServiceTierRejectsNonRequestTiers checks that tiers the Responses
// API cannot be asked for fail at model construction.
func TestWithServiceTierRejectsNonRequestTiers(t *testing.T) {
	t.Parallel()

	provider, err := NewProvider("sk-test-key")
	require.NoError(t, err)

	for _, tier := range []llm.ServiceTier{"", " ", "auto", " AUTO ", llm.ServiceTierBatch, llm.ServiceTierReserved, llm.ServiceTierProvisionedThroughput} {
		t.Run(string(tier), func(t *testing.T) {
			t.Parallel()

			_, err := provider.NewModel(ModelGPT6_1Sol, WithServiceTier(tier))
			require.ErrorContains(t, err, "service tier")
		})
	}
}

// TestServiceTierPricesFromResponse drives the reported tier through the
// response mapper and SelectorFromResponse into the catalog: OpenAI reports
// Fast mode as "priority" for GPT-5.6 and earlier and may report "fast" for
// later models, and both must reach the Fast card.
func TestServiceTierPricesFromResponse(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	for _, tt := range []struct {
		name          string
		model         string
		reported      responses.ResponseServiceTier
		wantTier      llm.ServiceTier
		input, output int64
	}{
		{"GPT-6.1 Sol fast", ModelGPT6_1Sol, "fast", llm.ServiceTierPriority, 400_000, 2_000_000},
		{"GPT-6.1 Sol priority", ModelGPT6_1Sol, responses.ResponseServiceTierPriority, llm.ServiceTierPriority, 400_000, 2_000_000},
		{"GPT-5.6 Sol priority", ModelGPT5_6Sol, responses.ResponseServiceTierPriority, llm.ServiceTierPriority, 800_000, 4_000_000},
		{"GPT-6 Astra flex", ModelGPT6Astra, responses.ResponseServiceTierFlex, llm.ServiceTierFlex, 500_000, 2_500_000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			resp, err := NewResponseMapper().FromProvider(&responses.Response{
				ID:          "resp_tier",
				Model:       tt.model,
				Status:      responses.ResponseStatusCompleted,
				ServiceTier: tt.reported,
				Output: []responses.ResponseOutputItemUnion{{
					Type:    outputTypeMessage,
					Content: []responses.ResponseOutputMessageContentUnion{{Type: contentTypeOutputText, Text: "hi"}},
				}},
				Usage: responses.ResponseUsage{InputTokens: 1_000, OutputTokens: 1_000, TotalTokens: 2_000},
			})
			require.NoError(t, err)
			assert.Equal(t, tt.wantTier, resp.ServiceTier)

			cost, err := prices.Calculate(ProviderName, resp.InvokedModelID, resp.Usage, pricing.CalcRequest{Selector: pricing.SelectorFromResponse(resp)})
			require.NoError(t, err)
			assert.Equal(t, pricing.Selector{ServiceTier: tt.wantTier}, cost.AppliedSelector)
			assert.Empty(t, cost.Fallbacks)
			assert.Equal(t, tt.input, cost.Breakdown[pricing.UsageFieldInput])
			assert.Equal(t, tt.output, cost.Breakdown[pricing.UsageFieldOutput])
		})
	}
}

// TestServiceTierPricing pins Batch, Flex, and Fast rates from
// developers.openai.com/api/docs/pricing for a representative set, at and
// above the 272K long-context threshold. Every bucket is 1M tokens, so each
// expected amount is the published USD/M rate in microcents; 0 marks a rate
// the page leaves unpublished ("-"), which must surface as unpriced.
func TestServiceTierPricing(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 1_000_000, CachedInputTokens: 1_000_000, CacheCreationUnknownTTLTokens: 1_000_000, OutputTokens: 1_000_000}

	const short, long = 272_000, 272_001

	for _, tt := range []struct {
		name                         string
		model                        string
		tier                         pricing.Selector
		context, wantBracket         int64
		input, cached, write, output int64
	}{
		{"Astra batch", ModelGPT6Astra, batch, short, 0, 500_000_000, 50_000_000, 625_000_000, 2_500_000_000},
		{"Astra batch long", ModelGPT6Astra, batch, long, long, 1_000_000_000, 100_000_000, 1_250_000_000, 3_750_000_000},
		{"Astra flex", ModelGPT6Astra, flex, short, 0, 500_000_000, 50_000_000, 625_000_000, 2_500_000_000},
		{"Astra flex long", ModelGPT6Astra, flex, long, long, 1_000_000_000, 100_000_000, 1_250_000_000, 3_750_000_000},
		{"Astra fast", ModelGPT6Astra, fast, short, 0, 2_000_000_000, 200_000_000, 2_500_000_000, 10_000_000_000},
		{"Astra fast long", ModelGPT6Astra, fast, long, long, 4_000_000_000, 400_000_000, 5_000_000_000, 15_000_000_000},

		{"6.1 Sol batch", ModelGPT6_1Sol, batch, short, 0, 100_000_000, 5_000_000, 125_000_000, 500_000_000},
		{"6.1 Sol batch long", ModelGPT6_1Sol, batch, long, long, 200_000_000, 10_000_000, 250_000_000, 750_000_000},
		{"6.1 Sol flex long", ModelGPT6_1Sol, flex, long, long, 200_000_000, 10_000_000, 250_000_000, 750_000_000},
		{"6.1 Sol fast", ModelGPT6_1Sol, fast, short, 0, 400_000_000, 20_000_000, 500_000_000, 2_000_000_000},
		{"6.1 Sol fast long", ModelGPT6_1Sol, fast, long, long, 800_000_000, 40_000_000, 1_000_000_000, 3_000_000_000},

		{"5.5 batch", ModelGPT5_5, batch, short, 0, 250_000_000, 25_000_000, 0, 1_500_000_000},
		{"5.5 batch long", ModelGPT5_5, batch, long, long, 500_000_000, 50_000_000, 0, 2_250_000_000},
		{"5.5 flex long", ModelGPT5_5, flex, long, long, 500_000_000, 50_000_000, 0, 2_250_000_000},
		{"5.5 fast", ModelGPT5_5, fast, short, 0, 1_250_000_000, 125_000_000, 0, 7_500_000_000},
		// No long-context Fast rate is published for gpt-5.5 or gpt-5.4, so a
		// long Fast request bills at the flat Fast rate, never at zero.
		{"5.5 fast long", ModelGPT5_5, fast, long, 0, 1_250_000_000, 125_000_000, 0, 7_500_000_000},
		{"5.4 fast long", ModelGPT5_4, fast, long, 0, 500_000_000, 50_000_000, 0, 3_000_000_000},

		{"5.4 mini batch", ModelGPT5_4Mini, batch, short, 0, 37_500_000, 3_750_000, 0, 225_000_000},
		{"5.4 mini flex", ModelGPT5_4Mini, flex, short, 0, 37_500_000, 3_750_000, 0, 225_000_000},
		{"5.4 mini fast", ModelGPT5_4Mini, fast, short, 0, 150_000_000, 15_000_000, 0, 900_000_000},

		{"4o batch", ModelGPT4O, batch, 100_000, 0, 125_000_000, 0, 0, 500_000_000},
		{"4o fast", ModelGPT4O, fast, 100_000, 0, 425_000_000, 212_500_000, 0, 1_700_000_000},

		{"o3 batch", ModelO3, batch, 100_000, 0, 100_000_000, 0, 0, 400_000_000},
		{"o3 flex", ModelO3, flex, 100_000, 0, 100_000_000, 25_000_000, 0, 400_000_000},
		{"o3 fast", ModelO3, fast, 100_000, 0, 350_000_000, 87_500_000, 0, 1_400_000_000},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cost, err := prices.Calculate(ProviderName, tt.model, usage, pricing.CalcRequest{Selector: tt.tier, ContextTokens: tt.context})
			require.NoError(t, err)
			assert.Equal(t, tt.tier, cost.AppliedSelector, "the tier card must match its own selector")
			assert.Empty(t, cost.Fallbacks)
			assert.Equal(t, tt.wantBracket, cost.AppliedBracketMinContextTokens)

			for field, want := range map[pricing.UsageField]int64{
				pricing.UsageFieldInput:                   tt.input,
				pricing.UsageFieldCachedInput:             tt.cached,
				pricing.UsageFieldCacheCreationUnknownTTL: tt.write,
				pricing.UsageFieldOutput:                  tt.output,
			} {
				if want == 0 {
					assert.Contains(t, cost.Unpriced, field)
					continue
				}

				assert.Equal(t, want, cost.Breakdown[field], field)
			}
		})
	}
}

// TestServiceTierPricingOmitsUnlistedTiers checks that a tier the pricing
// page has no row for falls back to Standard rates rather than borrowing a
// neighbouring row: gpt-4o has no Flex row, nano models have no Fast row,
// Batch lists only dated gpt-3.5-turbo snapshots, and the retired chat
// models appear in no tier table.
func TestServiceTierPricingOmitsUnlistedTiers(t *testing.T) {
	t.Parallel()

	prices, err := pricing.NewCatalog(pricing.WithSource(Catalog()))
	require.NoError(t, err)

	usage := &llm.TokenUsage{InputTokens: 1_000_000}

	for _, tt := range []struct {
		model string
		tier  pricing.Selector
		input int64
	}{
		{ModelGPT4O, flex, 250_000_000},
		{ModelGPT41, flex, 200_000_000},
		{ModelGPT5Nano, fast, 5_000_000},
		{ModelGPT5_4Nano, fast, 20_000_000},
		{ModelGPT5_2Pro, flex, 2_100_000_000},
		{ModelGPT35Turbo, batch, 50_000_000},
		{ModelGPT5_2Instant, batch, 175_000_000},
		{ModelGPT5_3ChatLatest, fast, 175_000_000},
	} {
		t.Run(tt.model+"/"+string(tt.tier.ServiceTier), func(t *testing.T) {
			t.Parallel()

			cost, err := prices.Calculate(ProviderName, tt.model, usage, pricing.CalcRequest{Selector: tt.tier})
			require.NoError(t, err)
			assert.True(t, cost.AppliedSelector.IsZero(), "no %s card is published", tt.tier.ServiceTier)
			assert.Equal(t, tt.input, cost.Breakdown[pricing.UsageFieldInput])
		})
	}
}

// TestServiceTierCardsMirrorDefaultBrackets guards the rule that an override
// card carries its own brackets: a tier card on a context-tiered model must
// switch at the same thresholds, or large tier requests bill at short-context
// rates; a tier card on a flat model must stay flat. The exceptions are the
// Fast cards of gpt-5.5 and gpt-5.4, which are flat because OpenAI publishes
// no long-context Fast rate for them.
func TestServiceTierCardsMirrorDefaultBrackets(t *testing.T) {
	t.Parallel()

	flatByDesign := map[string]bool{ModelGPT5_5: true, ModelGPT5_4: true}

	thresholds := func(card pricing.RateCard) []int64 {
		out := make([]int64, 0, len(card.Brackets))
		for _, b := range card.Brackets {
			out = append(out, b.MinContextTokens)
		}

		return out
	}

	for _, o := range Catalog().All() {
		for _, ov := range o.Pricing.Overrides {
			want := thresholds(o.Pricing.Default)
			if flatByDesign[o.ID] && ov.Match == fast {
				want = []int64{}
			}

			assert.Equal(t, want, thresholds(ov.RateCard),
				"%s %s card brackets", o.ID, ov.Match.ServiceTier)
		}
	}
}

// TestServiceTierCardsAreOrdered is a transcription tripwire: the pricing
// page orders columns input, cached, cache writes, output, while
// pricing.NewRates takes input, output, cached. On every published card
// cached input is cheaper than input, which is cheaper than output; Batch
// and Flex undercut Standard and Fast costs more.
func TestServiceTierCardsAreOrdered(t *testing.T) {
	t.Parallel()

	for _, o := range Catalog().All() {
		for _, ov := range o.Pricing.Overrides {
			cards := []pricing.Rates{ov.RateCard.Base}
			defaults := []pricing.Rates{o.Pricing.Default.Base}

			// TestServiceTierCardsMirrorDefaultBrackets pins the bracket
			// thresholds; pair only the brackets both cards have.
			for i, b := range ov.RateCard.Brackets {
				if i < len(o.Pricing.Default.Brackets) {
					cards = append(cards, b.Rates)
					defaults = append(defaults, o.Pricing.Default.Brackets[i].Rates)
				}
			}

			for i, r := range cards {
				if r.InputPerMillion == 0 {
					continue // unpublished long-context rates
				}

				name := o.ID + " " + string(ov.Match.ServiceTier)
				assert.Less(t, r.InputPerMillion, r.OutputPerMillion, name)

				if r.CachedInputPerMillion != 0 {
					assert.Less(t, r.CachedInputPerMillion, r.InputPerMillion, name)
				}

				switch ov.Match {
				case batch, flex:
					assert.Less(t, r.InputPerMillion, defaults[i].InputPerMillion, name)
					assert.Less(t, r.OutputPerMillion, defaults[i].OutputPerMillion, name)
				case fast:
					assert.Greater(t, r.InputPerMillion, defaults[i].InputPerMillion, name)
					assert.Greater(t, r.OutputPerMillion, defaults[i].OutputPerMillion, name)
				}
			}
		}
	}
}

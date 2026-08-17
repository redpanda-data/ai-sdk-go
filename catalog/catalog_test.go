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

package catalog

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

const mutated = "mutated"

// The test registry deliberately does not reuse the built-in one: these
// tests pin the mechanics, not the shipped data.
func testRegistry() Registry {
	return Registry{
		"acme/robin-1": {
			Name: "Robin 1", Series: "robin",
			Released: MustDate("2025-01-10"),
		},
		"acme/robin-2": {
			Name: "Robin 2", Series: "robin",
			Released: MustDate("2025-08-01"), Knowledge: MustDate("2025-05-31"),
		},
		"acme/robin-3": {
			Name: "Robin 3", Series: "robin",
			Released: MustDate("2026-03-15"),
		},
		"acme/wren-1": {
			Name: "Wren 1", Series: "wren",
			Released: MustDate("2025-06-01"),
		},
	}
}

func validEntry(id string, model ModelID) Entry {
	return Entry{
		ID:    id,
		Model: model,
		Capabilities: llm.ModelCapabilities{
			Streaming: true, Tools: true, MultiTurn: true, SystemPrompts: true,
		},
		Constraints: llm.ModelConstraints{
			MaxInputTokens:  200_000,
			MaxOutputTokens: 8_192,
			SupportedParams: []string{"max_tokens", "temperature"},
		},
		Pricing: pricing.FlatInfo(3.00, 15.00, 0.30),
	}
}

func mustCatalog(t *testing.T, entries ...Entry) *Catalog {
	t.Helper()

	c, err := New("acme", entries, WithRegistry(testRegistry()))
	require.NoError(t, err)

	return c
}

func TestNewValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mutate  func(*Entry)
		wantErr string
	}{
		{
			name:    "missing ID",
			mutate:  func(e *Entry) { e.ID = "" },
			wantErr: `entries[0] "": ID is required`,
		},
		{
			name:    "unregistered model",
			mutate:  func(e *Entry) { e.Model = "acme/unknown" },
			wantErr: `entries[0] "robin-2": Model "acme/unknown" is not in the Facts registry`,
		},
		{
			name:    "missing model",
			mutate:  func(e *Entry) { e.Model = "" },
			wantErr: `Model is required`,
		},
		{
			name:    "zero max input",
			mutate:  func(e *Entry) { e.Constraints.MaxInputTokens = 0 },
			wantErr: `Constraints.MaxInputTokens must be > 0`,
		},
		{
			name:    "zero max output",
			mutate:  func(e *Entry) { e.Constraints.MaxOutputTokens = 0 },
			wantErr: `Constraints.MaxOutputTokens must be > 0`,
		},
		{
			name:    "authored deprecated stage",
			mutate:  func(e *Entry) { e.Life.Stage = StageDeprecated },
			wantErr: `Life.Stage "deprecated" is derived from dates and cannot be authored`,
		},
		{
			name:    "authored retired stage",
			mutate:  func(e *Entry) { e.Life.Stage = StageRetired },
			wantErr: `Life.Stage "retired" is derived from dates and cannot be authored`,
		},
		{
			name:    "invalid stage",
			mutate:  func(e *Entry) { e.Life.Stage = "beta" },
			wantErr: `Life.Stage "beta" is not a valid stage`,
		},
		{
			name: "retires before deprecated",
			mutate: func(e *Entry) {
				e.Life.Deprecated = MustDate("2026-06-01")
				e.Life.Retires = MustDate("2026-05-01")
			},
			wantErr: `Life.Retires 2026-05-01 is before Life.Deprecated 2026-06-01`,
		},
		{
			name: "deprecated before available",
			mutate: func(e *Entry) {
				e.Life.Available = MustDate("2026-06-01")
				e.Life.Deprecated = MustDate("2026-05-01")
			},
			wantErr: `Life.Deprecated 2026-05-01 is before Life.Available 2026-06-01`,
		},
		{
			name: "retires and floor both set",
			mutate: func(e *Entry) {
				e.Life.Retires = MustDate("2027-05-01")
				e.Life.RetirementNotBefore = MustDate("2027-01-01")
			},
			wantErr: `mutually exclusive`,
		},
		{
			name:    "efforts without reasoning capability",
			mutate:  func(e *Entry) { e.Reasoning.Efforts = []llm.ReasoningEffort{"low"} },
			wantErr: `Reasoning.Efforts is set but Capabilities.Reasoning is false`,
		},
		{
			name: "vision without image modality",
			mutate: func(e *Entry) {
				e.Capabilities.Vision = true
				e.Modalities.Input = []Modality{ModalityText}
			},
			wantErr: `Capabilities.Vision is true but Modalities.Input lacks "image"`,
		},
		{
			name: "tuning default exceeds max output",
			mutate: func(e *Entry) {
				e.Tuning.DefaultMaxOutputTokens = 10_000
			},
			wantErr: `Tuning.DefaultMaxOutputTokens 10000 must be <= Constraints.MaxOutputTokens 8192`,
		},
		{
			name: "tuning effort not supported",
			mutate: func(e *Entry) {
				e.Tuning.DefaultReasoningEffort = "high"
			},
			wantErr: `Tuning.DefaultReasoningEffort "high" is not in Reasoning.Efforts`,
		},
		{
			name: "tuning compaction at or above window",
			mutate: func(e *Entry) {
				e.Tuning.CompactAtInputTokens = 200_000
			},
			wantErr: `Tuning.CompactAtInputTokens 200000 must be < Constraints.MaxInputTokens 200000`,
		},
		{
			name: "unpriced input",
			mutate: func(e *Entry) {
				e.Pricing = pricing.Info{Default: pricing.RateCard{
					Base: pricing.Rates{OutputPerMillion: 1_000_000},
				}}
			},
			wantErr: `Pricing.Default.Base.InputPerMillion is unpriced`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			entry := validEntry("robin-2", "acme/robin-2")
			tt.mutate(&entry)

			_, err := New("acme", []Entry{entry}, WithRegistry(testRegistry()))
			require.Error(t, err)
			require.ErrorContains(t, err, tt.wantErr)
			// Every validation error names the provider so joined errors
			// from multiple catalogs stay attributable.
			require.ErrorContains(t, err, "acme")
		})
	}
}

func TestNewCrossEntryValidation(t *testing.T) {
	t.Parallel()

	t.Run("duplicate offering IDs", func(t *testing.T) {
		t.Parallel()

		_, err := New("acme", []Entry{
			validEntry("robin-2", "acme/robin-2"),
			validEntry("robin-2", "acme/robin-2"),
		}, WithRegistry(testRegistry()))
		require.Error(t, err)
		require.ErrorContains(t, err, `entries[1] "robin-2": duplicate offering ID`)
	})

	t.Run("alias collides with offering ID", func(t *testing.T) {
		t.Parallel()

		a := validEntry("robin-2", "acme/robin-2")
		b := validEntry("robin-3", "acme/robin-3")
		b.Aliases = []string{"robin-2"}

		_, err := New("acme", []Entry{a, b}, WithRegistry(testRegistry()))
		require.Error(t, err)
		require.ErrorContains(t, err, `alias "robin-2" on "robin-3" collides with an offering ID`)
	})

	t.Run("alias registered twice", func(t *testing.T) {
		t.Parallel()

		a := validEntry("robin-2", "acme/robin-2")
		a.Aliases = []string{"robin-latest"}
		b := validEntry("robin-3", "acme/robin-3")
		b.Aliases = []string{"robin-latest"}

		_, err := New("acme", []Entry{a, b}, WithRegistry(testRegistry()))
		require.Error(t, err)
		require.ErrorContains(t, err, `alias "robin-latest"`)
		require.ErrorContains(t, err, `already registered`)
	})

	t.Run("replaced by must resolve", func(t *testing.T) {
		t.Parallel()

		e := validEntry("robin-2", "acme/robin-2")
		e.Life.ReplacedBy = "robin-9"

		_, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
		require.Error(t, err)
		require.ErrorContains(t, err, `Life.ReplacedBy "robin-9" is not an offering in this catalog`)
	})

	t.Run("empty provider", func(t *testing.T) {
		t.Parallel()

		_, err := New("", nil)
		require.Error(t, err)
		require.ErrorContains(t, err, "provider name is required")
	})
}

func TestNewNormalization(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	require.Empty(t, e.Label)
	require.Empty(t, e.Modalities.Input)
	require.Empty(t, e.Life.Stage)

	c := mustCatalog(t, e)

	o, ok := c.Lookup("robin-2")
	require.True(t, ok)
	assert.Equal(t, "Robin 2", o.Label, "empty Label defaults to Facts.Name")
	assert.Equal(t, []Modality{ModalityText}, o.Modalities.Input)
	assert.Equal(t, []Modality{ModalityText}, o.Modalities.Output)
	assert.Equal(t, StageGA, o.Life.Stage, "empty Stage defaults to GA")
	assert.Equal(t, "acme", o.Provider())
	assert.Equal(t, "Robin 2", o.Facts().Name)
}

func TestResolve(t *testing.T) {
	t.Parallel()

	robin2 := validEntry("robin-2", "acme/robin-2")
	robin2.Aliases = []string{"robin-latest"}
	robin20522 := validEntry("robin-2-0522", "acme/robin-2")
	c := mustCatalog(t,
		robin2,
		robin20522,
		validEntry("robin-3", "acme/robin-3"),
	)

	tests := []struct {
		requested string
		wantID    string
		wantOK    bool
	}{
		{"robin-2", "robin-2", true},                // exact
		{"robin-latest", "robin-2", true},           // alias
		{"robin-2-0522", "robin-2-0522", true},      // exact beats prefix
		{"robin-2-0522-beta", "robin-2-0522", true}, // longest prefix wins over robin-2
		{"robin-3-20270101", "robin-3", true},       // snapshot suffix, dash boundary
		{"robin-3@default", "robin-3", true},        // at-sign boundary
		{"robin-3.1", "robin-3", true},              // dot boundary
		{"robin-latest-20270101", "robin-2", true},  // alias participates in prefix matching
		{"robin-30", "", false},                     // no boundary byte: not a prefix match
		{"robin-9", "", false},                      // unknown model in a known series
		{"wren-1", "", false},                       // not offered by this provider
		{"", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.requested, func(t *testing.T) {
			t.Parallel()

			o, ok := c.Resolve(tt.requested)
			require.Equal(t, tt.wantOK, ok)

			if tt.wantOK {
				assert.Equal(t, tt.wantID, o.ID)
			}
		})
	}
}

func TestResolveForwardCompat(t *testing.T) {
	t.Parallel()

	// A model launched an hour ago must behave correctly with zero
	// commits: snapshots of a known offering resolve to it, and a new
	// version that is NOT in the catalog must not accidentally match an
	// older sibling.
	c := mustCatalog(t, validEntry("robin-2", "acme/robin-2"))

	o, ok := c.Resolve("robin-2-20270101")
	require.True(t, ok)
	assert.Equal(t, "robin-2", o.ID)

	_, ok = c.Resolve("robin-21")
	assert.False(t, ok, "robin-21 must not resolve to robin-2")

	_, ok = c.Resolve("robin-3")
	assert.False(t, ok, "unknown next generation must report unknown, not match robin-2")
}

func TestLookupImmutability(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.Aliases = []string{"robin-latest"}
	e.Reasoning = ReasoningSupport{}
	e.Attributes = map[string]string{"zone": "a"}
	e.Pricing = e.Pricing.WithOverride(
		pricing.Selector{Speed: llm.SpeedFast},
		pricing.RateCard{Base: pricing.NewRates(6.00, 30.00, 0.60)},
	)
	c := mustCatalog(t, e)

	got, ok := c.Lookup("robin-2")
	require.True(t, ok)

	// Mutate every reference-typed field of the returned copy.
	got.Aliases[0] = mutated
	got.Constraints.SupportedParams[0] = mutated
	got.Modalities.Input[0] = mutated
	got.Attributes["zone"] = mutated
	got.Pricing.Overrides[0].RateCard.Base.InputPerMillion = 1

	again, ok := c.Lookup("robin-2")
	require.True(t, ok)
	assert.Equal(t, "robin-latest", again.Aliases[0])
	assert.Equal(t, "max_tokens", again.Constraints.SupportedParams[0])
	assert.Equal(t, ModalityText, again.Modalities.Input[0])
	assert.Equal(t, "a", again.Attributes["zone"])
	assert.Equal(t, int64(600_000_000), again.Pricing.Overrides[0].RateCard.Base.InputPerMillion)

	// All() copies too.
	all := c.All()
	require.Len(t, all, 1)
	all[0].Aliases[0] = mutated
	again, _ = c.Lookup("robin-2")
	assert.Equal(t, "robin-latest", again.Aliases[0])

	// PricingByID copies and includes aliases.
	pm := c.PricingByID()
	require.Contains(t, pm, "robin-2")
	require.Contains(t, pm, "robin-latest")
	entry := pm["robin-2"]
	entry.Overrides[0].RateCard.Base.InputPerMillion = 1
	again, _ = c.Lookup("robin-2")
	assert.Equal(t, int64(600_000_000), again.Pricing.Overrides[0].RateCard.Base.InputPerMillion)
}

func TestResolveAliasSurvivesReordering(t *testing.T) {
	t.Parallel()

	// The catalog sorts offerings by ID at freeze time. Author the
	// aliased entry LAST and with the lexicographically largest ID, so
	// its authored index differs from its sorted index — this is the
	// regression case where alias indexes captured before sorting would
	// dangle and the alias would resolve to the wrong offering.
	wren := validEntry("wren-1", "acme/wren-1")
	zed := validEntry("robin-3", "acme/robin-3")
	zed.Aliases = []string{"robin-latest"}

	c := mustCatalog(t, wren, zed)

	o, ok := c.Resolve("robin-latest")
	require.True(t, ok)
	assert.Equal(t, "robin-3", o.ID, "alias must resolve to its owner after freeze reordering")
}

func TestSuccessor(t *testing.T) {
	t.Parallel()

	c := mustCatalog(t,
		validEntry("robin-1", "acme/robin-1"),
		validEntry("robin-2", "acme/robin-2"),
		// Two offerings of robin-3 (geo-variant shape): succession is
		// per logical model, so the variants cannot tie.
		validEntry("robin-3", "acme/robin-3"),
		validEntry("eu.robin-3", "acme/robin-3"),
		validEntry("wren-1", "acme/wren-1"),
	)

	s, ok := c.Successor("acme/robin-1")
	require.True(t, ok)
	assert.Equal(t, ModelID("acme/robin-2"), s)

	s, ok = c.Successor("acme/robin-2")
	require.True(t, ok)
	assert.Equal(t, ModelID("acme/robin-3"), s)

	_, ok = c.Successor("acme/robin-3")
	assert.False(t, ok, "newest of the series has no successor")

	_, ok = c.Successor("acme/wren-1")
	assert.False(t, ok, "single-member series has no successor")

	_, ok = c.Successor("acme/unknown")
	assert.False(t, ok)
}

func TestPriceTier(t *testing.T) {
	t.Parallel()

	tier := func(t *testing.T, info pricing.Info) PriceTier {
		t.Helper()

		e := validEntry("robin-2", "acme/robin-2")
		e.Pricing = info

		c, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
		require.NoError(t, err)

		o, ok := c.Lookup("robin-2")
		require.True(t, ok)

		return o.PriceTier()
	}

	t.Run("low", func(t *testing.T) {
		t.Parallel()
		// blended = (3*0.25 + 2.00)/4 = $0.6875/M
		assert.Equal(t, PriceTierLow, tier(t, pricing.FlatInfo(0.25, 2.00, 0.025)))
	})

	t.Run("medium", func(t *testing.T) {
		t.Parallel()
		// blended = (3*3 + 15)/4 = $6/M
		assert.Equal(t, PriceTierMedium, tier(t, pricing.FlatInfo(3.00, 15.00, 0.30)))
	})

	t.Run("high", func(t *testing.T) {
		t.Parallel()
		// blended = (3*5 + 25)/4 = $10/M — boundary is inclusive-high
		assert.Equal(t, PriceTierHigh, tier(t, pricing.FlatInfo(5.00, 25.00, 0.50)))
	})

	t.Run("low-medium boundary", func(t *testing.T) {
		t.Parallel()
		// blended = exactly $2.50/M ⇒ medium
		assert.Equal(t, PriceTierMedium, tier(t, pricing.FlatInfo(2.50, 2.50, 0)))
	})

	t.Run("free", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, PriceTierFree, tier(t, pricing.FlatInfo(pricing.RateFree, pricing.RateFree, pricing.RateFree)))
	})

	t.Run("half free blends", func(t *testing.T) {
		t.Parallel()
		// Free input, priced output: blended = 15/4 = $3.75/M ⇒ medium,
		// not free and not unknown.
		assert.Equal(t, PriceTierMedium, tier(t, pricing.FlatInfo(pricing.RateFree, 15.00, 0)))
	})
}

func TestPriceTierUnknownNeverCheap(t *testing.T) {
	t.Parallel()

	// An unpriced (zero) rate cannot pass New's pricing validation, so
	// exercise the derivation directly: a zero rate must yield Unknown,
	// never a cheap bucket.
	o := Offering{Entry: Entry{Pricing: pricing.Info{Default: pricing.RateCard{
		Base: pricing.Rates{OutputPerMillion: 1_000_000},
	}}}}
	assert.Equal(t, PriceTierUnknown, o.PriceTier())

	o.Pricing.Default.Base = pricing.Rates{}
	assert.Equal(t, PriceTierUnknown, o.PriceTier())
}

func TestDefaultRegistryIntegrity(t *testing.T) {
	t.Parallel()

	reg := DefaultRegistry()
	require.NotEmpty(t, reg)

	for id, f := range reg {
		assert.NotEmpty(t, f.Name, "%s: Name", id)
		assert.NotEmpty(t, f.Series, "%s: Series", id)
		assert.False(t, f.Released.IsZero(), "%s: Released", id)
	}

	// DefaultRegistry returns a copy.
	first := DefaultRegistry()
	first[ModelClaudeOpus5] = Facts{}
	second := DefaultRegistry()
	assert.NotEqual(t, first[ModelClaudeOpus5], second[ModelClaudeOpus5])
}

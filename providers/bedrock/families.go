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
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// family declares one logical Bedrock model and the inference-profile
// variants it is published under. expandFamilies turns each declaration
// into per-variant catalog entries, so a model that AWS serves through
// four geo profiles is authored once instead of four times.
//
// Pricing is authored data, never derived: AWS's current rule is that
// every geo/in-region rate is exactly 1.10x the global rate, but that is
// pinned by TestGeoGlobalRatio as a tripwire rather than computed here —
// a future pricing exception must be a data edit, not a schema redesign.
type family struct {
	// BareID is the vendor-namespaced model ID without a geo prefix:
	// "anthropic.claude-opus-5". Geo variants are derived as
	// "<profile>." + BareID.
	BareID string
	// Model is the canonical cross-provider identity.
	Model catalog.ModelID
	// DisplayName is the undecorated display name; variants get " (US)" /
	// " (Global)" style suffixes.
	DisplayName string

	// Profiles lists the published inference profiles, e.g.
	// {"global", "us", "eu"}. Empty means the model is bare-only.
	Profiles []string
	// BareInvokable registers the bare ID itself. Bare-only on-demand
	// models (Mistral, Gemma, GPT-5.6) set this with no Profiles;
	// profile-only Claude models leave it false.
	BareInvokable bool

	// Mantle marks families served exclusively on the bedrock-mantle
	// OpenAI-compatible endpoint. Mantle families must be bare-only:
	// AWS publishes no inference profiles for them.
	Mantle bool
	// MantleRegions records the bedrock-mantle regions the model card
	// publishes, registered in mantleRegions. It never blocks a call: the
	// list is copied by hand and AWS adds regions without notice, so a
	// stale list must not refuse requests that would work. It only adds a
	// region hint to a model-not-found error from another region. Valid
	// only with Mantle and BareInvokable.
	MantleRegions []string
	// NoCachePoints marks Converse families that reject CachePoint blocks
	// (AccessDeniedException "your request did not allow prompt caching").
	// NewModel turns prompt caching off for them regardless of the
	// provider's caching setting.
	NoCachePoints bool
	// DataSharing marks families that require the account to opt in to
	// provider data sharing; it surfaces as the
	// ModelMetadataRequiresProviderDataSharing attribute.
	DataSharing bool

	Capabilities llm.ModelCapabilities
	Constraints  llm.ModelConstraints
	// Modalities lists the input/output content kinds every variant of
	// the family accepts. Empty normalizes to text-only, so a family
	// whose Capabilities declare Vision must list ModalityImage here or
	// catalog.New rejects it.
	Modalities catalog.Modalities
	Reasoning  catalog.ReasoningSupport
	Life       catalog.Lifecycle

	// Rates is the geo / in-region rate card, used for the bare ID and
	// every non-global profile.
	Rates pricing.RateCard
	// GlobalRates is the global-profile rate card; required exactly when
	// "global" is in Profiles.
	GlobalRates *pricing.RateCard
	// Overrides layers selector-keyed rate cards (a speed or service
	// tier) on Rates, for the same variants. Each card carries its own
	// brackets.
	Overrides []pricing.Override
	// GlobalOverrides layers the same way on GlobalRates; allowed only
	// when "global" is in Profiles, and required there when Overrides is
	// set.
	GlobalOverrides []pricing.Override

	// ProfileRegions opts the family into exact geo routing: a source
	// region → profile map registered in profileRegionResolvers.
	ProfileRegions map[string]string
}

// profileLabels maps a profile prefix to its display suffix.
var profileLabels = map[string]string{
	"global": " (Global)",
	"us":     " (US)",
	"eu":     " (EU)",
	"au":     " (AU)",
	"jp":     " (JP)",
}

// expandFamilies turns family declarations into per-variant catalog
// entries, in deterministic order (bare first, then declared profile
// order). It also accumulates the mantle ID set consulted by
// IsMantleModel. Invalid declarations panic: families are compile-time
// literals exercised by every test run.
func expandFamilies(families []family) ([]catalog.Entry, map[string]bool) {
	var entries []catalog.Entry

	mantle := make(map[string]bool)

	for _, f := range families {
		if f.Mantle && (len(f.Profiles) > 0 || !f.BareInvokable) {
			panic(fmt.Sprintf("bedrock: mantle family %s must be bare-only", f.BareID)) //nolint:forbidigo // authoring error, not runtime
		}

		if len(f.MantleRegions) > 0 && (!f.Mantle || !f.BareInvokable) {
			panic(fmt.Sprintf("bedrock: family %s sets MantleRegions without Mantle and BareInvokable", f.BareID)) //nolint:forbidigo // authoring error, not runtime
		}

		hasGlobal := false

		for _, p := range f.Profiles {
			if _, ok := profileLabels[p]; !ok {
				panic(fmt.Sprintf("bedrock: family %s references unknown profile %q", f.BareID, p)) //nolint:forbidigo // authoring error, not runtime
			}

			if p == "global" {
				hasGlobal = true
			}
		}

		if hasGlobal != (f.GlobalRates != nil) {
			panic(fmt.Sprintf("bedrock: family %s must set GlobalRates exactly when the global profile is published", f.BareID)) //nolint:forbidigo // authoring error, not runtime
		}

		if !hasGlobal && len(f.GlobalOverrides) > 0 {
			panic(fmt.Sprintf("bedrock: family %s sets GlobalOverrides without the global profile", f.BareID)) //nolint:forbidigo // authoring error, not runtime
		}

		if hasGlobal && len(f.Overrides) > 0 && len(f.GlobalOverrides) == 0 {
			panic(fmt.Sprintf("bedrock: family %s sets Overrides without GlobalOverrides for the global profile", f.BareID)) //nolint:forbidigo // authoring error, not runtime
		}

		// geo is the inference-profile geography ("us", "global", ...);
		// empty for bare IDs, which run in the calling region.
		variant := func(id, labelSuffix, geo string, rates pricing.RateCard, overrides []pricing.Override) catalog.Entry {
			var attrs map[string]string
			if f.DataSharing || geo != "" {
				attrs = make(map[string]string, 2)
				if f.DataSharing {
					attrs[ModelMetadataRequiresProviderDataSharing] = "true"
				}

				if geo != "" {
					attrs[ModelMetadataInferenceGeo] = geo
				}
			}

			return catalog.Entry{
				ID:           id,
				Model:        f.Model,
				DisplayName:  f.DisplayName + labelSuffix,
				Capabilities: f.Capabilities,
				Constraints:  f.Constraints,
				Modalities:   f.Modalities,
				Reasoning:    f.Reasoning,
				Life:         f.Life,
				Pricing:      rateInfo(rates, overrides),
				Attributes:   attrs,
			}
		}

		if f.BareInvokable {
			entries = append(entries, variant(f.BareID, "", "", f.Rates, f.Overrides))

			if f.Mantle {
				mantle[f.BareID] = true
			}
		}

		for _, p := range f.Profiles {
			rates, overrides := f.Rates, f.Overrides
			if p == "global" {
				rates, overrides = *f.GlobalRates, f.GlobalOverrides
			}

			entries = append(entries, variant(p+"."+f.BareID, profileLabels[p], p, rates, overrides))
		}
	}

	return entries, mantle
}

// rateInfo builds a variant's pricing from its default card and any
// selector overrides.
func rateInfo(card pricing.RateCard, overrides []pricing.Override) pricing.Info {
	info := pricing.Info{Default: card}
	for _, o := range overrides {
		info = info.WithOverride(o.Match, o.RateCard)
	}

	return info
}

// buildProfileRegionResolvers collects the per-family geo routing maps
// into the resolver tables consumed by NewModel and
// IsModelAllowedFromRegion, so routing is single-sourced from the family
// declarations.
func buildProfileRegionResolvers(families []family) (map[string]func(string) (string, bool), map[string]map[string]string) {
	resolvers := make(map[string]func(string) (string, bool))
	regions := make(map[string]map[string]string)

	for _, f := range families {
		if f.ProfileRegions == nil {
			continue
		}

		table := f.ProfileRegions
		resolvers[f.BareID] = func(region string) (string, bool) {
			profile, ok := table[region]
			return profile, ok
		}
		regions[f.BareID] = table
	}

	return resolvers, regions
}

// buildMantleRegions collects each mantle family's published regions into
// the bare-ID → region set behind the mantle model-not-found hint,
// single-sourced from the family declarations (family.MantleRegions).
func buildMantleRegions(families []family) map[string]map[string]bool {
	regions := make(map[string]map[string]bool)

	for _, f := range families {
		if len(f.MantleRegions) == 0 {
			continue
		}

		set := make(map[string]bool, len(f.MantleRegions))
		for _, r := range f.MantleRegions {
			set[r] = true
		}

		regions[f.BareID] = set
	}

	return regions
}

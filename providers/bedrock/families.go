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

		// geo is the inference-profile geography ("us", "global", ...);
		// empty for bare IDs, which run in the calling region.
		variant := func(id, labelSuffix, geo string, rates pricing.RateCard) catalog.Entry {
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
				Pricing:      pricing.Info{Default: rates},
				Attributes:   attrs,
			}
		}

		if f.BareInvokable {
			entries = append(entries, variant(f.BareID, "", "", f.Rates))

			if f.Mantle {
				mantle[f.BareID] = true
			}
		}

		for _, p := range f.Profiles {
			rates := f.Rates
			if p == "global" {
				rates = *f.GlobalRates
			}

			entries = append(entries, variant(p+"."+f.BareID, profileLabels[p], p, rates))
		}
	}

	return entries, mantle
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

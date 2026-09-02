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

// Package snapshot renders catalogs into the committed JSON artifact
// (catalog/snapshot.json): the review surface for catalog changes, the
// read format for non-Go consumers, and the wire format a future
// remote-refresh layer would serve.
//
// It reads only the catalog's public API — the wire format cannot depend
// on package internals. Output is deterministic (sorted, no timestamps,
// only time-independent derivations), so the artifact changes exactly
// when the authored data changes.
package snapshot

import (
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"
	"time"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// SchemaVersion identifies the snapshot wire format. It bumps only on a
// rename, removal, or semantic change of an existing field — additions
// are not version bumps. Tolerant-reader contract: consumers MUST ignore
// unknown fields and MUST NOT require optional ones.
const SchemaVersion = 1

// Encode writes the deterministic JSON snapshot of the given catalogs:
// providers sorted by name, offerings sorted by ID, facts sorted by
// ModelID.
func Encode(w io.Writer, catalogs ...*catalog.Catalog) error {
	snap := snapshotDTO{
		SchemaVersion: SchemaVersion,
		Facts:         map[string]factsDTO{},
	}

	sorted := slices.Clone(catalogs)
	slices.SortFunc(sorted, func(a, b *catalog.Catalog) int {
		switch {
		case a.Provider() < b.Provider():
			return -1
		case a.Provider() > b.Provider():
			return 1
		default:
			return 0
		}
	})

	for _, c := range sorted {
		if c == nil {
			continue
		}

		prov := providerDTO{
			Provider: c.Provider(),
		}

		for _, o := range c.All() {
			f := o.Facts()
			dto := factsDTO{
				DisplayName: f.DisplayName,
				Description: f.Description,
				Series:      f.Series,
				Released:    dateString(f.Released),
				Knowledge:   dateString(f.Knowledge),
				OpenWeights: f.OpenWeights,
			}

			if prev, ok := snap.Facts[string(o.Model)]; ok && prev != dto {
				return fmt.Errorf("snapshot: conflicting facts for %q across providers", o.Model)
			}

			snap.Facts[string(o.Model)] = dto

			replacement := ""
			if r, ok := c.Replacement(o.ID); ok {
				replacement = string(r)
			}

			prov.Offerings = append(prov.Offerings, offeringDTO{
				ID:           o.ID,
				Model:        string(o.Model),
				DisplayName:  o.DisplayName,
				Aliases:      o.Aliases,
				Capabilities: capabilitiesDTO(o.Capabilities),
				Constraints: constraintsDTO{
					MaxInputTokens:    o.Constraints.MaxInputTokens,
					MaxOutputTokens:   o.Constraints.MaxOutputTokens,
					SupportedParams:   o.Constraints.SupportedParams,
					TemperatureRange:  temperatureRangeDTO(o.Constraints.TemperatureRange),
					MutuallyExclusive: o.Constraints.MutuallyExclusive,
					ConditionalRules:  conditionalRulesDTO(o.Constraints.ConditionalRules),
				},
				Modalities: modalitiesDTO{
					Input:  o.Modalities.Input,
					Output: o.Modalities.Output,
				},
				Reasoning: reasoningDTO{
					Efforts:  o.Reasoning.Efforts,
					Adaptive: o.Reasoning.Adaptive,
					Budget:   o.Reasoning.Budget,
				},
				Speeds:     o.Speeds,
				Pricing:    pricingDTOFrom(o.Pricing),
				Lifecycle:  lifecycleDTOFrom(o.Life),
				Attributes: sortedAttributes(o.Attributes),
				Derived: derivedDTO{
					PriceTier:   string(o.PriceTier()),
					Replacement: replacement,
				},
			})
		}

		snap.Providers = append(snap.Providers, prov)
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")

	return enc.Encode(snap)
}

// dateString renders a date as "YYYY-MM-DD", or "" when unset.
func dateString(t time.Time) string {
	if t.IsZero() {
		return ""
	}

	return t.UTC().Format(time.DateOnly)
}

// ---- DTOs ------------------------------------------------------------
//
// The snapshot has its own DTOs rather than serializing the Go read
// model: the Go types are free to evolve, the wire format is a
// contract. Field tags are snake_case; encoding/json sorts map keys, so
// output is deterministic. Dates are "YYYY-MM-DD" strings (empty when
// unset): the provider's published form, no implied intra-day precision.

type snapshotDTO struct {
	SchemaVersion int                 `json:"schema_version"`
	Facts         map[string]factsDTO `json:"facts"`
	Providers     []providerDTO       `json:"providers"`
}

type factsDTO struct {
	DisplayName string `json:"display_name"`
	Description string `json:"description,omitempty"`
	Series      string `json:"series"`
	Released    string `json:"released"`
	Knowledge   string `json:"knowledge,omitempty"`
	OpenWeights bool   `json:"open_weights,omitempty"`
}

type providerDTO struct {
	Provider  string        `json:"provider"`
	Offerings []offeringDTO `json:"offerings"`
}

type offeringDTO struct {
	ID           string          `json:"id"`
	Model        string          `json:"model"`
	DisplayName  string          `json:"display_name"`
	Aliases      []string        `json:"aliases,omitempty"`
	Capabilities map[string]bool `json:"capabilities"`
	Constraints  constraintsDTO  `json:"constraints"`
	Modalities   modalitiesDTO   `json:"modalities"`
	Reasoning    reasoningDTO    `json:"reasoning"`
	Speeds       []llm.Speed     `json:"speeds,omitempty"`
	Pricing      pricingDTO      `json:"pricing"`
	Lifecycle    lifecycleDTO    `json:"lifecycle"`
	Attributes   []attributeDTO  `json:"attributes,omitempty"`
	Derived      derivedDTO      `json:"derived"`
}

func capabilitiesDTO(c llm.ModelCapabilities) map[string]bool {
	// A map keeps the snapshot additive when llm.ModelCapabilities
	// grows; encoding/json sorts the keys.
	return map[string]bool{
		"streaming":         c.Streaming,
		"tools":             c.Tools,
		"json_mode":         c.JSONMode,
		"structured_output": c.StructuredOutput,
		"vision":            c.Vision,
		"audio":             c.Audio,
		"multi_turn":        c.MultiTurn,
		"system_prompts":    c.SystemPrompts,
		"reasoning":         c.Reasoning,
	}
}

type constraintsDTO struct {
	MaxInputTokens  int      `json:"max_input_tokens"`
	MaxOutputTokens int      `json:"max_output_tokens"`
	SupportedParams []string `json:"supported_params,omitempty"`
	// TemperatureRange is [min, max]; absent when the entry states none.
	TemperatureRange  []float64            `json:"temperature_range,omitempty"`
	MutuallyExclusive [][]string           `json:"mutually_exclusive,omitempty"`
	ConditionalRules  []conditionalRuleDTO `json:"conditional_rules,omitempty"`
}

type conditionalRuleDTO struct {
	Condition string   `json:"condition"`
	Disables  []string `json:"disables,omitempty"`
	Message   string   `json:"message,omitempty"`
}

func temperatureRangeDTO(r [2]float64) []float64 {
	if r == [2]float64{} {
		return nil
	}

	return []float64{r[0], r[1]}
}

func conditionalRulesDTO(rules []llm.ConditionalRule) []conditionalRuleDTO {
	out := make([]conditionalRuleDTO, 0, len(rules))
	for _, r := range rules {
		out = append(out, conditionalRuleDTO{
			Condition: r.Condition,
			Disables:  r.Disables,
			Message:   r.Message,
		})
	}

	return out
}

type modalitiesDTO struct {
	Input  []catalog.Modality `json:"input"`
	Output []catalog.Modality `json:"output"`
}

type reasoningDTO struct {
	Efforts  []llm.ReasoningEffort `json:"efforts,omitempty"`
	Adaptive bool                  `json:"adaptive,omitempty"`
	Budget   bool                  `json:"budget,omitempty"`
}

type pricingDTO struct {
	Default   rateCardDTO       `json:"default"`
	Overrides []pricingOverride `json:"overrides,omitempty"`
}

type pricingOverride struct {
	ServiceTier string      `json:"service_tier,omitempty"`
	Speed       string      `json:"speed,omitempty"`
	Region      string      `json:"region,omitempty"`
	RateCard    rateCardDTO `json:"rate_card"`
}

type rateCardDTO struct {
	Base     ratesDTO     `json:"base"`
	Brackets []bracketDTO `json:"brackets,omitempty"`
}

type bracketDTO struct {
	MinContextTokens int64    `json:"min_context_tokens"`
	Rates            ratesDTO `json:"rates"`
}

// ratesDTO mirrors pricing.Rates. The unit travels in every field name
// so a consumer typing the schema cannot mistake the values for dollars
// or cents: microcents per million tokens, i.e. $5.00/M is 500_000_000.
type ratesDTO struct {
	Input                   int64 `json:"input_microcents_per_million"`
	Output                  int64 `json:"output_microcents_per_million"`
	CachedInput             int64 `json:"cached_input_microcents_per_million,omitempty"`
	CacheCreation5m         int64 `json:"cache_creation_5m_microcents_per_million,omitempty"`
	CacheCreation1h         int64 `json:"cache_creation_1h_microcents_per_million,omitempty"`
	CacheCreationUnknownTTL int64 `json:"cache_creation_unknown_ttl_microcents_per_million,omitempty"`
}

func pricingDTOFrom(info pricing.Info) pricingDTO {
	dto := pricingDTO{Default: rateCardDTOFrom(info.Default)}
	for _, ov := range info.Overrides {
		dto.Overrides = append(dto.Overrides, pricingOverride{
			ServiceTier: string(ov.Match.ServiceTier),
			Speed:       string(ov.Match.Speed),
			Region:      ov.Match.Region,
			RateCard:    rateCardDTOFrom(ov.RateCard),
		})
	}

	return dto
}

func rateCardDTOFrom(card pricing.RateCard) rateCardDTO {
	dto := rateCardDTO{Base: ratesDTOFrom(card.Base)}
	for _, b := range card.Brackets {
		dto.Brackets = append(dto.Brackets, bracketDTO{
			MinContextTokens: b.MinContextTokens,
			Rates:            ratesDTOFrom(b.Rates),
		})
	}

	return dto
}

func ratesDTOFrom(r pricing.Rates) ratesDTO {
	return ratesDTO{
		Input:                   r.InputPerMillion,
		Output:                  r.OutputPerMillion,
		CachedInput:             r.CachedInputPerMillion,
		CacheCreation5m:         r.CacheCreation5mPerMillion,
		CacheCreation1h:         r.CacheCreation1hPerMillion,
		CacheCreationUnknownTTL: r.CacheCreationUnknownTTLPerMillion,
	}
}

type lifecycleDTO struct {
	Stage      string `json:"stage"`
	Available  string `json:"available,omitempty"`
	Deprecated string `json:"deprecated,omitempty"`
	Retires    string `json:"retires,omitempty"`
	ReplacedBy string `json:"replaced_by,omitempty"`
}

func lifecycleDTOFrom(l catalog.Lifecycle) lifecycleDTO {
	return lifecycleDTO{
		Stage:      string(l.Stage),
		Available:  dateString(l.Available),
		Deprecated: dateString(l.Deprecated),
		Retires:    dateString(l.Retires),
		ReplacedBy: l.ReplacedBy,
	}
}

type attributeDTO struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

func sortedAttributes(attrs map[string]string) []attributeDTO {
	if len(attrs) == 0 {
		return nil
	}

	keys := slices.Sorted(maps.Keys(attrs))

	out := make([]attributeDTO, 0, len(keys))
	for _, k := range keys {
		out = append(out, attributeDTO{Key: k, Value: attrs[k]})
	}

	return out
}

// derivedDTO carries the time-independent derivations. Replacement is
// catalog.Replacement — the announced replaced_by when set, otherwise the
// series successor — as a ModelID, so consumers never re-implement the
// precedence.
type derivedDTO struct {
	PriceTier   string `json:"price_tier"`
	Replacement string `json:"replacement,omitempty"`
}

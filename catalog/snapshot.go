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
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// SnapshotSchemaVersion identifies the snapshot wire format. It bumps
// only on a rename, removal, or semantic change of an existing field —
// additions are not version bumps.
//
// Tolerant-reader contract: consumers MUST ignore unknown fields, and
// MUST NOT require fields this package documents as optional. The
// snapshot is the review artifact for catalog changes and the read
// format for non-Go consumers; a future remote-refresh layer uses it as
// its wire format unchanged.
const SnapshotSchemaVersion = 1

// EncodeSnapshot writes the deterministic JSON snapshot of the given
// catalogs: providers sorted by name, offerings sorted by ID, facts
// sorted by ModelID, and no timestamps — only time-independent derived
// fields (price tier, successor) are included, so the artifact changes
// exactly when the authored data changes.
func EncodeSnapshot(w io.Writer, catalogs ...*Catalog) error {
	snap := snapshotDTO{
		SchemaVersion: SnapshotSchemaVersion,
		Facts:         map[string]factsDTO{},
	}

	sorted := slices.Clone(catalogs)
	slices.SortFunc(sorted, func(a, b *Catalog) int {
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

		for id, f := range c.facts {
			dto := factsDTO(f)

			if prev, ok := snap.Facts[string(id)]; ok && prev != dto {
				return fmt.Errorf("catalog: snapshot: conflicting facts for %q across providers", id)
			}

			snap.Facts[string(id)] = dto
		}

		prov := providerDTO{
			Provider: c.Provider(),
		}

		for _, o := range c.offerings {
			successor := ""
			if s, ok := c.Successor(o.Model); ok {
				successor = string(s)
			}

			prov.Offerings = append(prov.Offerings, offeringDTO{
				ID:           o.ID,
				Model:        string(o.Model),
				Label:        o.Label,
				Aliases:      slices.Clone(o.Aliases),
				Capabilities: capabilitiesDTO(o.Capabilities),
				Constraints: constraintsDTO{
					MaxInputTokens:  o.Constraints.MaxInputTokens,
					MaxOutputTokens: o.Constraints.MaxOutputTokens,
					SupportedParams: slices.Clone(o.Constraints.SupportedParams),
				},
				Modalities: modalitiesDTO{
					Input:  slices.Clone(o.Modalities.Input),
					Output: slices.Clone(o.Modalities.Output),
				},
				Reasoning: reasoningDTO{
					Efforts:  slices.Clone(o.Reasoning.Efforts),
					Adaptive: o.Reasoning.Adaptive,
					Budget:   o.Reasoning.Budget,
				},
				Speeds:     slices.Clone(o.Speeds),
				Pricing:    pricingDTOFrom(o.Pricing),
				Lifecycle:  lifecycleDTOFrom(o.Life),
				Tuning:     tuningDTOFrom(o.Tuning),
				Attributes: sortedAttributes(o.Attributes),
				Derived: derivedDTO{
					PriceTier: string(o.PriceTier()),
					Successor: successor,
				},
			})
		}

		snap.Providers = append(snap.Providers, prov)
	}

	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")

	return enc.Encode(snap)
}

// ---- DTOs ------------------------------------------------------------
//
// The snapshot has its own DTOs rather than serializing the Go read
// model: the Go types are free to evolve, the wire format is a
// contract. Field tags are snake_case; encoding/json sorts map keys, so
// output is deterministic.

type snapshotDTO struct {
	SchemaVersion int                 `json:"schema_version"`
	Facts         map[string]factsDTO `json:"facts"`
	Providers     []providerDTO       `json:"providers"`
}

type factsDTO struct {
	Name        string `json:"name"`
	Series      string `json:"series"`
	Released    Date   `json:"released"`
	Knowledge   Date   `json:"knowledge,omitzero"`
	OpenWeights bool   `json:"open_weights,omitempty"`
}

type providerDTO struct {
	Provider  string        `json:"provider"`
	Offerings []offeringDTO `json:"offerings"`
}

type offeringDTO struct {
	ID           string          `json:"id"`
	Model        string          `json:"model"`
	Label        string          `json:"label"`
	Aliases      []string        `json:"aliases,omitempty"`
	Capabilities map[string]bool `json:"capabilities"`
	Constraints  constraintsDTO  `json:"constraints"`
	Modalities   modalitiesDTO   `json:"modalities"`
	Reasoning    reasoningDTO    `json:"reasoning"`
	Speeds       []llm.Speed     `json:"speeds,omitempty"`
	Pricing      pricingDTO      `json:"pricing"`
	Lifecycle    lifecycleDTO    `json:"lifecycle"`
	Tuning       *tuningDTO      `json:"tuning,omitempty"`
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
}

type modalitiesDTO struct {
	Input  []Modality `json:"input"`
	Output []Modality `json:"output"`
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

// ratesDTO mirrors pricing.Rates in microcents per million tokens.
type ratesDTO struct {
	Input                   int64 `json:"input_per_million"`
	Output                  int64 `json:"output_per_million"`
	CachedInput             int64 `json:"cached_input_per_million,omitempty"`
	CacheCreation5m         int64 `json:"cache_creation_5m_per_million,omitempty"`
	CacheCreation1h         int64 `json:"cache_creation_1h_per_million,omitempty"`
	CacheCreationUnknownTTL int64 `json:"cache_creation_unknown_ttl_per_million,omitempty"`
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
	Stage               string `json:"stage"`
	Available           Date   `json:"available,omitzero"`
	Deprecated          Date   `json:"deprecated,omitzero"`
	Retires             Date   `json:"retires,omitzero"`
	RetirementNotBefore Date   `json:"retirement_not_before,omitzero"`
	ReplacedBy          string `json:"replaced_by,omitempty"`
}

func lifecycleDTOFrom(l Lifecycle) lifecycleDTO {
	return lifecycleDTO{
		Stage:               string(l.Stage),
		Available:           l.Available,
		Deprecated:          l.Deprecated,
		Retires:             l.Retires,
		RetirementNotBefore: l.RetirementNotBefore,
		ReplacedBy:          l.ReplacedBy,
	}
}

type tuningDTO struct {
	DefaultMaxOutputTokens int    `json:"default_max_output_tokens,omitempty"`
	DefaultReasoningEffort string `json:"default_reasoning_effort,omitempty"`
	CompactAtInputTokens   int    `json:"compact_at_input_tokens,omitempty"`
}

func tuningDTOFrom(t Tuning) *tuningDTO {
	if t == (Tuning{}) {
		return nil
	}

	return &tuningDTO{
		DefaultMaxOutputTokens: t.DefaultMaxOutputTokens,
		DefaultReasoningEffort: string(t.DefaultReasoningEffort),
		CompactAtInputTokens:   t.CompactAtInputTokens,
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

type derivedDTO struct {
	PriceTier string `json:"price_tier"`
	Successor string `json:"successor,omitempty"`
}

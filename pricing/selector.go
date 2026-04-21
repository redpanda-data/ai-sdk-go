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

package pricing

import (
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Selector identifies request metadata that can change the price of
// the same model.
//
// An Override's Match may leave fields empty: empty fields act as
// wildcards during resolution. An override whose Match is entirely
// empty is rejected at build time because it would shadow Info.Default.
//
// Selector is a concrete struct by design. Adding a new dimension (e.g.
// Tenancy, Endpoint) is an additive API bump rather than a free-form
// key/value bag that silently becomes load-bearing.
type Selector struct {
	ServiceTier llm.ServiceTier
	Speed       llm.Speed
	Region      string
}

// IsZero reports whether the selector has no dimensions set.
func (s Selector) IsZero() bool {
	return s.ServiceTier == "" && s.Speed == "" && s.Region == ""
}

// Override binds a selector match to one rate card inside an Info.
type Override struct {
	Match    Selector
	RateCard RateCard
}

func normalizeSelector(selector Selector) Selector {
	// ServiceTier and Speed go through the canonical llm normalizers
	// so aliases (ServiceTier's "standard"/"auto" → "default",
	// Speed's casing and dash variants) collapse to the same form
	// provider responses normalize to.
	selector.ServiceTier = llm.NormalizeServiceTier(string(selector.ServiceTier))
	selector.Speed = llm.NormalizeSpeed(string(selector.Speed))
	selector.Region = strings.ToLower(strings.TrimSpace(selector.Region))

	return selector
}

func selectorString(selector Selector) string {
	parts := make([]string, 0, 3)
	if selector.ServiceTier != "" {
		parts = append(parts, "service_tier="+string(selector.ServiceTier))
	}

	if selector.Speed != "" {
		parts = append(parts, "speed="+string(selector.Speed))
	}

	if selector.Region != "" {
		parts = append(parts, "region="+selector.Region)
	}

	if len(parts) == 0 {
		return "default"
	}

	return strings.Join(parts, ",")
}

func selectorMatches(candidate, actual Selector) bool {
	return selectorFieldMatches(string(candidate.ServiceTier), string(actual.ServiceTier)) &&
		selectorFieldMatches(string(candidate.Speed), string(actual.Speed)) &&
		selectorFieldMatches(candidate.Region, actual.Region)
}

func selectorsOverlap(a, b Selector) bool {
	return selectorFieldsOverlap(string(a.ServiceTier), string(b.ServiceTier)) &&
		selectorFieldsOverlap(string(a.Speed), string(b.Speed)) &&
		selectorFieldsOverlap(a.Region, b.Region)
}

func selectorFieldMatches(candidate, actual string) bool {
	return candidate == "" || candidate == actual
}

func selectorFieldsOverlap(a, b string) bool {
	return a == "" || b == "" || a == b
}

func selectorSpecificity(selector Selector) int {
	score := 0
	if selector.ServiceTier != "" {
		score++
	}

	if selector.Speed != "" {
		score++
	}

	if selector.Region != "" {
		score++
	}

	return score
}

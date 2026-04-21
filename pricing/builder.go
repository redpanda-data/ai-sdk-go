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
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"slices"
)

// schemaVersion seeds the catalog version hash. Bump it when the
// calculator's resolution rules, rounding, or Cost shape changes so
// existing rate data produces a different Version() even if the numbers
// are identical.
const schemaVersion = "v1"

// Option configures a catalog during NewCatalog construction.
type Option func(*catalogBuilder)

// WithProvider registers one provider's model pricing definitions. The
// provider argument is used for duplicate diagnostics and scope-
// prefixed error messages, not for lookup keys — runtime resolution is
// by bare model ID.
//
// If the same model ID is registered twice (from the same or different
// providers), NewCatalog returns a duplicate-pricing error rather than
// silently clobbering. Intentional replacements should go through
// WithOverride.
func WithProvider(provider string, models map[string]Info) Option {
	return func(b *catalogBuilder) {
		for id, info := range models {
			if existing, ok := b.models[id]; ok {
				b.buildErrs = append(b.buildErrs,
					fmt.Errorf("duplicate pricing for model %q from providers %q and %q",
						id, existing.provider, provider))

				continue
			}

			b.models[id] = catalogEntry{
				provider: provider,
				info:     cloneInfo(info),
			}
		}
	}
}

// WithOverride replaces the pricing of an existing model ID. Unknown
// IDs surface as errors from NewCatalog.
func WithOverride(modelID string, info Info) Option {
	return func(b *catalogBuilder) {
		b.overrides[modelID] = cloneInfo(info)
	}
}

// NewCatalog constructs a validated pricing catalog from the given
// options. Options apply in order: providers register their models,
// overrides replace individual entries, and the builder then
// normalizes and validates the result (duplicate IDs, ambiguous
// selectors, malformed rates, etc.) — any of which surface as a joined
// error.
func NewCatalog(opts ...Option) (*Catalog, error) {
	b := &catalogBuilder{
		models:    make(map[string]catalogEntry),
		overrides: make(map[string]Info),
	}

	for _, opt := range opts {
		opt(b)
	}

	return b.build()
}

// catalogBuilder is the internal accumulator that Option functions
// mutate. It is unexported because the public API is NewCatalog +
// options; catalogs are built once at startup, not progressively.
type catalogBuilder struct {
	models    map[string]catalogEntry
	overrides map[string]Info
	buildErrs []error
}

// catalogEntry tracks which provider registered each model so the
// builder can produce a useful diagnostic when two providers register
// the exact same ID. Provider is not part of lookups.
type catalogEntry struct {
	provider string
	info     Info
}

func (b *catalogBuilder) build() (*Catalog, error) {
	result := make(map[string]Info, len(b.models))
	errs := slices.Clone(b.buildErrs)

	for id, entry := range b.models {
		normalized, err := normalizeInfo(entry.info, entry.provider+"/"+id)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		result[id] = normalized
	}

	for id, override := range b.overrides {
		if _, ok := result[id]; !ok {
			errs = append(errs, fmt.Errorf("override for unknown model %q", id))
			continue
		}

		normalized, err := normalizeInfo(override, "override/"+id)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		result[id] = normalized
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	return &Catalog{
		models:  result,
		version: computeVersion(result),
	}, nil
}

func normalizeInfo(info Info, scope string) (Info, error) {
	normalized := cloneInfo(info)
	errs := make([]error, 0)

	var err error

	normalized.Default, err = normalizeRateCard(normalized.Default, scope+"/default")
	if err != nil {
		errs = append(errs, err)
	}

	if len(normalized.Overrides) > 0 {
		cleaned := make([]Override, 0, len(normalized.Overrides))
		seen := make(map[Selector]int, len(normalized.Overrides))

		for idx, override := range normalized.Overrides {
			match := normalizeSelector(override.Match)
			if match.IsZero() {
				errs = append(errs, fmt.Errorf("%s/override[%d]: empty Selector shadows Default; remove the override or add a dimension", scope, idx))
				continue
			}

			card, err := normalizeRateCard(override.RateCard, fmt.Sprintf("%s/override[%s]", scope, selectorString(match)))
			if err != nil {
				errs = append(errs, err)
				continue
			}

			if prev, exists := seen[match]; exists {
				errs = append(errs, fmt.Errorf("%s/override[%d]: duplicate selector %q (already defined at override[%d])",
					scope, idx, selectorString(match), prev))

				continue
			}

			seen[match] = idx
			cleaned = append(cleaned, Override{Match: match, RateCard: card})
		}

		if err := validateOverrides(cleaned, scope); err != nil {
			errs = append(errs, err)
		}

		normalized.Overrides = cleaned
	}

	if len(errs) > 0 {
		return Info{}, errors.Join(errs...)
	}

	return normalized, nil
}

func normalizeRateCard(card RateCard, scope string) (RateCard, error) {
	normalized := cloneRateCard(card)
	errs := make([]error, 0)

	if err := validateRates(normalized.Base, scope+"/base"); err != nil {
		errs = append(errs, err)
	}

	slices.SortFunc(normalized.Brackets, func(a, b Bracket) int {
		return cmp.Compare(a.MinContextTokens, b.MinContextTokens)
	})

	var lastMin int64 = -1

	for idx, bracket := range normalized.Brackets {
		if bracket.MinContextTokens <= 0 {
			errs = append(errs, fmt.Errorf("%s/bracket[%d]: MinContextTokens must be > 0 (a zero threshold shadows RateCard.Base)", scope, idx))
		}

		if bracket.MinContextTokens == lastMin {
			errs = append(errs, fmt.Errorf("%s/bracket[%d]: duplicate MinContextTokens %d", scope, idx, bracket.MinContextTokens))
		}

		lastMin = bracket.MinContextTokens

		if err := validateRates(bracket.Rates, fmt.Sprintf("%s/bracket[%d]", scope, idx)); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		return RateCard{}, errors.Join(errs...)
	}

	return normalized, nil
}

func validateRates(rates Rates, scope string) error {
	if isValidRate(rates.InputPerMillion) &&
		isValidRate(rates.OutputPerMillion) &&
		isValidRate(rates.CachedInputPerMillion) &&
		isValidRate(rates.CacheCreation5mPerMillion) &&
		isValidRate(rates.CacheCreation1hPerMillion) &&
		isValidRate(rates.CacheCreationUnknownTTLPerMillion) {
		return nil
	}

	return fmt.Errorf("%s: every rate must be non-negative or RateFree", scope)
}

func isValidRate(v int64) bool {
	return v >= 0 || v == RateFree
}

func validateOverrides(overrides []Override, scope string) error {
	if len(overrides) < 2 {
		return nil
	}

	for i := range overrides {
		for j := i + 1; j < len(overrides); j++ {
			a := overrides[i].Match
			b := overrides[j].Match

			if !selectorsOverlap(a, b) {
				continue
			}

			if selectorSpecificity(a) == selectorSpecificity(b) {
				return fmt.Errorf("%s: ambiguous selectors %q and %q have the same specificity and overlap",
					scope, selectorString(a), selectorString(b))
			}
		}
	}

	return nil
}

func computeVersion(models map[string]Info) string {
	keys := make([]string, 0, len(models))
	for key := range models {
		keys = append(keys, key)
	}

	slices.Sort(keys)

	h := sha256.New()
	fmt.Fprintf(h, "schema=%s\n", schemaVersion)

	for _, key := range keys {
		info := models[key]
		fmt.Fprintf(h, "model=%s\n", key)
		writeRateCard(h, "default", info.Default)

		// Sort so append order does not affect the hash.
		sortedOverrides := slices.Clone(info.Overrides)
		slices.SortFunc(sortedOverrides, func(a, b Override) int {
			return cmp.Compare(selectorString(a.Match), selectorString(b.Match))
		})

		for _, override := range sortedOverrides {
			writeRateCard(h, "override="+selectorString(override.Match), override.RateCard)
		}
	}

	sum := h.Sum(nil)

	return hex.EncodeToString(sum[:8])
}

func writeRateCard(h io.Writer, label string, card RateCard) {
	writeRates(h, label+"/base", card.Base)

	for idx, bracket := range card.Brackets {
		fmt.Fprintf(h, "%s/bracket[%d]/min_context=%d\n", label, idx, bracket.MinContextTokens)
		writeRates(h, fmt.Sprintf("%s/bracket[%d]/rates", label, idx), bracket.Rates)
	}
}

func writeRates(h io.Writer, label string, rates Rates) {
	fmt.Fprintf(h, "%s:%d:%d:%d:%d:%d:%d\n",
		label,
		rates.InputPerMillion,
		rates.CachedInputPerMillion,
		rates.CacheCreation5mPerMillion,
		rates.CacheCreation1hPerMillion,
		rates.CacheCreationUnknownTTLPerMillion,
		rates.OutputPerMillion,
	)
}

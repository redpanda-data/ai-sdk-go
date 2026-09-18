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
	"maps"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// schemaVersion seeds the catalog version hash. Bump it when the
// calculator's resolution rules, rounding, or Cost shape changes so
// existing rate data produces a different Version() even if the numbers
// are identical.
//
// v2: keys are {provider, model} rather than bare model ID, and the
// version hash folds the provider in.
const schemaVersion = "v2"

// Source is what WithSource registers from; a provider's *catalog.Catalog
// satisfies it. It is an interface because pricing must not import
// catalog — the dependency runs the other way.
type Source interface {
	// Provider is the ProviderKey the source's models register under.
	Provider() llm.ProviderID
	PricingByID() map[string]Info
}

// Option configures a catalog during NewCatalog construction.
type Option func(*catalogBuilder)

// WithSource registers one source's model pricing under
// src.Provider(). Prefer it over WithProvider: the key comes from the
// source, so no caller hand-types it. A nil src, or one
// reporting an empty Provider(), fails the build.
func WithSource(src Source) Option {
	return func(b *catalogBuilder) {
		if src == nil {
			b.buildErrs = append(b.buildErrs,
				errors.New("WithSource given a nil source"))

			return
		}

		// A typed-nil *catalog.Catalog is not a nil interface; it lands here
		// with an empty Provider(). %T names it among joined build errors.
		if src.Provider() == "" {
			b.buildErrs = append(b.buildErrs,
				fmt.Errorf("WithSource: source %T reports an empty provider name (a nil catalog?)", src))

			return
		}

		b.registerModels(src.Provider(), src.PricingByID())
	}
}

// WithProvider registers one provider's model pricing under the given
// ProviderKey; it stays for callers holding a bare map, WithSource is
// preferred. The same model ID twice under one provider is a
// duplicate-pricing error from NewCatalog, never a silent clobber;
// intentional replacements go through WithOverride.
func WithProvider(provider ProviderKey, models map[string]Info) Option {
	return func(b *catalogBuilder) {
		b.registerModels(provider, models)
	}
}

func (b *catalogBuilder) registerModels(provider ProviderKey, models map[string]Info) {
	if provider == "" {
		b.buildErrs = append(b.buildErrs,
			errors.New("empty provider key"))

		return
	}

	// Record from the registration intent, not from the models that
	// survive normalization: a provider registered with an empty pricing
	// map is still known, so a later lookup reports ErrUnknownModel rather
	// than a false ErrUnknownProvider.
	b.providers[provider] = struct{}{}

	for id, info := range models {
		key := modelKey{provider: provider, model: id}
		if _, ok := b.models[key]; ok {
			b.buildErrs = append(b.buildErrs,
				fmt.Errorf("duplicate pricing for model %q under provider %q", id, provider))

			continue
		}

		b.models[key] = cloneInfo(info)
	}
}

// WithOverride replaces the pricing of an existing {provider, model}
// entry. An unknown pair, or the same pair passed twice, is an error
// from NewCatalog rather than last-writer-wins.
func WithOverride(provider ProviderKey, modelID string, info Info) Option {
	return func(b *catalogBuilder) {
		key := modelKey{provider: provider, model: modelID}
		if _, exists := b.overrides[key]; exists {
			b.buildErrs = append(b.buildErrs,
				fmt.Errorf("duplicate override for model %q under provider %q", modelID, provider))

			return
		}

		b.overrides[key] = cloneInfo(info)
	}
}

// NewCatalog constructs a validated pricing catalog from the given
// options. Options apply in order: sources and providers register
// their models, overrides replace individual entries, and the builder
// then normalizes and validates the result (duplicate keys, ambiguous
// selectors, malformed rates, etc.) — any of which surface as a joined
// error.
func NewCatalog(opts ...Option) (*Catalog, error) {
	b := &catalogBuilder{
		models:    make(map[modelKey]Info),
		overrides: make(map[modelKey]Info),
		providers: make(map[ProviderKey]struct{}),
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
	models    map[modelKey]Info
	overrides map[modelKey]Info
	providers map[ProviderKey]struct{}
	buildErrs []error
}

func (b *catalogBuilder) build() (*Catalog, error) {
	result := make(map[modelKey]Info, len(b.models))
	providers := maps.Clone(b.providers)
	errs := slices.Clone(b.buildErrs)

	for key, info := range b.models {
		normalized, err := normalizeInfo(info, string(key.provider)+"/"+key.model)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		result[key] = normalized
	}

	for key, override := range b.overrides {
		if _, ok := result[key]; !ok {
			errs = append(errs, fmt.Errorf("override for unknown model %q under provider %q", key.model, key.provider))
			continue
		}

		normalized, err := normalizeInfo(override, "override/"+string(key.provider)+"/"+key.model)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		result[key] = normalized
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	return &Catalog{
		models:    result,
		providers: providers,
		version:   computeVersion(result, providers),
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

	var errs []error

	for i := range overrides {
		for j := i + 1; j < len(overrides); j++ {
			a := overrides[i].Match
			b := overrides[j].Match

			if !selectorsOverlap(a, b) {
				continue
			}

			if selectorSpecificity(a) == selectorSpecificity(b) {
				errs = append(errs,
					fmt.Errorf("%s: ambiguous selectors %q and %q have the same specificity and overlap",
						scope, selectorString(a), selectorString(b)))
			}
		}
	}

	return errors.Join(errs...)
}

func computeVersion(models map[modelKey]Info, providers map[ProviderKey]struct{}) string {
	keys := make([]modelKey, 0, len(models))
	for key := range models {
		keys = append(keys, key)
	}

	slices.SortFunc(keys, func(a, b modelKey) int {
		return cmp.Or(
			cmp.Compare(a.provider, b.provider),
			cmp.Compare(a.model, b.model),
		)
	})

	h := sha256.New()
	fmt.Fprintf(h, "schema=%s\n", schemaVersion)

	// Fold in the known-provider set, not just providers reachable through
	// a model: an empty-map provider is still known and changes how a miss
	// is classified, so two catalogs differing only there must hash apart.
	for _, provider := range slices.Sorted(maps.Keys(providers)) {
		fmt.Fprintf(h, "known_provider=%q\n", provider)
	}

	for _, key := range keys {
		info := models[key]
		fmt.Fprintf(h, "provider=%q\nmodel=%q\n", key.provider, key.model)
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

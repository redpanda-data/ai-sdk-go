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
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Catalog is an immutable, validated view of one provider's model
// offerings. The zero value is unusable; construct with New. The name
// follows the context.Context / pricing.Catalog convention: it is the
// package's central concept.
type Catalog struct {
	provider  string
	offerings []Offering     // sorted by ID
	byID      map[string]int // offering ID -> index into offerings
	byAlias   map[string]int // alias -> index into offerings
	// prefixIDs holds every offering ID and alias sorted by descending
	// length, for longest-prefix resolution.
	prefixIDs []prefixCandidate
	facts     map[ModelID]Facts
	// bySeries maps a series to its member ModelIDs sorted by
	// Facts.Released ascending (ties broken by ModelID), restricted to
	// models offered in this catalog.
	bySeries map[string][]ModelID
}

type prefixCandidate struct {
	id  string
	idx int
}

// Option configures New.
type Option func(*config)

type config struct {
	registry Registry
}

// WithRegistry overrides the Facts registry used to resolve each entry's
// ModelID. The default is DefaultRegistry(). The map is defensively
// copied.
func WithRegistry(r Registry) Option {
	return func(c *config) {
		c.registry = maps.Clone(r)
	}
}

// New validates a provider's authored entries and freezes them into a
// Catalog.
//
// There is no external truth to check against — the entry IS the source
// of truth. Validation catches what the compiler cannot in hand-written
// literals:
//
//   - forgotten fields that compile to harmful zeros: token limits > 0,
//     input/output rates authored (zero means "unpriced"; free is
//     spelled pricing.RateFree)
//   - the same fact authored in two fields: Capabilities.Vision/Audio
//     must agree with Modalities.Input; Reasoning.Efforts requires
//     Capabilities.Reasoning
//   - cross-entry sanity: unique IDs and aliases; Model registered in
//     the Registry; ReplacedBy resolves; dates ordered and date-only;
//     deprecated/retired stages never authored (they derive from dates)
//
// Errors are joined and path-qualified:
//
//	catalog: anthropic: entries[2] "claude-sonnet-5": Constraints.MaxInputTokens must be > 0
func New(provider string, entries []Entry, opts ...Option) (*Catalog, error) {
	cfg := config{}
	for _, opt := range opts {
		opt(&cfg)
	}

	if cfg.registry == nil {
		cfg.registry = DefaultRegistry()
	}

	if provider == "" {
		return nil, errors.New("catalog: provider name is required")
	}

	c := &Catalog{
		provider: provider,
		byID:     make(map[string]int, len(entries)),
		byAlias:  make(map[string]int),
		facts:    make(map[ModelID]Facts),
		bySeries: make(map[string][]ModelID),
	}

	var errs []error

	fail := func(i int, e Entry, format string, args ...any) {
		prefix := fmt.Sprintf("catalog: %s: entries[%d] %q: ", provider, i, e.ID)
		errs = append(errs, fmt.Errorf(prefix+format, args...))
	}

	c.offerings = make([]Offering, 0, len(entries))

	for i, e := range entries {
		if e.ID == "" {
			fail(i, e, "ID is required")
			continue
		}

		if _, dup := c.byID[e.ID]; dup {
			fail(i, e, "duplicate offering ID")
			continue
		}

		facts, ok := cfg.registry[e.Model]
		switch {
		case e.Model == "":
			fail(i, e, "Model is required")
		case !ok:
			fail(i, e, "Model %q is not in the Facts registry", e.Model)
		case facts.Released.IsZero():
			fail(i, e, "Facts for %q have a zero Released date", e.Model)
		case facts.Series == "":
			fail(i, e, "Facts for %q have an empty Series", e.Model)
		case !isDateOnly(facts.Released) || !isDateOnly(facts.Knowledge):
			fail(i, e, "Facts dates for %q must be date-only (midnight UTC): construct with catalog.MustDate", e.Model)
		}

		// A zero limit is a forgotten field, not a choice.
		if e.Constraints.MaxInputTokens <= 0 {
			fail(i, e, "Constraints.MaxInputTokens must be > 0")
		}

		if e.Constraints.MaxOutputTokens <= 0 {
			fail(i, e, "Constraints.MaxOutputTokens must be > 0")
		}

		// A zero rate means "unpriced" downstream, so core rates must be
		// authored — a truly free rate is pricing.RateFree.
		if e.Pricing.Default.Base.InputPerMillion == 0 {
			fail(i, e, "Pricing.Default.Base.InputPerMillion is unpriced: author a rate or pricing.RateFree")
		}

		if e.Pricing.Default.Base.OutputPerMillion == 0 {
			fail(i, e, "Pricing.Default.Base.OutputPerMillion is unpriced: author a rate or pricing.RateFree")
		}

		// Normalize before validating: the shape checks compare the
		// capability booleans against Modalities, and an entry that omits
		// Modalities entirely only grows its text-only default here.
		// Validating first would let Vision-with-no-modalities through and
		// then normalize it into the exact contradiction validateShape
		// exists to reject.
		normalized := normalizeEntry(e, facts)

		validateLifecycle(normalized, func(format string, args ...any) { fail(i, e, format, args...) })
		validateShape(normalized, func(format string, args ...any) { fail(i, e, format, args...) })

		idx := len(c.offerings)
		// Clone on the way in as well as on the way out: the author keeps
		// their []Entry (and shared vars inside it), and the catalog must
		// not alias memory a caller can still write to.
		c.offerings = append(c.offerings, cloneOffering(Offering{
			Entry:    normalized,
			provider: provider,
			facts:    facts,
		}))
		c.byID[e.ID] = idx

		if ok {
			c.facts[e.Model] = facts
		}
	}

	// Alias registration and ReplacedBy resolution need the full ID set,
	// so they run after the first pass.
	for idx := range c.offerings {
		o := &c.offerings[idx]

		for _, alias := range o.Aliases {
			if alias == "" {
				errs = append(errs, fmt.Errorf("catalog: %s: entries %q: empty alias", provider, o.ID))
				continue
			}

			if _, clash := c.byID[alias]; clash {
				errs = append(errs, fmt.Errorf("catalog: %s: alias %q on %q collides with an offering ID", provider, alias, o.ID))
				continue
			}

			if prev, dup := c.byAlias[alias]; dup {
				errs = append(errs, fmt.Errorf("catalog: %s: alias %q on %q already registered on %q", provider, alias, o.ID, c.offerings[prev].ID))
				continue
			}

			c.byAlias[alias] = idx
		}

		if rb := o.Life.ReplacedBy; rb != "" {
			if _, ok := c.byID[rb]; !ok {
				errs = append(errs, fmt.Errorf("catalog: %s: %q: Life.ReplacedBy %q is not an offering in this catalog", provider, o.ID, rb))
			}
		}
	}

	// Pricing is validated by the pricing builder, which enforces rate
	// sanity and override consistency; its errors carry the model ID.
	if _, err := pricing.NewCatalog(pricing.WithProvider(provider, pricingMap(c.offerings))); err != nil {
		errs = append(errs, fmt.Errorf("catalog: %s: %w", provider, err))
	}

	if len(errs) > 0 {
		return nil, errors.Join(errs...)
	}

	c.freeze()

	return c, nil
}

// MustNew is New that panics on error. Intended for provider package
// initialization, where the entries are compile-time literals and every
// catalog is constructed by tests.
func MustNew(provider string, entries []Entry, opts ...Option) *Catalog {
	c, err := New(provider, entries, opts...)
	if err != nil {
		panic(err) //nolint:forbidigo // authoring error, not runtime
	}

	return c
}

// validateLifecycle checks date order and rejects authored derived
// stages: whether a date has passed must never depend on a flag someone
// remembered to flip.
func validateLifecycle(e Entry, fail func(string, ...any)) {
	l := e.Life

	switch l.Stage {
	case "", StagePreview, StageGA:
	case StageDeprecated, StageRetired:
		fail("Life.Stage %q is derived from dates and cannot be authored", l.Stage)
	default:
		fail("Life.Stage %q is not a valid stage", l.Stage)
	}

	// Enforce the midnight-UTC convention, so classification cannot
	// drift with an author's timezone.
	for name, d := range map[string]time.Time{
		"Available": l.Available, "Deprecated": l.Deprecated, "Retires": l.Retires,
	} {
		if !isDateOnly(d) {
			fail("Life.%s %s must be date-only (midnight UTC): construct with catalog.MustDate", name, d)
		}
	}

	if !l.Available.IsZero() && !l.Deprecated.IsZero() && l.Deprecated.Before(l.Available) {
		fail("Life.Deprecated %s is before Life.Available %s", dateString(l.Deprecated), dateString(l.Available))
	}

	if !l.Deprecated.IsZero() && !l.Retires.IsZero() && l.Retires.Before(l.Deprecated) {
		fail("Life.Retires %s is before Life.Deprecated %s", dateString(l.Retires), dateString(l.Deprecated))
	}

	if !l.Available.IsZero() && !l.Retires.IsZero() && l.Retires.Before(l.Available) {
		fail("Life.Retires %s is before Life.Available %s", dateString(l.Retires), dateString(l.Available))
	}
}

// validateShape rejects contradictions between overlapping fields: the
// capability booleans and the modality list state the same fact twice
// and must not drift apart.
//
// It runs on the normalized entry, so Modalities.Input is always
// populated — declaring Vision without listing ModalityImage is an
// authoring error even when the modality list was left implicit.
func validateShape(e Entry, fail func(string, ...any)) {
	// Reasoning without effort knobs exists (Claude Sonnet 4.5), so only
	// the reverse — controls without the capability — is impossible.
	if len(e.Reasoning.Efforts) > 0 && !e.Capabilities.Reasoning {
		fail("Reasoning.Efforts is set but Capabilities.Reasoning is false")
	}

	if (e.Reasoning.Adaptive || e.Reasoning.Budget) && !e.Capabilities.Reasoning {
		fail("Reasoning.Adaptive/Budget is set but Capabilities.Reasoning is false")
	}

	if e.Capabilities.Vision && !slices.Contains(e.Modalities.Input, ModalityImage) {
		fail("Capabilities.Vision is true but Modalities.Input lacks %q", ModalityImage)
	}

	if e.Capabilities.Audio && !slices.Contains(e.Modalities.Input, ModalityAudio) {
		fail("Capabilities.Audio is true but Modalities.Input lacks %q", ModalityAudio)
	}
}

// normalizeEntry applies authored-shape defaults: display label, explicit
// modalities, and GA stage.
func normalizeEntry(e Entry, facts Facts) Entry {
	if e.DisplayName == "" {
		e.DisplayName = facts.DisplayName
	}

	if len(e.Modalities.Input) == 0 {
		e.Modalities.Input = []Modality{ModalityText}
	}

	if len(e.Modalities.Output) == 0 {
		e.Modalities.Output = []Modality{ModalityText}
	}

	// Modalities are the canonical source for the Vision/Audio booleans:
	// an image or audio input modality derives the matching capability, so
	// authors write the fact once. (The reverse — an authored boolean with
	// no matching modality — is a contradiction validateShape rejects.)
	e.Capabilities.Vision = e.Capabilities.Vision || slices.Contains(e.Modalities.Input, ModalityImage)
	e.Capabilities.Audio = e.Capabilities.Audio || slices.Contains(e.Modalities.Input, ModalityAudio)

	if e.Life.Stage == "" {
		e.Life.Stage = StageGA
	}

	return e
}

// Provider returns the provider name this catalog was built for.
func (c *Catalog) Provider() string {
	if c == nil {
		return ""
	}

	return c.provider
}

// Len returns the number of offerings.
func (c *Catalog) Len() int {
	if c == nil {
		return 0
	}

	return len(c.offerings)
}

// Lookup returns the offering with the exact given ID. The result is a
// deep copy: an Offering embeds slices and maps (constraints, pricing
// overrides, aliases, attributes), and handing out the stored value
// would let a caller mutate the catalog — racing such a write against a
// concurrent read is a fatal concurrent map access, not a recoverable
// panic.
func (c *Catalog) Lookup(offeringID string) (Offering, bool) {
	if c == nil {
		return Offering{}, false
	}

	idx, ok := c.byID[offeringID]
	if !ok {
		return Offering{}, false
	}

	return cloneOffering(c.offerings[idx]), true
}

// Resolve maps a caller-supplied model string to an offering:
//
//  1. exact offering ID
//  2. exact alias
//  3. longest boundary-aware prefix over IDs and aliases, so snapshot
//     forms resolve to their family ("claude-sonnet-4-5-20250929" →
//     "claude-sonnet-4-5", "o3-2025-04-16" → "o3") and the longest
//     candidate wins ("claude-opus-4-5-..." resolves to claude-opus-4-5,
//     never claude-opus-4).
//
// A prefix only matches when the byte after the candidate is one of
// '-', '.', ':', or '@' AND the remainder starts with a digit — the
// shape of a snapshot or version stamp ("-20250929", ".1", "@001").
// Word suffixes never match: "gpt-5-chat-latest" is a different product
// than "gpt-5" and must resolve as unknown, not be silently binned into
// the family's metadata and pricing.
//
// ok == false means the catalog does not know this model. Callers must
// treat unknown as "stop enforcing", not "assume a baseline": no
// capabilities and no pricing may be guessed for it.
func (c *Catalog) Resolve(requested string) (Offering, bool) {
	if c == nil || requested == "" {
		return Offering{}, false
	}

	if idx, ok := c.byID[requested]; ok {
		return cloneOffering(c.offerings[idx]), true
	}

	if idx, ok := c.byAlias[requested]; ok {
		return cloneOffering(c.offerings[idx]), true
	}

	for _, cand := range c.prefixIDs {
		if len(requested) <= len(cand.id) {
			continue
		}

		if !strings.HasPrefix(requested, cand.id) {
			continue
		}

		switch requested[len(cand.id)] {
		case '-', '.', ':', '@':
			// Only snapshot/version-stamp shapes may prefix-match; see
			// the doc comment above.
			if rest := requested[len(cand.id)+1:]; len(rest) > 0 && rest[0] >= '0' && rest[0] <= '9' {
				return cloneOffering(c.offerings[cand.idx]), true
			}
		}
	}

	return Offering{}, false
}

// All returns every offering, deep-copied, sorted by offering ID.
// Retired offerings are included: the catalog is append-only so that
// historical usage stays priceable and retired rows stay renderable.
func (c *Catalog) All() []Offering {
	if c == nil {
		return nil
	}

	out := make([]Offering, len(c.offerings))
	for i, o := range c.offerings {
		out[i] = cloneOffering(o)
	}

	return out
}

// Facts returns the registered facts for a ModelID offered in this
// catalog.
func (c *Catalog) Facts(id ModelID) (Facts, bool) {
	if c == nil {
		return Facts{}, false
	}

	f, ok := c.facts[id]

	return f, ok
}

// Successor returns the next-newer model (by Facts.Released) in the same
// Series, restricted to models offered in this catalog. It is the
// derived complement of the authored Lifecycle.ReplacedBy: use ReplacedBy
// when the provider announced a replacement, Successor for generation
// math.
func (c *Catalog) Successor(id ModelID) (ModelID, bool) {
	if c == nil {
		return "", false
	}

	f, ok := c.facts[id]
	if !ok {
		return "", false
	}

	members := c.bySeries[f.Series]

	pos := slices.Index(members, id)
	if pos < 0 || pos+1 >= len(members) {
		return "", false
	}

	return members[pos+1], true
}

// PricingByID returns a model ID → pricing map covering every offering
// ID and every exact alias, in the shape pricing.NewCatalog's
// WithProvider expects:
//
//	pricing.NewCatalog(pricing.WithProvider(prov.Name(), prov.Catalog().PricingByID()))
//
// Aliases are included so exact-ID billing lookups keep working for
// alias requests; snapshot/timestamped IDs are not enumerable and must
// go through Resolve first (see the package example).
func (c *Catalog) PricingByID() map[string]pricing.Info {
	if c == nil {
		return nil
	}

	return pricingMap(c.offerings)
}

// freeze builds the derived indexes once validation has passed.
func (c *Catalog) freeze() {
	// Alias indexes point into the authored order; resolve them to owner
	// IDs before sorting invalidates them.
	aliasOwner := make(map[string]string, len(c.byAlias))
	for alias, idx := range c.byAlias {
		aliasOwner[alias] = c.offerings[idx].ID
	}

	slices.SortFunc(c.offerings, func(a, b Offering) int {
		return strings.Compare(a.ID, b.ID)
	})

	// Sorting invalidated the index maps; rebuild them.
	clear(c.byID)
	clear(c.byAlias)

	for idx, o := range c.offerings {
		c.byID[o.ID] = idx
	}

	for alias, ownerID := range aliasOwner {
		c.byAlias[alias] = c.byID[ownerID]
	}

	c.prefixIDs = make([]prefixCandidate, 0, len(c.byID)+len(c.byAlias))
	for id, idx := range c.byID {
		c.prefixIDs = append(c.prefixIDs, prefixCandidate{id: id, idx: idx})
	}

	for alias, idx := range c.byAlias {
		c.prefixIDs = append(c.prefixIDs, prefixCandidate{id: alias, idx: idx})
	}

	slices.SortFunc(c.prefixIDs, func(a, b prefixCandidate) int {
		if d := len(b.id) - len(a.id); d != 0 {
			return d
		}

		return strings.Compare(a.id, b.id)
	})

	series := make(map[string]map[ModelID]struct{})
	for id, f := range c.facts {
		if series[f.Series] == nil {
			series[f.Series] = make(map[ModelID]struct{})
		}

		series[f.Series][id] = struct{}{}
	}

	for s, members := range series {
		ids := slices.Collect(maps.Keys(members))
		slices.SortFunc(ids, func(a, b ModelID) int {
			fa, fb := c.facts[a], c.facts[b]
			if fa.Released.Before(fb.Released) {
				return -1
			}

			if fb.Released.Before(fa.Released) {
				return 1
			}

			return strings.Compare(string(a), string(b))
		})
		c.bySeries[s] = ids
	}
}

func pricingMap(offerings []Offering) map[string]pricing.Info {
	m := make(map[string]pricing.Info, len(offerings))
	for _, o := range offerings {
		m[o.ID] = o.Pricing.Clone()
		for _, alias := range o.Aliases {
			m[alias] = o.Pricing.Clone()
		}
	}

	return m
}

func cloneOffering(o Offering) Offering {
	o.Aliases = slices.Clone(o.Aliases)
	o.Constraints.SupportedParams = slices.Clone(o.Constraints.SupportedParams)

	// ConditionalRules nest a Disables slice; slices.Clone alone would
	// share its backing array across copies.
	o.Constraints.ConditionalRules = slices.Clone(o.Constraints.ConditionalRules)
	for i := range o.Constraints.ConditionalRules {
		o.Constraints.ConditionalRules[i].Disables = slices.Clone(o.Constraints.ConditionalRules[i].Disables)
	}

	if o.Constraints.MutuallyExclusive != nil {
		groups := make([][]string, len(o.Constraints.MutuallyExclusive))
		for i, g := range o.Constraints.MutuallyExclusive {
			groups[i] = slices.Clone(g)
		}

		o.Constraints.MutuallyExclusive = groups
	}

	o.Modalities.Input = slices.Clone(o.Modalities.Input)
	o.Modalities.Output = slices.Clone(o.Modalities.Output)
	o.Reasoning.Efforts = slices.Clone(o.Reasoning.Efforts)
	o.Speeds = slices.Clone(o.Speeds)
	o.Pricing = o.Pricing.Clone()
	o.Attributes = maps.Clone(o.Attributes)

	return o
}

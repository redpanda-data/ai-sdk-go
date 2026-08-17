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

// View is a time-dependent classification of a Catalog. Every predicate
// that depends on the clock lives here and nowhere else, so nothing on
// Catalog silently changes meaning when a retirement date passes —
// derived artifacts built from Catalog alone stay reproducible.
//
// Callers that mean "now" say so at the call site: c.Now() or
// c.At(catalog.Today()).
type View struct {
	catalog *Catalog
	asOf    Date
}

// At returns the classification of the catalog as of the given date.
func (c *Catalog) At(asOf Date) View {
	return View{catalog: c, asOf: asOf}
}

// Now returns the classification as of today (UTC).
func (c *Catalog) Now() View {
	return c.At(Today())
}

// AsOf returns the date this view classifies at.
func (v View) AsOf() Date {
	return v.asOf
}

// Stage returns the derived lifecycle stage of an offering at the view's
// date. Precedence: retired > deprecated > authored stage.
//
// Retirement derives exclusively from Lifecycle.Retires (inclusive
// boundary). A published RetirementNotBefore floor never retires an
// offering — a lower bound is not a shutdown date.
func (v View) Stage(offeringID string) (Stage, bool) {
	if v.catalog == nil {
		return "", false
	}

	idx, ok := v.catalog.byID[offeringID]
	if !ok {
		return "", false
	}

	return v.stageOf(v.catalog.offerings[idx]), true
}

// IsRetired reports whether the offering's announced shutdown date has
// arrived. Unknown offerings report false.
func (v View) IsRetired(offeringID string) bool {
	s, ok := v.Stage(offeringID)
	return ok && s == StageRetired
}

// IsDeprecated reports whether the offering is deprecated (but not yet
// retired) at the view's date.
func (v View) IsDeprecated(offeringID string) bool {
	s, ok := v.Stage(offeringID)
	return ok && s == StageDeprecated
}

// Current returns every offering of the newest model in each Series that
// is not retired at the view's date.
//
// Generation is computed per logical ModelID and then mapped back to
// offerings, so a model served through several variants (Bedrock geo
// profiles) contributes all of them rather than having them compete.
// When every offering of a series' newest model is retired, the
// next-newest model with a live offering represents the series.
//
// Generation and lifecycle are orthogonal: a deprecated offering of the
// newest model still appears here (it IS the current generation; render
// the deprecation badge alongside), while an old-generation GA offering
// appears in Previous, not here.
func (v View) Current() []Offering {
	return v.collect(func(o Offering, currentModel ModelID) bool {
		return o.Model == currentModel && v.stageOf(o) != StageRetired
	})
}

// Previous returns every available (non-retired) offering that is not of
// its series' current generation — regardless of lifecycle stage.
func (v View) Previous() []Offering {
	return v.collect(func(o Offering, currentModel ModelID) bool {
		return o.Model != currentModel && v.stageOf(o) != StageRetired
	})
}

// Deprecated returns every offering whose deprecation date has arrived
// and whose retirement has not.
func (v View) Deprecated() []Offering {
	return v.collect(func(o Offering, _ ModelID) bool {
		return v.stageOf(o) == StageDeprecated
	})
}

// Retired returns every offering whose announced shutdown date has
// arrived.
func (v View) Retired() []Offering {
	return v.collect(func(o Offering, _ ModelID) bool {
		return v.stageOf(o) == StageRetired
	})
}

func (v View) stageOf(o Offering) Stage {
	l := o.Life

	if !l.Retires.IsZero() && !v.asOf.Before(l.Retires) {
		return StageRetired
	}

	if !l.Deprecated.IsZero() && !v.asOf.Before(l.Deprecated) {
		return StageDeprecated
	}

	return l.Stage
}

// collect walks the offerings in catalog order, passing each one the
// current-generation ModelID of its series, and deep-copies matches.
func (v View) collect(match func(Offering, ModelID) bool) []Offering {
	if v.catalog == nil {
		return nil
	}

	current := v.currentModels()

	var out []Offering

	for _, o := range v.catalog.offerings {
		if match(o, current[o.facts.Series]) {
			out = append(out, cloneOffering(o))
		}
	}

	return out
}

// currentModels resolves each series to its current-generation ModelID
// at the view's date: the newest-released member that still has at least
// one non-retired offering.
func (v View) currentModels() map[string]ModelID {
	liveModels := make(map[ModelID]bool)

	for _, o := range v.catalog.offerings {
		if v.stageOf(o) != StageRetired {
			liveModels[o.Model] = true
		}
	}

	current := make(map[string]ModelID, len(v.catalog.bySeries))

	for series, members := range v.catalog.bySeries {
		// members are sorted by Released ascending; walk from newest.
		for i := len(members) - 1; i >= 0; i-- {
			if liveModels[members[i]] {
				current[series] = members[i]
				break
			}
		}
	}

	return current
}

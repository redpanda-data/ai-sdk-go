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

package main

import (
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// lockedKeys lists, for each field in RULES.md's Field table, the snapshot
// keys a lock covers: "key" or "key.subkey" on the offering, or "facts.key"
// on its model. A field with one key locks that value; one with several, or a
// decision's own keys, lock an object of those present, keyed by their last
// part.
var lockedKeys = map[string][]string{
	reconcile.FieldNewModel: nil,
	reconcile.FieldNewCopy:  nil,
	"price":                 {"pricing"},
	"limit":                 {"constraints.max_input_tokens", "constraints.max_output_tokens"},
	"capability":            {"capabilities", "modalities", "reasoning", "speeds"},
	"lifecycle":             {"lifecycle.stage", "lifecycle.available", "lifecycle.deprecated", "lifecycle.retires"},
	"replacement":           {"lifecycle.replaced_by"},
	"release_date":          {"facts.released"},
	"knowledge_cutoff":      {"facts.knowledge"},
	// Served locations aren't in the snapshot, so malformed refuses a
	// decision on them.
	"location": nil,
}

var decisionID = regexp.MustCompile(`^DEC-[0-9]{3}$`)

// checkDecisions returns one problem per decision that is malformed or that
// the snapshot contradicts. asks maps each RULES.md Ask to its question.
func checkDecisions(file, snapshotJSON []byte, asks map[string]string) []string {
	entries, err := reconcile.ParseDecisions(file)
	if err != nil {
		return []string{"decisions.yaml: " + err.Error()}
	}

	snap, err := reconcile.ParseSnapshot(snapshotJSON)
	if err != nil {
		return []string{err.Error()}
	}

	var problems []string

	ids, taken := map[string]bool{}, map[string][]reconcile.Decision{}

	for _, d := range entries {
		key := d.Offering + " " + d.Field
		// Two decisions on one offering and field clash when one answers a
		// question the other does, so the run would pick by file order.
		clash := slices.IndexFunc(taken[key], func(o reconcile.Decision) bool {
			return o.Answers(d.Rule) || d.Answers(o.Rule)
		})

		switch problem := malformed(d, asks); {
		case problem != "":
			problems = append(problems, d.ID+": "+problem)
			continue
		case ids[d.ID]:
			problems = append(problems, d.ID+": duplicate ID")
			continue
		case clash >= 0:
			problems = append(problems, d.ID+": "+taken[key][clash].ID+" already covers "+key)
			continue
		}

		ids[d.ID], taken[key] = true, append(taken[key], d)

		pkg, id, _ := strings.Cut(d.Offering, ":")
		offering, exists := snap.Offering(pkg, id)
		where := fmt.Sprintf("%s %s %s %s: ", d.ID, d.Kind, d.Offering, d.Field)
		// A hold on new_model or new_copy keeps the offering out; every
		// other decision is about an offering the catalogue carries.
		out := d.Kind == reconcile.KindHold && adds(d.Field)

		switch {
		case out && exists:
			problems = append(problems, where+"the offering is in the catalogue")
		case !out && !exists:
			problems = append(problems, where+"the offering is missing")
		case d.Lock != nil:
			if got, want := canonical(lockedPart(d, offering, snap.Facts)), canonical(d.Lock); got != want {
				problems = append(problems, where+"got "+got+", locked "+want)
			}
		}
	}

	return problems
}

func malformed(d reconcile.Decision, asks map[string]string) string {
	pkg, id, _ := strings.Cut(d.Offering, ":")
	_, knownField := lockedKeys[d.Field]
	added := adds(d.Field)

	switch {
	case !decisionID.MatchString(d.ID):
		return "an ID is DEC- and three digits"
	case d.Kind != reconcile.KindKeep && d.Kind != reconcile.KindHold:
		return "kind is keep or hold"
	case reconcile.PackageProviders[pkg] == "":
		return "unknown package " + pkg
	case id == "":
		return "an offering is <package>:<model ID>"
	case !knownField:
		return "unknown field " + d.Field
	case d.Field == "location":
		return "location takes no decision yet: served locations are not in the snapshot"
	case d.Kind == reconcile.KindHold && !added && d.Rule == "":
		return "a hold on " + d.Field + " names the rule whose question it answers"
	case added && d.Lock != nil:
		return d.Field + " takes no lock"
	case added && d.Rule != "":
		// It answers every question and skip about its offering, whatever
		// rule raised it; skipped rows name Defaults, which a decision can't.
		return d.Field + " takes no rule"
	case d.Kind == reconcile.KindHold && d.Rule == reconcile.JudgmentRule:
		return "a hold can't name no-rule: it would hide every question no rule covers on " + d.Field + ", so add a rule to RULES.md"
	case d.Kind == reconcile.KindHold && d.Lock != nil:
		return "a hold takes no lock"
	case d.Kind == reconcile.KindKeep && !added && d.Lock == nil:
		return "a keep needs a lock"
	case d.Rule != "" && asks[d.Rule] == "":
		return "rule " + d.Rule + " asks no question"
	case len(d.Keys) > 0 && (d.Kind != reconcile.KindKeep || added):
		return "only a keep on a value names keys"
	case d.Reason == "":
		return "a decision needs a reason"
	}

	if problem := badDates(d); problem != "" {
		return problem
	}

	return badKeys(d)
}

// badDates returns what is wrong with a decision's dates: decided and
// review_by are dates, and review_by is at most a year after decided.
func badDates(d reconcile.Decision) string {
	decided, decidedErr := catalog.ParseDate(d.Decided)
	reviewBy, reviewErr := catalog.ParseDate(d.ReviewBy)

	switch {
	case decidedErr != nil || reviewErr != nil:
		return "decided and review_by are YYYY-MM-DD dates"
	case !reviewBy.After(decided) || reviewBy.After(decided.AddDate(1, 0, 0)):
		return "review_by is after decided, and at most a year after it"
	}

	return ""
}

// badKeys returns what is wrong with a keep's keys: each is part of its
// field, and no two lock the same name.
func badKeys(d reconcile.Decision) string {
	names := map[string]string{}

	for _, key := range d.Keys {
		name := key[strings.LastIndex(key, ".")+1:]

		switch {
		case !inField(d.Field, key):
			return "key " + key + " isn't part of " + d.Field + ": name one of its snapshot keys, or one level under one"
		case names[name] != "":
			return "keys " + names[name] + " and " + key + " would both lock " + name
		}

		names[name] = key
	}

	return ""
}

// adds reports whether a field is about an offering the catalogue doesn't
// carry yet: a new model, or a new copy of one.
func adds(field string) bool {
	return field == reconcile.FieldNewModel || field == reconcile.FieldNewCopy
}

// inField reports whether key is one of field's snapshot keys (lockedKeys),
// or one level under one that is a whole offering key, as lockedPart's
// lookup reads it.
func inField(field, key string) bool {
	parent, child, nested := strings.Cut(key, ".")

	for _, path := range lockedKeys[field] {
		if key == path || nested && parent == path && child != "" && !strings.Contains(child, ".") {
			return true
		}
	}

	return false
}

// lockedPart returns the part of the snapshot a decision locks: the parts
// its keys name, or else its field's (see lockedKeys).
func lockedPart(d reconcile.Decision, offering map[string]any, facts map[string]map[string]any) any {
	model, _ := offering["model"].(string)
	lookup := func(path string) (any, bool) {
		first, rest, nested := strings.Cut(path, ".")
		if first == "facts" {
			value, ok := facts[model][rest]
			return value, ok
		}

		value, ok := offering[first]
		if nested {
			object, _ := value.(map[string]any)
			value, ok = object[rest]
		}

		return value, ok
	}

	paths := lockedKeys[d.Field]
	if len(d.Keys) > 0 {
		paths = d.Keys
	} else if len(paths) == 1 {
		value, _ := lookup(paths[0])
		return value
	}

	part := map[string]any{}

	for _, path := range paths {
		if value, ok := lookup(path); ok {
			part[path[strings.LastIndex(path, ".")+1:]] = value
		}
	}

	return part
}

// canonical is a value's JSON with sorted keys, so a lock written in YAML and
// a value decoded from the snapshot compare equal when they hold the same data.
func canonical(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprintf("<%T that isn't JSON>", value)
	}

	return string(raw)
}

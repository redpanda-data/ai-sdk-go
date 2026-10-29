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
	"maps"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

const inferenceGeo = "inference_geo"

// rateNames are the base rates a price note names, in this order.
var rateNames = []struct{ key, name string }{
	{"input_microcents_per_million", "input"},
	{"output_microcents_per_million", "output"},
	{"cached_input_microcents_per_million", "cache read"},
	{"cache_creation_5m_microcents_per_million", "cache write 5m"},
	{"cache_creation_1h_microcents_per_million", "cache write 1h"},
}

// effect is a change row's note on what it does, written from the catalogue
// before and after the patch, never from the report's values. It is empty
// when there is nothing to add. The patched catalogue comes from code the
// agent wrote, so every string read from it passes safeValue (see safeText).
func (base Base) effect(change Change, provider string) string {
	pkg, id := reconcile.PackageOf(change.Offering, provider)
	before, _ := base.Before.Offering(pkg, id)
	after, found := base.After.Offering(pkg, id)

	if !found {
		return ""
	}

	switch change.Field {
	case "price":
		return priceNote(before, after)
	case reconcile.FieldNewModel, reconcile.FieldNewCopy:
		if before != nil {
			return ""
		}

		if geo := attribute(after, inferenceGeo); geo != "" {
			return "new on " + pkg + ", region " + code(geo)
		}

		return "new on " + pkg
	case "lifecycle":
		return base.retirementNote(after)
	case "replacement":
		return base.replacementNote(pkg, after)
	}

	return ""
}

// unreported names each offering the patch adds, changes or removes, and
// each model whose facts it changes, that no report row names, as the report
// would (RULES.md, Report values). Derived values change with their
// neighbours, so they don't count. IDs that aren't safe to print are counted.
func (base Base) unreported(report Report, provider string) []string {
	if base.Before == nil || base.After == nil {
		return nil
	}

	named, models := map[[2]string]bool{}, map[string]bool{}
	name := func(offering string) {
		pkg, id := reconcile.PackageOf(offering, provider)
		named[[2]string{pkg, id}] = true

		for _, snapshot := range []*reconcile.Snapshot{base.Before, base.After} {
			if o, found := snapshot.Offering(pkg, id); found {
				models[safeText(o["model"])] = true
			}
		}
	}

	for _, row := range report.Changes {
		name(row.Offering)
	}

	for _, row := range report.NeedsHuman {
		name(row.Offering)
	}

	for _, row := range report.Skipped {
		name(row.Offering)
	}

	var names []string

	unprintable := 0

	for pkg := range reconcile.PackageProviders {
		ids := map[string]bool{}

		for _, snapshot := range []*reconcile.Snapshot{base.Before, base.After} {
			for _, o := range snapshot.Offerings(pkg) {
				id, _ := o["id"].(string)
				ids[id] = true
			}
		}

		for id := range ids {
			before, _ := base.Before.Offering(pkg, id)
			after, _ := base.After.Offering(pkg, id)

			switch {
			case named[[2]string{pkg, id}] || reflect.DeepEqual(reconcile.WithoutDerived(before), reconcile.WithoutDerived(after)):
			case safeText(id) == "":
				unprintable++
			default:
				label := pkg + ":" + id
				if pkg == provider {
					label = id
				}

				if after == nil {
					label += " (removed)"
				}

				names = append(names, label)
			}
		}
	}

	for model := range base.After.Facts {
		if !models[model] && safeText(model) != "" && !reflect.DeepEqual(base.Before.Facts[model], base.After.Facts[model]) {
			names = append(names, model+" (facts)")
		}
	}

	slices.Sort(names)

	switch {
	case unprintable == 1:
		names = append(names, "1 offering with an ID that isn't safe to print")
	case unprintable > 1:
		names = append(names, fmt.Sprintf("%d offerings with IDs that aren't safe to print", unprintable))
	}

	return names
}

// priceNote names each base rate that changed, old → new in dollars per
// million tokens. A change only to brackets or overrides gets a line too.
func priceNote(before, after map[string]any) string {
	old, now := baseRates(before), baseRates(after)

	var notes []string

	for _, rate := range rateNames {
		switch {
		case now[rate.key] == old[rate.key]:
		case old[rate.key] > 0:
			notes = append(notes, fmt.Sprintf("%s %s → %s (%s)", rate.name, dollars(old[rate.key]), dollars(now[rate.key]), percent(old[rate.key], now[rate.key])))
		default:
			notes = append(notes, fmt.Sprintf("%s %s → %s", rate.name, dollars(old[rate.key]), dollars(now[rate.key])))
		}
	}

	if len(notes) == 0 && before != nil && !reflect.DeepEqual(before["pricing"], after["pricing"]) {
		return "brackets or overrides changed"
	}

	return strings.Join(notes, ", ")
}

func (base Base) retirementNote(after map[string]any) string {
	life, _ := after["lifecycle"].(map[string]any)
	retires := safeText(life["retires"])

	day, err := time.Parse(time.DateOnly, retires)
	if err != nil {
		return ""
	}

	today, err := time.Parse(time.DateOnly, base.Today)
	if err != nil {
		return "retires " + retires
	}

	if day.Before(today) {
		return "retired " + retires
	}

	return fmt.Sprintf("retires %s, in %d days", retires, int(day.Sub(today).Hours()/24))
}

// replacementNote flags a replacement with no copy in the retiring copy's
// region: callers who follow it leave that region. It names the input price
// change and what the replacement can't do that the retiring copy could.
func (base Base) replacementNote(pkg string, after map[string]any) string {
	geo := attribute(after, inferenceGeo)
	life, _ := after["lifecycle"].(map[string]any)
	target, found := base.After.Offering(pkg, safeText(life["replaced_by"]))
	model := safeText(target["model"])

	if geo == "" || !found || model == "" || slices.ContainsFunc(base.After.Offerings(pkg), func(o map[string]any) bool {
		return o["model"] == model && attribute(o, inferenceGeo) == geo
	}) {
		return ""
	}

	note := fmt.Sprintf("⚑ no %s copy of %s: callers who follow it leave %s", code(geo), code(model), code(geo))

	if change := inputChange(after, target); change != "" {
		note += "; " + change
	}

	if lost := lostFeatures(after, target); len(lost) > 0 {
		note += "; loses " + valueList(lost)
	}

	return note
}

// lostFeatures lists what the source offering has and the target doesn't:
// capabilities, input and output kinds, and request parameters, sorted.
func lostFeatures(source, target map[string]any) []string {
	lost := map[string]bool{}

	sourceCaps, _ := source["capabilities"].(map[string]any)
	targetCaps, _ := target["capabilities"].(map[string]any)

	for name, on := range sourceCaps {
		if on == true && targetCaps[name] != true && safeValue(name) {
			lost[name] = true
		}
	}

	for _, direction := range []string{"input", "output"} {
		for _, kind := range texts(source, "modalities", direction) {
			if !slices.Contains(texts(target, "modalities", direction), kind) {
				lost[kind+" "+direction] = true
			}
		}
	}

	for _, param := range texts(source, "constraints", "supported_params") {
		if !slices.Contains(texts(target, "constraints", "supported_params"), param) {
			lost[param] = true
		}
	}

	return slices.Sorted(maps.Keys(lost))
}

// safeText is a snapshot value as a string, or "" when it isn't one or isn't
// safe to put in the PR body.
func safeText(value any) string {
	s, _ := value.(string)
	if !safeValue(s) {
		return ""
	}

	return s
}

// texts reads a list of strings at object.key, keeping the safe ones.
func texts(offering map[string]any, object, key string) []string {
	parent, _ := offering[object].(map[string]any)
	list, _ := parent[key].([]any)

	var values []string

	for _, item := range list {
		if value := safeText(item); value != "" {
			values = append(values, value)
		}
	}

	return values
}

// attribute reads one of an offering's key/value attributes.
func attribute(offering map[string]any, key string) string {
	list, _ := offering["attributes"].([]any)

	for _, item := range list {
		pair, _ := item.(map[string]any)
		if pair["key"] == key {
			return safeText(pair["value"])
		}
	}

	return ""
}

// baseRates reads an offering's default base rates, in microcents.
func baseRates(offering map[string]any) map[string]int64 {
	pricing, _ := offering["pricing"].(map[string]any)
	defaults, _ := pricing["default"].(map[string]any)
	base, _ := defaults["base"].(map[string]any)
	rates := map[string]int64{}

	for key, raw := range base {
		if number, ok := raw.(json.Number); ok {
			if value, err := number.Int64(); err == nil {
				rates[key] = value
			}
		}
	}

	return rates
}

// inputChange is "input ±N%" between two offerings' base input rates, or ""
// when either has none.
func inputChange(from, to map[string]any) string {
	old, now := baseRates(from)["input_microcents_per_million"], baseRates(to)["input_microcents_per_million"]
	if old <= 0 || now <= 0 {
		return ""
	}

	return "input " + percent(old, now)
}

// dollars writes a microcent rate as dollars per million tokens.
func dollars(microcents int64) string {
	return strconv.FormatFloat(float64(microcents)/1e8, 'f', -1, 64)
}

// percent is the change from one rate to another, rounded, with one decimal
// below 1% so a small change doesn't read as none.
func percent(from, to int64) string {
	change := float64(to-from) * 100 / float64(from)
	if math.Abs(change) < 1 {
		return fmt.Sprintf("%+.1f%%", change)
	}

	return fmt.Sprintf("%+.0f%%", change)
}

// impact is what one kind of change means for customers, traced from the
// code that reads each field in ai-sdk-go and in ADP: its key and its name in
// the impact table, a short form for a change row's Effect cell, a sentence
// for the table, whether it is easy to revert, and whether it changes
// requests or money (acts), which the table puts in bold.
type impact struct {
	key, name, short, long, revert string
	acts                           bool
}

// impacts is every kind of change, in the impact table's order.
var impacts = []impact{
	{"price down", "price cut", "", "New calls cost less after the SDK upgrade or ADP deploy; past spend stays; ADP budgets fill more slowly.", "yes", true},
	{"price up", "price rise", "", "New calls cost more after the SDK upgrade or ADP deploy; past spend stays; ADP budgets block sooner.", "yes", true},
	{"price", "price change", "", "New calls cost more or less on some rates, tiers, regions or context sizes; past spend stays.", "yes", true},
	{"max input", "max input tokens", "agents compact at the new window", "Agents compact at the new window and send requests up to it; the ADP console shows it. If the real limit is lower, calls fail once, then retry trimmed. Rates unchanged.", "yes", true},
	{"max output", "max output tokens", "changes the largest max_tokens accepted", "Changes the largest max_tokens the Bedrock and Anthropic providers and the ADP playground accept; agents and costs unchanged, except Bedrock Claude agents.", "yes", true},
	{"flag off", "capability turned off", "labels only", "Labels only: requests and costs unchanged; code that checks the flag stops using the feature.", "yes", false},
	{"flag on", "capability turned on", "labels; code that checks it starts using it", "Labels only for requests; code that checks the flag starts using the feature, which the vendor may still reject.", "yes", true},
	{"modality", "input or output kinds", "none", "None: nothing in the SDK or ADP reads modalities.", "–", false},
	{"reasoning", "reasoning options", "setup fails for a removed effort", "Removing an effort makes SDK model setup and ADP agent saves fail for it.", "yes", true},
	{"launch date", "launch date", "none (label)", "None: nothing reads the launch date.", "–", false},
	{"deprecation", "deprecation date", "label only", "Label only for customers; CI's live-test check may fail.", "yes", false},
	{"retirement", "shutdown date", "", "From that date ADP's model list hides it; calls work until the vendor shuts it down.", "yes", true},
	{"new offering", "new offering", "new ID: callable, listed, priced", "The new ID becomes callable, listed and priced; before, the SDK refused it.", "no, once used", true},
	{"replacement", "replacement", "", "Migration hint only; no request moves to the replacement.", "yes", false},
	{"other", "other", "", "Not mapped yet: read the row.", "check", true},
}

// kindIndex is a kind's place in impacts.
func kindIndex(key string) int {
	return slices.IndexFunc(impacts, func(kind impact) bool { return kind.key == key })
}

// sub reads one of an offering's objects.
func sub(offering map[string]any, key string) map[string]any {
	object, _ := offering[key].(map[string]any)
	return object
}

// kinds names what a change row changed, from the offering before and after,
// never from the report's values, in the impact table's order.
func kinds(field string, before, after map[string]any) []string {
	var found []string

	add := func(kind string) {
		if !slices.Contains(found, kind) {
			found = append(found, kind)
		}
	}

	switch field {
	case "price":
		old, now := baseRates(before), baseRates(after)
		up, down := false, false

		for key, rate := range now {
			up, down = up || rate > old[key], down || rate < old[key]
		}

		switch {
		case up && !down:
			add("price up")
		case down && !up:
			add("price down")
		default:
			add("price")
		}
	case "limit":
		b, a := sub(before, "constraints"), sub(after, "constraints")
		if !reflect.DeepEqual(b["max_input_tokens"], a["max_input_tokens"]) {
			add("max input")
		}

		if !reflect.DeepEqual(b["max_output_tokens"], a["max_output_tokens"]) {
			add("max output")
		}
	case "capability":
		b, a := sub(before, "capabilities"), sub(after, "capabilities")
		for key, now := range a {
			switch {
			case b[key] == true && now == false:
				add("flag off")
			case b[key] == false && now == true:
				add("flag on")
			}
		}

		if !reflect.DeepEqual(before["modalities"], after["modalities"]) {
			add("modality")
		}

		if !reflect.DeepEqual(before["reasoning"], after["reasoning"]) {
			add("reasoning")
		}
	case "lifecycle":
		b, a := sub(before, "lifecycle"), sub(after, "lifecycle")
		for _, date := range [][2]string{{"available", "launch date"}, {"deprecated", "deprecation"}, {"retires", "retirement"}} {
			if !reflect.DeepEqual(b[date[0]], a[date[0]]) {
				add(date[1])
			}
		}

		if !reflect.DeepEqual(b["stage"], a["stage"]) && a["stage"] == "deprecated" {
			add("deprecation")
		}
	case reconcile.FieldNewModel, reconcile.FieldNewCopy:
		add("new offering")
	case "replacement":
		add("replacement")
	}

	if len(found) == 0 {
		add("other")
	}

	slices.SortFunc(found, func(x, y string) int { return kindIndex(x) - kindIndex(y) })

	return found
}

// rowKinds is kinds for one change row in the vendor's catalogue, or none
// without one.
func (base Base) rowKinds(change Change, provider string) []string {
	if base.Before == nil || base.After == nil {
		return nil
	}

	pkg, id := reconcile.PackageOf(change.Offering, provider)
	before, _ := base.Before.Offering(pkg, id)
	after, _ := base.After.Offering(pkg, id)

	return kinds(change.Field, before, after)
}

// rowEffect is a change row's Effect cell: its price, date or region note
// when it has one, else the short form of each kind.
func (base Base) rowEffect(change Change, provider string) string {
	if note := base.effect(change, provider); note != "" {
		return note
	}

	var shorts []string

	for _, kind := range base.rowKinds(change, provider) {
		if short := impacts[kindIndex(kind)].short; short != "" {
			shorts = append(shorts, short)
		}
	}

	return strings.Join(shorts, "; ")
}

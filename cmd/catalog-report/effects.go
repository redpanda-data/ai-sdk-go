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
// agent wrote, so every string read from it passes safeValue (see text).
func (base Base) effect(change Change, provider string) string {
	pkg, id := packageOf(change.Offering, provider)
	before, _ := base.Before.Offering(pkg, id)
	after, found := base.After.Offering(pkg, id)

	if !found {
		return ""
	}

	switch change.Field {
	case "price":
		return priceNote(before, after)
	case reconcile.FieldNewModel:
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

// packageOf names a report offering's package and ID: <package>:<id>, or an
// ID in the run's own package (RULES.md R2). Bedrock IDs contain ":", so only
// a known package counts as a prefix.
func packageOf(offering, provider string) (string, string) {
	if pkg, id, found := strings.Cut(offering, ":"); found && reconcile.PackageProviders[pkg] != "" {
		return pkg, id
	}

	return provider, offering
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
	retires := text(life["retires"])

	day, err := time.Parse(time.DateOnly, retires)
	if err != nil {
		return ""
	}

	today, err := time.Parse(time.DateOnly, base.Today)
	if err != nil {
		return "retires " + retires
	}

	return fmt.Sprintf("retires %s, in %d days", retires, int(day.Sub(today).Hours()/24))
}

// replacementNote flags a replacement with no copy in the retiring copy's
// region: callers who follow it leave that region. It names the input price
// change and what the replacement can't do that the retiring copy could.
func (base Base) replacementNote(pkg string, after map[string]any) string {
	geo := attribute(after, inferenceGeo)
	life, _ := after["lifecycle"].(map[string]any)
	target, found := base.After.Offering(pkg, text(life["replaced_by"]))
	model := text(target["model"])

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

// text is a snapshot value as a string, or "" when it isn't one or isn't
// safe to put in the PR body.
func text(value any) string {
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
		if value := text(item); value != "" {
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
			return text(pair["value"])
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

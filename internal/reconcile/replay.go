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

package reconcile

import (
	"reflect"
	"slices"
)

// replayItem is one thing a snapshot holds: an offering of a snapshot
// provider, or a model's facts when provider is empty.
type replayItem struct {
	provider, id string
}

// value is the item in s, without derived values; nil when s lacks it.
func (s Snapshot) value(item replayItem) any {
	if item.provider == "" {
		if facts, found := s.Facts[item.id]; found {
			return facts
		}

		return nil
	}

	for _, provider := range s.Providers {
		if provider.Provider != item.provider {
			continue
		}

		for _, offering := range provider.Offerings {
			if offering["id"] == item.id {
				return WithoutDerived(offering)
			}
		}
	}

	return nil
}

// label names the item as a report would, from the run of vendor.
func (item replayItem) label(vendor string) string {
	if item.provider == "" {
		return item.id + " (facts)"
	}

	for pkg, provider := range PackageProviders {
		if provider == item.provider {
			if pkg == vendor {
				return item.id
			}

			return pkg + ":" + item.id
		}
	}

	return item.provider + ":" + item.id
}

// ReplayMismatches checks that a vendor's commit, replayed onto a newer
// base with no model, is exactly the original change: parent is the
// catalogue the bot's commit was made on, bot is that commit's, base is the
// replay's base and replay is the replay's, all as committed. It returns
// why it isn't, one line per item, or nothing when it is.
//
//   - Every offering or model's facts the bot's commit changed must be
//     unchanged on the base since, and the replay must make the same change.
//   - Every offering a report row names, its model's facts and every other
//     offering of that model must be unchanged on the base since, so no row
//     or question outlives its reason: the agent wrote it looking at every
//     host's copy.
//   - The replay must change nothing else.
//
// Derived values change with their neighbours, so they don't count.
func ReplayMismatches(parent, bot, base, replay Snapshot, report Report, vendor string) []string {
	snapshots := []Snapshot{parent, bot, base, replay}
	items := map[replayItem]bool{}

	for _, snap := range snapshots {
		for model := range snap.Facts {
			items[replayItem{id: model}] = true
		}

		for _, provider := range snap.Providers {
			for _, offering := range provider.Offerings {
				if id, ok := offering["id"].(string); ok {
					items[replayItem{provider: provider.Provider, id: id}] = true
				}
			}
		}
	}

	// modelOf is the model an offering is a copy of, from the first of parent,
	// bot and base that has it; "" for a model's facts.
	modelOf := func(item replayItem) string {
		for _, snap := range snapshots[:3] {
			if values, ok := snap.value(item).(map[string]any); ok {
				if model, ok := values["model"].(string); ok {
					return model
				}
			}
		}

		return ""
	}

	named := map[replayItem]bool{}

	name := func(offering string) {
		pkg, id := PackageOf(offering, vendor)
		item := replayItem{provider: PackageProviders[pkg], id: id}
		named[item] = true

		if model := modelOf(item); model != "" {
			named[replayItem{id: model}] = true
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

	same := func(a, b Snapshot, item replayItem) bool { return reflect.DeepEqual(a.value(item), b.value(item)) }

	var mismatches []string

	for item := range items {
		label := item.label(vendor)

		if !same(parent, bot, item) {
			if !same(parent, base, item) {
				mismatches = append(mismatches, label+": the base changed it after the bot's commit")
			}

			if !same(bot, replay, item) {
				mismatches = append(mismatches, label+": the replay doesn't make the bot's change")
			}

			continue
		}

		if named[item] && !same(parent, base, item) {
			mismatches = append(mismatches, label+": a report row names it, and the base changed it after the bot's commit")
		} else if model := modelOf(item); model != "" && named[replayItem{id: model}] && !same(parent, base, item) {
			mismatches = append(mismatches, label+": a report row names its model, and the base changed it after the bot's commit")
		}

		if !same(base, replay, item) {
			mismatches = append(mismatches, label+": the replay changes it, and the bot's commit didn't")
		}
	}

	slices.Sort(mismatches)

	return mismatches
}

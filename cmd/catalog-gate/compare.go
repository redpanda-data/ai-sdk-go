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
	"fmt"
	"maps"
	"net/url"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// rows are a report's changes and needs_human rows as "offering field", the
// ones safe to print: a baseline report comes from an earlier run. Each maps
// to its rows' values, sorted so the report's row order doesn't count, which
// are compared but never printed.
func rows(report reconcile.Report) map[string]string {
	grouped := map[string][]string{}
	add := func(offering, field string, values ...string) {
		if reconcile.OfferingPattern.MatchString(offering) && reconcile.OfferingPattern.MatchString(field) {
			key := offering + " " + field
			grouped[key] = append(grouped[key], strings.Join(values, "\x00"))
		}
	}

	for _, change := range report.Changes {
		add(change.Offering, change.Field, change.Old, change.New)
	}

	for _, item := range report.NeedsHuman {
		add(item.Offering, item.Field, append([]string{item.Catalogue, item.Page}, item.Values...)...)
	}

	found := make(map[string]string, len(grouped))
	for key, values := range grouped {
		slices.Sort(values)
		found[key] = strings.Join(values, "\x01")
	}

	return found
}

// pageKey is a URL without its query, fragment and .md, .md.txt or .html
// ending, so a page the agent cited matches the gate's copy of it.
func pageKey(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return raw
	}

	trimmed := strings.TrimSuffix(parsed.Path, "/")
	for _, ending := range []string{".md.txt", ".md", ".html"} {
		trimmed = strings.TrimSuffix(trimmed, ending)
	}

	return strings.ToLower(parsed.Hostname()) + trimmed
}

// compareRuns is the record job's summary. It names the sources the agent
// cited that the gate doesn't fingerprint, and, when every page and input
// matched the baseline (same), what this run and the baseline run found
// differently: either is a finding a skip would have lost.
func compareRuns(report, baselineReport []byte, pages fingerprints, same bool) (string, error) {
	current, err := reconcile.ParseReport(report)
	if err != nil {
		return "", err
	}

	covered := map[string]bool{}
	for _, s := range pages.Sources {
		covered[pageKey(s.URL)] = true
	}

	var uncovered []string

	note := func(source string) {
		if !covered[pageKey(source)] {
			uncovered = append(uncovered, "- `"+source+"`")
		}
	}

	for _, change := range current.Changes {
		note(change.SourceURL)
	}

	for _, item := range current.NeedsHuman {
		for _, source := range item.SourceURLs {
			note(source)
		}
	}

	slices.Sort(uncovered)
	uncovered = slices.Compact(uncovered)

	text := "### Run comparison\n\n"
	if len(uncovered) > 0 {
		text += fmt.Sprintf("**Cited sources the gate doesn't fingerprint (%d):** a skip could miss a change on these pages.\n\n%s\n\n", len(uncovered), strings.Join(uncovered, "\n"))
	} else {
		text += "Every cited source is on the gate's page list.\n\n"
	}

	if !same {
		return text + "The pages or inputs changed since the baseline run, so the two runs' findings aren't compared.\n", nil
	}

	baseline, err := reconcile.ParseReport(baselineReport)
	if err != nil {
		return text + "No report from the baseline run to compare with.\n", nil //nolint:nilerr // a missing or old baseline report only means no comparison
	}

	now, before := rows(current), rows(baseline)
	// pick lists the rows of a that keep returns true for.
	pick := func(a map[string]string, keep func(row, values string) bool) string {
		var found []string

		for _, row := range slices.Sorted(maps.Keys(a)) {
			if keep(row, a[row]) {
				found = append(found, row)
			}
		}

		if len(found) == 0 {
			return ""
		}

		return "`" + strings.Join(found, "`, `") + "`"
	}
	only := func(a, b map[string]string) string {
		return pick(a, func(row, _ string) bool { _, ok := b[row]; return !ok })
	}

	added, dropped := only(now, before), only(before, now)

	changed := pick(now, func(row, values string) bool { was, ok := before[row]; return ok && was != values })
	if added == "" && dropped == "" && changed == "" {
		return text + "**A skip would have lost nothing:** every page and input matched the baseline run, and this run found the same rows.\n", nil
	}

	text += "**A skip would have lost findings:** every page and input matched the baseline run, but the two runs found different rows or values, so shadow mode stays on.\n\n"
	if added != "" {
		text += "- Found now, not by the baseline run: " + added + "\n"
	}

	if dropped != "" {
		text += "- Found by the baseline run, not now: " + dropped + "\n"
	}

	if changed != "" {
		text += "- Found by both runs, with different values: " + changed + "\n"
	}

	return text, nil
}

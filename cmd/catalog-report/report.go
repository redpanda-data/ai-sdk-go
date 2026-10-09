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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"regexp"
	"slices"
	"strings"
)

// Report is the agent job's structured output. It deliberately has no
// free-text fields: everything rendered into a PR is an ID, an enum value,
// a short value matching valuePattern, or a URL on a named-source host.
type Report struct {
	Changes    []Change     `json:"changes"`
	NeedsHuman []NeedsHuman `json:"needs_human"`
	Skipped    []Skipped    `json:"skipped"`
}

// Change is one value the agent changed.
type Change struct {
	Offering  string `json:"offering"`
	Field     string `json:"field"`
	Old       string `json:"old"`
	New       string `json:"new"`
	SourceURL string `json:"source_url"`
}

// NeedsHuman is a value the agent couldn't decide.
type NeedsHuman struct {
	Offering   string   `json:"offering"`
	Field      string   `json:"field"`
	Values     []string `json:"values"`
	SourceURLs []string `json:"source_urls"`
}

// Skipped is an offering the agent left alone.
type Skipped struct {
	Offering string `json:"offering"`
	Reason   string `json:"reason"`
}

var (
	fields          = []string{"new_model", "price", "limit", "lifecycle", "replacement", "knowledge_cutoff", "release_date", "capability"}
	reasons         = []string{"not_on_named_sources", "conflicting_sources", "needs_new_pricing_shape", "out_of_scope"}
	offeringPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._:-]{0,79}$`)
	valuePattern    = regexp.MustCompile(`^[A-Za-z0-9 ._:/+-]{0,64}$`)
	// sourcePattern allows only characters that can't end a markdown link,
	// end a table cell or start HTML, so a URL always renders as one link.
	sourcePattern = regexp.MustCompile(`^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._~/%#?=&+:-]*)?$`)
)

// Parse decodes a report strictly: unknown fields are an error, and all three
// arrays must be present, so a missing or null report never reads as "no drift".
func Parse(data []byte) (Report, error) {
	// Pointers tell a missing or null array apart from an empty one.
	var raw struct {
		Changes    *[]Change     `json:"changes"`
		NeedsHuman *[]NeedsHuman `json:"needs_human"`
		Skipped    *[]Skipped    `json:"skipped"`
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&raw); err != nil {
		return Report{}, fmt.Errorf("parse report: %w", err)
	}

	if raw.Changes == nil || raw.NeedsHuman == nil || raw.Skipped == nil {
		return Report{}, errors.New("parse report: changes, needs_human and skipped are required")
	}

	return Report{Changes: *raw.Changes, NeedsHuman: *raw.NeedsHuman, Skipped: *raw.Skipped}, nil
}

// Validate lists every problem that would make the report unsafe to render.
func Validate(report Report, hosts []string) []string {
	var problems []string
	checkOffering := func(where, offering string) {
		if !offeringPattern.MatchString(offering) {
			problems = append(problems, where+": bad offering "+fmt.Sprintf("%q", offering))
		}
	}
	checkValue := func(where, value string) {
		if !valuePattern.MatchString(value) {
			problems = append(problems, where+": bad value "+fmt.Sprintf("%q", value))
		}
	}

	checkSource := func(where, raw string) {
		parsed, err := url.Parse(raw)
		if err != nil || !sourcePattern.MatchString(raw) || parsed.Scheme != "https" || !slices.Contains(hosts, parsed.Hostname()) {
			problems = append(problems, where+": source not on a named-source host: "+fmt.Sprintf("%q", raw))
		}
	}
	for index, change := range report.Changes {
		where := fmt.Sprintf("changes[%d]", index)
		checkOffering(where, change.Offering)

		if !slices.Contains(fields, change.Field) {
			problems = append(problems, where+": unknown field "+fmt.Sprintf("%q", change.Field))
		}

		checkValue(where, change.Old)
		checkValue(where, change.New)
		checkSource(where, change.SourceURL)
	}

	for index, item := range report.NeedsHuman {
		where := fmt.Sprintf("needs_human[%d]", index)
		checkOffering(where, item.Offering)

		if !slices.Contains(fields, item.Field) {
			problems = append(problems, where+": unknown field "+fmt.Sprintf("%q", item.Field))
		}

		for _, value := range item.Values {
			checkValue(where, value)
		}

		if len(item.SourceURLs) == 0 {
			problems = append(problems, where+": needs at least one source")
		}

		for _, source := range item.SourceURLs {
			checkSource(where, source)
		}
	}

	for index, item := range report.Skipped {
		where := fmt.Sprintf("skipped[%d]", index)
		checkOffering(where, item.Offering)

		if !slices.Contains(reasons, item.Reason) {
			problems = append(problems, where+": unknown reason "+fmt.Sprintf("%q", item.Reason))
		}
	}

	return problems
}

func code(value string) string { return "`" + value + "`" }

func link(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return code(raw)
	}

	return "[" + parsed.Hostname() + "](" + raw + ")"
}

// Render writes the PR body. Call it only on a report Validate accepted.
func Render(report Report, provider, snapshotDiff string) string {
	var body strings.Builder
	body.WriteString("## Catalogue drift: " + provider + "\n\n")

	if len(report.Changes)+len(report.NeedsHuman)+len(report.Skipped) == 0 && snapshotDiff == "" {
		body.WriteString("No drift found.\n")
		return body.String()
	}

	body.WriteString("Proposed by the reconcile-models workflow. The tables come from the agent's structured report; the PR job checked its format and that every source is a named source. Check each value against its source before approving.\n\nDon't push commits to this branch: each run rebuilds it from main and force-pushes, so they would be lost. Comment on the PR instead.\n\n")

	if len(report.Changes) > 0 {
		fmt.Fprintf(&body, "### Changes (%d)\n\n| Offering | Field | Old | New | Source |\n|---|---|---|---|---|\n", len(report.Changes))

		for _, change := range report.Changes {
			fmt.Fprintf(&body, "| %s | %s | %s | %s | %s |\n", code(change.Offering), change.Field, code(change.Old), code(change.New), link(change.SourceURL))
		}

		body.WriteString("\n")
	}

	if len(report.NeedsHuman) > 0 {
		fmt.Fprintf(&body, "### Needs a human (%d)\n\n| Offering | Field | Values | Sources |\n|---|---|---|---|\n", len(report.NeedsHuman))

		for _, item := range report.NeedsHuman {
			values := make([]string, 0, len(item.Values))
			for _, value := range item.Values {
				values = append(values, code(value))
			}

			sources := make([]string, 0, len(item.SourceURLs))
			for _, source := range item.SourceURLs {
				sources = append(sources, link(source))
			}

			fmt.Fprintf(&body, "| %s | %s | %s | %s |\n", code(item.Offering), item.Field, strings.Join(values, ", "), strings.Join(sources, ", "))
		}

		body.WriteString("\n")
	}

	if len(report.Skipped) > 0 {
		fmt.Fprintf(&body, "### Skipped (%d)\n\n| Offering | Reason |\n|---|---|\n", len(report.Skipped))

		for _, item := range report.Skipped {
			fmt.Fprintf(&body, "| %s | %s |\n", code(item.Offering), item.Reason)
		}

		body.WriteString("\n")
	}

	if snapshotDiff != "" {
		body.WriteString("<details><summary>Snapshot diff</summary>\n\n```diff\n" + snapshotDiff + "\n```\n</details>\n")
	}

	return body.String()
}

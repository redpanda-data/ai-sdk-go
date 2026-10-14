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

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
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

// NeedsHuman is a value the agent couldn't decide. Rule is the RULES.md rule
// that sent it to a person, or R4 when no rule covers the case.
type NeedsHuman struct {
	Offering   string   `json:"offering"`
	Field      string   `json:"field"`
	Values     []string `json:"values"`
	SourceURLs []string `json:"source_urls"`
	Rule       string   `json:"rule"`
}

// Skipped is an offering the agent left alone, and the rule that left it out.
type Skipped struct {
	Offering string `json:"offering"`
	Reason   string `json:"reason"`
	Rule     string `json:"rule"`
}

var (
	fields          = []string{"new_model", "price", "limit", "lifecycle", "replacement", "knowledge_cutoff", "release_date", "capability"}
	reasons         = []string{"not_on_named_sources", "conflicting_sources", "needs_new_pricing_shape", "out_of_scope", "limited_availability", "vendor_excluded"}
	offeringPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._:-]{0,79}$`)
	valuePattern    = regexp.MustCompile(`^[A-Za-z0-9 ._:/+-]{0,64}$`)
	// sourcePattern allows only characters that can't end a markdown link,
	// end a table cell or start HTML, so a URL always renders as one link.
	sourcePattern = regexp.MustCompile(`^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._~/%#?=&+:-]*)?$`)
	// codeFilePattern keeps listed file paths to plain path characters, so a
	// path can't break out of its code span in the PR body.
	codeFilePattern = regexp.MustCompile(`^[A-Za-z0-9_./-]{1,200}$`)
)

// ValidateCodeFiles lists every path that would be unsafe to render.
func ValidateCodeFiles(files []string) []string {
	var problems []string

	for _, file := range files {
		if !codeFilePattern.MatchString(file) {
			problems = append(problems, "bad code file path "+fmt.Sprintf("%q", file))
		}
	}

	return problems
}

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

// Validate lists every problem that would make the report unsafe to render,
// including a row that names a rule RULES.md doesn't have.
func Validate(report Report, hosts []string, rules reconcile.Rules) []string {
	var problems []string
	// A row's rule must exist and produce the row's outcome (RULES.md R4).
	checkRule := func(where, id, outcome string) {
		if _, ok := rules.Lines[id]; !ok {
			problems = append(problems, where+": unknown rule "+fmt.Sprintf("%q", id))
		} else if !rules.Fits(id, outcome) {
			problems = append(problems, where+": rule "+id+" doesn't produce "+outcome)
		}
	}
	checkOffering := func(where, offering string) {
		if !offeringPattern.MatchString(offering) {
			problems = append(problems, where+": bad offering "+fmt.Sprintf("%q", offering))
		}
	}
	checkValue := func(where, value string) {
		// "//" keeps a bare URL out of the tables: honest values (prices,
		// dates, IDs, "us/eu …") never contain it.
		if !valuePattern.MatchString(value) || strings.Contains(value, "//") {
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

		checkRule(where, item.Rule, "needs_human")
	}

	for index, item := range report.Skipped {
		where := fmt.Sprintf("skipped[%d]", index)
		checkOffering(where, item.Offering)

		if !slices.Contains(reasons, item.Reason) {
			problems = append(problems, where+": unknown reason "+fmt.Sprintf("%q", item.Reason))
		}

		checkRule(where, item.Rule, "skip:"+code(item.Reason))
	}

	return problems
}

func code(value string) string { return "`" + value + "`" }

// value renders a report value as code, or "none" when it's empty (e.g. the
// old value of a new model), since an empty code span shows as stray backticks.
func value(v string) string {
	if v == "" {
		return "none"
	}

	return code(v)
}

func link(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil {
		return code(raw)
	}

	return "[" + parsed.Hostname() + "](" + raw + ")"
}

// valueList renders report values as comma-separated code spans.
func valueList(values []string) string {
	rendered := make([]string, 0, len(values))
	for _, v := range values {
		rendered = append(rendered, value(v))
	}

	return strings.Join(rendered, ", ")
}

// Base is what Render reads from the base commit: a link prefix for its
// files (empty for no links), its rules and decisions, and today as YYYY-MM-DD.
type Base struct {
	URL       string
	Rules     reconcile.Rules
	Decisions []reconcile.Decision
	Today     string
}

// fileLink links text to a repo file at the base commit, at line when it's
// above 0. Without a base URL it's a code span. GitHub renders markdown
// files, so a line link to one needs ?plain=1.
func (base Base) fileLink(text, path string, line int) string {
	if base.URL == "" {
		return code(text)
	}

	target := base.URL + "/" + path
	if line > 0 {
		if strings.HasSuffix(path, ".md") {
			target += "?plain=1"
		}

		target += fmt.Sprintf("#L%d", line)
	}

	return "[" + text + "](" + target + ")"
}

// ruleLink links a rule ID to its row in RULES.md.
func (base Base) ruleLink(id string) string {
	return base.fileLink(id, reconcile.RulesPath, base.Rules.Lines[id])
}

// ruleCell names a row's rule, flagged when no rule covers the case.
func (base Base) ruleCell(id string) string {
	if id == reconcile.JudgmentRule {
		return "⚠ " + base.ruleLink(id)
	}

	return base.ruleLink(id)
}

// skippedRows groups skipped offerings by the rule and reason that left them
// out, in the order the report first names each group.
func (base Base) skippedRows(skipped []Skipped) []string {
	type group struct{ rule, reason string }

	var groups []group

	offerings := map[group][]string{}

	for _, item := range skipped {
		key := group{item.Rule, item.Reason}
		if _, seen := offerings[key]; !seen {
			groups = append(groups, key)
		}

		offerings[key] = append(offerings[key], item.Offering)
	}

	rows := make([]string, 0, len(groups))

	for _, key := range groups {
		why := code(key.reason)
		if meaning := base.Rules.Reasons[key.reason]; meaning != "" {
			why = meaning + " (" + why + ")"
		}

		rows = append(rows, fmt.Sprintf("| %s | %s | %s |\n", base.ruleCell(key.rule), why, valueList(offerings[key])))
	}

	return rows
}

// needsHumanBlock renders one needs_human row: its rule, the rule's fixed
// question, what the bot found, and the sources.
func (base Base) needsHumanBlock(number int, item NeedsHuman) string {
	why := "rule " + base.ruleLink(item.Rule)
	if item.Rule == reconcile.JudgmentRule {
		why = "⚠ no rule covers this (" + base.ruleLink(item.Rule) + ")"
	}

	sources := make([]string, 0, len(item.SourceURLs))
	for _, source := range item.SourceURLs {
		sources = append(sources, link(source))
	}

	block := fmt.Sprintf("**%d. %s · %s · %s**\n", number, code(item.Offering), item.Field, why)
	if question := base.Rules.Questions[item.Rule]; question != "" {
		block += "- **Question:** " + question + "\n"
	}

	if len(item.Values) > 0 {
		block += "- **Found:** " + valueList(item.Values) + "\n"
	}

	return block + "- **Sources:** " + strings.Join(sources, ", ") + "\n\n"
}

// sameOffering reports whether a report offering names a decision's. Report
// offerings in the run's own package have no prefix (RULES.md R2); Bedrock
// IDs contain ":", so the prefix can't be told apart by looking for one.
func sameOffering(reported, provider, decided string) bool {
	return decided == reported || decided == provider+":"+reported
}

// hold takes the needs_human and skipped rows a decision already answers out
// of the report, and returns the rest with the held table's rows.
func (base Base) hold(report Report, provider string) (Report, []string) {
	find := func(offering, field string) (reconcile.Decision, bool) {
		for _, d := range base.Decisions {
			if d.Field == field && sameOffering(offering, provider, d.Offering) {
				return d, true
			}
		}

		return reconcile.Decision{}, false
	}
	row := func(d reconcile.Decision, offering, saw string) string {
		return fmt.Sprintf("| %s | %s | %s | %s | %s |\n", base.fileLink(d.ID, reconcile.DecisionsPath, d.Line),
			code(offering), d.Field, saw, strings.ReplaceAll(d.Reason, "|", `\|`))
	}

	var held []string

	// Fresh slices: RenderCapped renders the same report more than once.
	var needs []NeedsHuman

	for _, item := range report.NeedsHuman {
		if d, ok := find(item.Offering, item.Field); ok {
			held = append(held, row(d, item.Offering, valueList(item.Values)))
		} else {
			needs = append(needs, item)
		}
	}

	var skipped []Skipped

	for _, item := range report.Skipped {
		if d, ok := find(item.Offering, reconcile.FieldNewModel); ok {
			held = append(held, row(d, item.Offering, item.Reason))
		} else {
			skipped = append(skipped, item)
		}
	}

	report.NeedsHuman, report.Skipped = needs, skipped

	return report, held
}

// summary is one line about decisions.yaml itself: how many decisions it
// holds, and which are past their review date. It shows on every run.
func (base Base) summary() string {
	if len(base.Decisions) == 0 {
		return ""
	}

	var overdue []string

	for _, d := range base.Decisions {
		if d.ReviewBy < base.Today {
			overdue = append(overdue, d.ID)
		}
	}

	due := "none."
	if len(overdue) > 0 {
		due = strings.Join(overdue, ", ") + " (still enforced)."
	}

	return fmt.Sprintf("Recorded decisions: %d, in %s. Past their review date: %s\n\n",
		len(base.Decisions), base.fileLink("decisions.yaml", reconcile.DecisionsPath, 0), due)
}

// Render writes the PR body. Call it only on a report Validate accepted and
// code files ValidateCodeFiles accepted. codeFiles are the files whose change
// isn't data only, listed first so a reviewer reads them as code.
func Render(report Report, provider, snapshotDiff string, codeFiles []string, base Base) string {
	var body strings.Builder
	body.WriteString("## Catalogue drift: " + provider + "\n\n")

	report, held := base.hold(report, provider)

	if len(report.Changes)+len(report.NeedsHuman)+len(report.Skipped)+len(held) == 0 && snapshotDiff == "" {
		body.WriteString("No drift found.\n\n" + base.summary())
		return body.String()
	}

	body.WriteString("Proposed by the reconcile-models workflow. The tables come from the agent's structured report; the PR job checked its format and that every source is a named source. Check each value against its source before approving.\n\nDon't push commits to this branch: each run rebuilds it from main and force-pushes, so they would be lost. To settle a row, merge a fix on main or record a decision in `.github/reconcile-models/decisions.yaml`. Later runs don't read PR comments.\n\n")
	body.WriteString(base.summary())

	if len(codeFiles) > 0 {
		fmt.Fprintf(&body, "### Code changed (%d)\n\nThese files change code, not only data. Review them as code before marking the PR ready: marking it ready runs them with secrets.\n\n", len(codeFiles))

		for _, file := range codeFiles {
			body.WriteString("- " + code(file) + "\n")
		}

		body.WriteString("\n")
	}

	if len(report.Changes) > 0 {
		fmt.Fprintf(&body, "### Changes (%d)\n\n| Offering | Field | Old | New | Source |\n|---|---|---|---|---|\n", len(report.Changes))

		for _, change := range report.Changes {
			fmt.Fprintf(&body, "| %s | %s | %s | %s | %s |\n", code(change.Offering), change.Field, value(change.Old), value(change.New), link(change.SourceURL))
		}

		body.WriteString("\n")
	}

	if len(report.NeedsHuman) > 0 {
		judged := 0

		for _, item := range report.NeedsHuman {
			if item.Rule == reconcile.JudgmentRule {
				judged++
			}
		}

		fmt.Fprintf(&body, "### Needs a human (%d): %d by a rule, %d with no rule\n\n", len(report.NeedsHuman), len(report.NeedsHuman)-judged, judged)

		for index, item := range report.NeedsHuman {
			body.WriteString(base.needsHumanBlock(index+1, item))
		}
	}

	if len(held) > 0 {
		fmt.Fprintf(&body, "### Held by decisions (%d)\n\nQuestions a recorded decision already answers. Nothing here changes.\n\n| Decision | Offering | Field | What the bot saw | Why it's settled |\n|---|---|---|---|---|\n", len(held))
		body.WriteString(strings.Join(held, "") + "\n")
	}

	if len(report.Skipped) > 0 {
		fmt.Fprintf(&body, "### Skipped (%d)\n\n| Rule | Why | Offerings |\n|---|---|---|\n", len(report.Skipped))
		body.WriteString(strings.Join(base.skippedRows(report.Skipped), "") + "\n")
	}

	if snapshotDiff != "" {
		body.WriteString("<details><summary>Snapshot diff</summary>\n\n```diff\n" + snapshotDiff + "\n```\n</details>\n")
	}

	return body.String()
}

// truncatedNote ends a snapshot diff that RenderCapped shortened.
const truncatedNote = "\n… (truncated; see the workflow artifact)"

// RenderCapped is Render, kept within limit bytes so GitHub accepts the PR
// body. Only the snapshot diff is shortened, and only between lines.
func RenderCapped(report Report, provider, snapshotDiff string, codeFiles []string, base Base, limit int) string {
	body := Render(report, provider, snapshotDiff, codeFiles, base)
	if len(body) <= limit {
		return body
	}

	room := limit - len(Render(report, provider, truncatedNote, codeFiles, base))
	diff := ""

	if room > 0 {
		diff = snapshotDiff[:min(room, len(snapshotDiff))]
		diff = diff[:max(strings.LastIndexByte(diff, '\n'), 0)]
	}

	return Render(report, provider, diff+truncatedNote, codeFiles, base)
}

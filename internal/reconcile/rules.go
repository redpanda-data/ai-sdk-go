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
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// RulesPath is RULES.md in the repo: the rules a reconcile run follows.
const RulesPath = ".github/reconcile-models/RULES.md"

// JudgmentRule is the Ask a report row names when no other rule covers its
// case, so the row rests on the agent's own judgment.
const JudgmentRule = "no-rule"

// Rules is what RULES.md defines, keyed for the report and its PR body.
type Rules struct {
	Lines      map[string]int      // rule ID: its line, for links
	Outcomes   map[string][]string // rule ID: its outcomes, one per item, notes dropped
	RuleFields map[string][]string // Default ID: the report fields its change rows are about
	Reasons    map[string]string   // skip reason: what it means
	Fields     map[string]string   // report field: what it covers
	Questions  map[string]string   // Ask ID: what a person decides for its needs_human rows
	Offered    map[string][]string // host package: the region prefixes it offers
	// NotOffered is, per host package, each region prefix asked about and
	// its note: "asked YYYY-MM-DD" or "declined YYYY-MM-DD".
	NotOffered map[string]map[string]string
}

// Fits reports whether a rule can produce an outcome ("needs_human" or
// "skip:`<reason>`"). JudgmentRule fits any row no other rule covers.
func (rules Rules) Fits(id, outcome string) bool {
	return id == JudgmentRule || slices.Contains(rules.Outcomes[id], outcome)
}

var (
	// RuleIDPattern is a rule ID: a slug that names the rule's behaviour
	// (region-not-offered). report.schema.json uses the same pattern.
	RuleIDPattern = regexp.MustCompile(`^[a-z]+(-[a-z0-9]+)+$`)
	parenthetical = regexp.MustCompile(`\([^)]*\)`)
	// A legend word is an item's first code term, cut at a placeholder:
	// "`skip:<reason>`" gives "skip:".
	legendTerm = regexp.MustCompile("`([^`<]+)")
	codeTerm   = regexp.MustCompile("`([^`]+)`")
	// A Not offered item is a region prefix in code and the text after it,
	// which must be its note: "`in` (asked 2026-10-15)".
	notOfferedItem = regexp.MustCompile("`([^`]+)`([^`]*)")
	regionNote     = regexp.MustCompile(`^\(((?:asked|declined) \d{4}-\d{2}-\d{2})\)$`)
)

// cells splits one markdown table row into trimmed cells.
func cells(line string) []string {
	parts := strings.Split(strings.Trim(strings.TrimSpace(line), "|"), "|")
	for i, part := range parts {
		parts[i] = strings.TrimSpace(part)
	}

	return parts
}

// legendWords reads the words a legend line defines, one per "·" item.
func legendWords(line string) []string {
	var words []string

	for item := range strings.SplitSeq(line, " · ") {
		if match := legendTerm.FindStringSubmatch(item); match != nil {
			words = append(words, match[1])
		}
	}

	return words
}

// ParseRules reads RULES.md and lists every way it is malformed. A table's
// kind is its header's first cell: Defaults ("Default") and Always asks
// ("Ask") hold the rules. The Outcome legend and the Skip reason and Field
// tables hold the values Defaults use, so they come before them in the file.
// Offered regions ("Host") holds each host's offered and not offered regions.
func ParseRules(markdown []byte) (Rules, []string) {
	rules := Rules{
		Lines: map[string]int{}, Outcomes: map[string][]string{}, RuleFields: map[string][]string{},
		Reasons: map[string]string{}, Fields: map[string]string{}, Questions: map[string]string{},
		Offered: map[string][]string{}, NotOffered: map[string]map[string]string{},
	}
	twoColumn := map[string]map[string]string{"Skip reason": rules.Reasons, "Field": rules.Fields}

	var (
		legend, problems []string
		header           string
	)

	for index, line := range strings.Split(string(markdown), "\n") {
		if strings.HasPrefix(line, "- **Outcome.**") {
			legend = legendWords(line)
		}

		if !strings.HasPrefix(strings.TrimSpace(line), "|") {
			header = ""
			continue
		}

		row := cells(line)

		switch {
		case header == "":
			header = row[0]
		case strings.HasPrefix(row[0], "---"):
		case header == "Default":
			problems = append(problems, rules.addDefault(row, index+1, legend)...)
		case header == "Ask":
			problems = append(problems, rules.addAsk(row, index+1)...)
		case header == "Host" && len(row) > 1:
			for _, term := range codeTerm.FindAllStringSubmatch(row[1], -1) {
				rules.Offered[row[0]] = append(rules.Offered[row[0]], term[1])
			}

			if len(row) > 2 {
				problems = append(problems, rules.addNotOffered(row[0], row[2])...)
			}
		case twoColumn[header] != nil && len(row) > 1:
			twoColumn[header][strings.Trim(row[0], "`")] = row[1]
		}
	}

	return rules, problems
}

// addNotOffered records a host's Not offered cell: each region prefix in code,
// followed by "(asked YYYY-MM-DD)" while the question waits or
// "(declined YYYY-MM-DD)" once the answer is no.
func (rules Rules) addNotOffered(host, cell string) []string {
	var problems []string

	for _, item := range notOfferedItem.FindAllStringSubmatch(cell, -1) {
		prefix, text := item[1], strings.TrimSpace(item[2])

		note := regionNote.FindStringSubmatch(text)
		if note == nil {
			problems = append(problems, fmt.Sprintf("%s: Not offered %s: want (asked YYYY-MM-DD) or (declined YYYY-MM-DD), got %q", host, prefix, text))
			continue
		}

		if rules.NotOffered[host] == nil {
			rules.NotOffered[host] = map[string]string{}
		}

		rules.NotOffered[host][prefix] = note[1]
	}

	return problems
}

// addRule records a rule's ID (its first cell) and line, and checks its
// cells are filled.
func (rules Rules) addRule(row []string, line int) []string {
	var problems []string

	id := row[0]

	if !RuleIDPattern.MatchString(id) {
		problems = append(problems, "bad rule ID "+id)
	}

	if _, ok := rules.Lines[id]; ok {
		problems = append(problems, "duplicate rule ID "+id)
	}

	rules.Lines[id] = line

	if slices.Contains(row, "") {
		problems = append(problems, id+": empty cell")
	}

	return problems
}

// addDefault records a Defaults row: Default | Field | When | Bot does | Why |
// Outcome. A Default never sends a row to a person; that case is an Ask.
func (rules Rules) addDefault(row []string, line int, legend []string) []string {
	if len(row) != 6 {
		return []string{row[0] + ": a Default row has 6 cells"}
	}

	id := row[0]
	problems := rules.addRule(row, line)

	for part := range strings.SplitSeq(parenthetical.ReplaceAllString(row[5], ""), ",") {
		outcome := strings.TrimSpace(part)
		rules.Outcomes[id] = append(rules.Outcomes[id], outcome)

		switch reason, skip := strings.CutPrefix(outcome, "skip:"); {
		case outcome == "needs_human":
			problems = append(problems, id+": a Default can't send a row to a person; make it an Ask")
		case !slices.ContainsFunc(legend, func(word string) bool { return strings.HasPrefix(outcome, word) }):
			problems = append(problems, id+": unknown outcome "+outcome)
		case skip:
			if _, known := rules.Reasons[strings.Trim(reason, "`")]; !known {
				problems = append(problems, id+": unknown skip reason "+reason)
			}
		}
	}

	for _, term := range codeTerm.FindAllStringSubmatch(row[1], -1) {
		if _, known := rules.Fields[term[1]]; !known {
			problems = append(problems, id+": unknown field "+term[1])
		}

		rules.RuleFields[id] = append(rules.RuleFields[id], term[1])
	}

	if slices.Contains(rules.Outcomes[id], "change") && len(rules.RuleFields[id]) == 0 {
		problems = append(problems, id+": a rule that makes a change names its field")
	}

	return problems
}

// addAsk records an Always asks row: Ask | When | Question | Why. Its rows go
// to a person with its fixed question.
func (rules Rules) addAsk(row []string, line int) []string {
	if len(row) != 4 {
		return []string{row[0] + ": an Ask row has 4 cells"}
	}

	id := row[0]
	rules.Outcomes[id] = []string{"needs_human"}
	rules.Questions[id] = row[2]

	return rules.addRule(row, line)
}

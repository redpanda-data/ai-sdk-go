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
	"regexp"
	"slices"
	"strings"
)

// RulesPath is RULES.md in the repo: the rules a reconcile run follows.
const RulesPath = ".github/reconcile-models/RULES.md"

// JudgmentRule is the rule a report row names when no other rule covers its
// case (RULES.md R4), so the row rests on the agent's own judgment.
const JudgmentRule = "R4"

// Rules is what RULES.md defines, keyed for the report and its PR body.
type Rules struct {
	Lines     map[string]int      // rule ID: its line, for links
	Produces  map[string][]string // rule ID: its Produces cell, one outcome per item, notes dropped
	Reasons   map[string]string   // skip reason: what it means
	Fields    map[string]string   // report field: what it covers
	Questions map[string]string   // rule ID: what a person decides for its needs_human rows
}

// Fits reports whether a rule can produce an outcome ("needs_human" or
// "skip:`<reason>`"). JudgmentRule fits any row no other rule covers.
func (rules Rules) Fits(id, outcome string) bool {
	return id == JudgmentRule || slices.Contains(rules.Produces[id], outcome)
}

var (
	ruleIDPattern = regexp.MustCompile(`^[A-Z][0-9]{1,2}$`)
	parenthetical = regexp.MustCompile(`\([^)]*\)`)
	// A legend word is an item's first code term, cut at a placeholder:
	// "`skip:<reason>`" gives "skip:".
	legendTerm = regexp.MustCompile("`([^`<]+)")
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

// ParseRules reads RULES.md and lists every way it is malformed: a rule row
// with a bad shape or a word its legend doesn't define, or a question that
// doesn't match exactly the rules that send rows to a person. A table's kind
// is its header's first cell.
func ParseRules(markdown []byte) (Rules, []string) {
	rules := Rules{
		Lines: map[string]int{}, Produces: map[string][]string{},
		Reasons: map[string]string{}, Fields: map[string]string{}, Questions: map[string]string{},
	}
	twoColumn := map[string]map[string]string{"Skip reason": rules.Reasons, "Field": rules.Fields, "Rule": rules.Questions}

	var (
		produces, enforced, problems []string
		header                       string
	)

	for index, line := range strings.Split(string(markdown), "\n") {
		switch {
		case strings.HasPrefix(line, "- **Produces.**"):
			produces = legendWords(line)
		case strings.HasPrefix(line, "- **Enforced.**"):
			enforced = legendWords(line)
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
		case header == "ID":
			problems = append(problems, rules.addRule(row, index+1, produces, enforced)...)
		case twoColumn[header] != nil && len(row) > 1:
			twoColumn[header][strings.Trim(row[0], "`")] = row[1]
		}
	}

	for id := range rules.Lines {
		asks := slices.Contains(rules.Produces[id], "needs_human")

		switch question := rules.Questions[id]; {
		case asks && question == "":
			problems = append(problems, id+" can send a row to a person but has no question")
		case !asks && question != "":
			problems = append(problems, id+" has a question but never sends a row to a person")
		}
	}

	for id := range rules.Questions {
		if _, ok := rules.Lines[id]; !ok {
			problems = append(problems, "a question for unknown rule "+id)
		}
	}

	return rules, problems
}

// addRule records one rule row and returns what's wrong with it.
func (rules Rules) addRule(row []string, line int, produces, enforced []string) []string {
	if len(row) != 5 {
		return []string{"rule row without 5 cells: " + strings.Join(row, " | ")}
	}

	var problems []string

	id := row[0]
	if !ruleIDPattern.MatchString(id) {
		problems = append(problems, "bad rule ID "+id)
	}

	if _, ok := rules.Lines[id]; ok {
		problems = append(problems, "duplicate rule ID "+id)
	}

	rules.Lines[id] = line

	if row[1] == "" || row[2] == "" {
		problems = append(problems, id+": empty rule or why")
	}

	for part := range strings.SplitSeq(parenthetical.ReplaceAllString(row[3], ""), ",") {
		rules.Produces[id] = append(rules.Produces[id], strings.TrimSpace(part))
	}

	return slices.Concat(problems, rules.checkWords(id, "Produces", row[3], produces), rules.checkWords(id, "Enforced", row[4], enforced))
}

// checkWords reports each comma-separated part of a cell that doesn't start
// with a legend word, and each skip reason the file doesn't list. Notes in
// brackets may hold commas, so they go first.
func (rules Rules) checkWords(id, column, cell string, words []string) []string {
	var problems []string

	for part := range strings.SplitSeq(parenthetical.ReplaceAllString(cell, ""), ",") {
		part = strings.TrimSpace(part)
		if !slices.ContainsFunc(words, func(word string) bool { return strings.HasPrefix(part, word) }) {
			problems = append(problems, id+": unknown "+column+" "+part)
		}

		if reason, ok := strings.CutPrefix(part, "skip:"); ok {
			if _, known := rules.Reasons[strings.Trim(reason, "`")]; !known {
				problems = append(problems, id+": unknown skip reason "+reason)
			}
		}
	}

	return problems
}

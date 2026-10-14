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
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	ruleIDPattern = regexp.MustCompile(`^[A-Z][0-9]{1,2}$`)
	parenthetical = regexp.MustCompile(`\([^)]*\)`)
	// A legend word is an item's first code term, cut at a placeholder:
	// "`skip:<reason>`" gives "skip:".
	legendTerm = regexp.MustCompile("`([^`<]+)")
)

// rulesFile is what TestRulesFile reads out of RULES.md.
type rulesFile struct {
	produces []string
	enforced []string
	ids      []string
	reasons  []string
	fields   []string
	problems []string
}

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

// readRules reads the legend, checks the rule tables, and collects the skip
// reasons and fields RULES.md lists. A table's kind is its header's first cell.
func readRules(markdown string) rulesFile {
	var file rulesFile

	header := ""

	for line := range strings.SplitSeq(markdown, "\n") {
		switch {
		case strings.HasPrefix(line, "- **Produces.**"):
			file.produces = legendWords(line)
		case strings.HasPrefix(line, "- **Enforced.**"):
			file.enforced = legendWords(line)
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
			file.checkRule(row)
		case header == "Skip reason":
			file.reasons = append(file.reasons, strings.Trim(row[0], "`"))
		case header == "Field":
			file.fields = append(file.fields, strings.Trim(row[0], "`"))
		}
	}

	return file
}

func (file *rulesFile) checkRule(row []string) {
	if len(row) != 5 {
		file.problems = append(file.problems, "rule row without 5 cells: "+strings.Join(row, " | "))
		return
	}

	id := row[0]
	if !ruleIDPattern.MatchString(id) {
		file.problems = append(file.problems, "bad rule ID "+id)
	}

	if slices.Contains(file.ids, id) {
		file.problems = append(file.problems, "duplicate rule ID "+id)
	}

	file.ids = append(file.ids, id)

	if row[1] == "" || row[2] == "" {
		file.problems = append(file.problems, id+": empty rule or why")
	}

	file.checkWords(id, "Produces", row[3], file.produces)
	file.checkWords(id, "Enforced", row[4], file.enforced)
}

// checkWords reports each comma-separated part of a cell that doesn't start
// with a legend word, and each skip reason the file doesn't list. Notes in
// brackets may hold commas, so they go first.
func (file *rulesFile) checkWords(id, column, cell string, words []string) {
	for part := range strings.SplitSeq(parenthetical.ReplaceAllString(cell, ""), ",") {
		part = strings.TrimSpace(part)
		if !slices.ContainsFunc(words, func(word string) bool { return strings.HasPrefix(part, word) }) {
			file.problems = append(file.problems, id+": unknown "+column+" "+part)
		}

		if reason, ok := strings.CutPrefix(part, "skip:"); ok && !slices.Contains(file.reasons, strings.Trim(reason, "`")) {
			file.problems = append(file.problems, id+": unknown skip reason "+reason)
		}
	}
}

func TestReadRules(t *testing.T) {
	t.Parallel()

	const preamble = "- **Produces.** `change` (a hunk) · `skip:<reason>` (a row)\n" +
		"- **Enforced.** `prompt` (told) · `PR job` · `none`. A `*` marks a gap.\n\n" +
		"| Skip reason | Meaning |\n|---|---|\n| `out_of_scope` | not tracked |\n\n" +
		"| ID | Rule | Why | Produces | Enforced |\n|---|---|---|---|---|\n"

	tests := []struct {
		name        string
		rows        string
		wantProblem string
	}{
		{"valid rows", "| T1 | A rule. | A reason. | skip:`out_of_scope` | prompt* (note, with a comma) |\n| T2 | A rule. | A reason. | change | prompt, PR job |\n", ""},
		{"duplicate ID", "| T1 | A. | B. | change | prompt |\n| T1 | C. | D. | change | prompt |\n", "duplicate rule ID T1"},
		{"short row", "| T1 | A. | B. | change |\n", "without 5 cells"},
		{"bad ID", "| t1 | A. | B. | change | prompt |\n", "bad rule ID"},
		{"empty why", "| T1 | A. |  | change | prompt |\n", "empty rule or why"},
		{"unknown produces", "| T1 | A. | B. | change, maybe | prompt |\n", "unknown Produces maybe"},
		{"unknown skip reason", "| T1 | A. | B. | skip:`typo` | prompt |\n", "unknown skip reason"},
		{"unknown enforced", "| T1 | A. | B. | change | prompt, hope |\n", "unknown Enforced hope"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			problems := readRules(preamble + tt.rows).problems
			if tt.wantProblem == "" {
				assert.Empty(t, problems)
				return
			}

			assert.Contains(t, strings.Join(problems, "\n"), tt.wantProblem)
		})
	}
}

func TestRulesFile(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "RULES.md"))
	require.NoError(t, err)

	file := readRules(string(raw))
	assert.Empty(t, file.problems)
	assert.NotEmpty(t, file.ids, "RULES.md lists no rules")

	// RULES.md and the validator must list the same values; the schema is
	// held to the validator by TestSchemaMatchesValidator.
	assert.ElementsMatch(t, reasons, file.reasons, "skip reasons in RULES.md vs report.go")
	assert.ElementsMatch(t, fields, file.fields, "fields in RULES.md vs report.go")
}

func TestRulesReachTheAgent(t *testing.T) {
	t.Parallel()

	workflow, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)
	assert.Contains(t, string(workflow), "--append-system-prompt-file .github/reconcile-models/RULES.md",
		"the workflow must hand RULES.md to the agent")

	prompt, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "prompt.md"))
	require.NoError(t, err)

	// The rules live in RULES.md alone: a skip reason or field named in the
	// prompt means a rule was copied there.
	for _, word := range slices.Concat(reasons, fields) {
		assert.NotContains(t, string(prompt), "`"+word+"`", "prompt.md restates a rule")
	}
}

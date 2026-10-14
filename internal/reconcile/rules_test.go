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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestParseRules(t *testing.T) {
	t.Parallel()

	const preamble = "- **Produces.** `change` (a hunk) · `skip:<reason>` (a row) · `needs_human` (asks)\n" +
		"- **Enforced.** `prompt` (told) · `PR job` · `none`. A `*` marks a gap.\n\n" +
		"| Skip reason | Meaning |\n|---|---|\n| `out_of_scope` | not tracked |\n\n" +
		"| Field | Covers |\n|---|---|\n| `price` | rates |\n\n" +
		"| ID | Rule | Why | Produces | Enforced |\n|---|---|---|---|---|\n"

	t.Run("reads lines, meanings, fields and questions", func(t *testing.T) {
		t.Parallel()

		rules, problems := ParseRules([]byte(preamble +
			"| T1 | A rule. | A reason. | skip:`out_of_scope` | prompt* (note, with a comma) |\n" +
			"| L3 | A rule. | A reason. | needs_human (a note) | prompt, PR job |\n\n" +
			"| Rule | Question for a person |\n|---|---|\n| L3 | Which replacement? |\n"))
		assert.Empty(t, problems)
		assert.Equal(t, map[string]int{"T1": 14, "L3": 15}, rules.Lines)
		assert.Equal(t, map[string]string{"out_of_scope": "not tracked"}, rules.Reasons)
		assert.Equal(t, map[string]string{"price": "rates"}, rules.Fields)
		assert.Equal(t, map[string]string{"L3": "Which replacement?"}, rules.Questions)
		assert.Equal(t, map[string][]string{"T1": {"skip:`out_of_scope`"}, "L3": {"needs_human"}}, rules.Produces)
	})

	tests := []struct {
		name        string
		rows        string
		wantProblem string
	}{
		{"duplicate ID", "| T1 | A. | B. | change | prompt |\n| T1 | C. | D. | change | prompt |\n", "duplicate rule ID T1"},
		{"short row", "| T1 | A. | B. | change |\n", "without 5 cells"},
		{"bad ID", "| t1 | A. | B. | change | prompt |\n", "bad rule ID"},
		{"empty why", "| T1 | A. |  | change | prompt |\n", "empty rule or why"},
		{"unknown produces", "| T1 | A. | B. | change, maybe | prompt |\n", "unknown Produces maybe"},
		{"unknown skip reason", "| T1 | A. | B. | skip:`typo` | prompt |\n", "unknown skip reason"},
		{"unknown enforced", "| T1 | A. | B. | change | prompt, hope |\n", "unknown Enforced hope"},
		{"a rule that asks a person needs a question", "| L3 | A. | B. | change, needs_human | prompt |\n", "L3 can send a row to a person but has no question"},
		{"a question for an unknown rule", "| T1 | A. | B. | change | prompt |\n\n| Rule | Question for a person |\n|---|---|\n| X9 | Why? |\n", "a question for unknown rule X9"},
		{"a question for a rule that asks nobody", "| T1 | A. | B. | change | prompt |\n\n| Rule | Question for a person |\n|---|---|\n| T1 | Why? |\n", "T1 has a question but never sends a row to a person"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, problems := ParseRules([]byte(preamble + tt.rows))
			assert.Contains(t, strings.Join(problems, "\n"), tt.wantProblem)
		})
	}
}

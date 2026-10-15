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

	const preamble = "- **Outcome.** `change` (a hunk) · `skip:<reason>` (a row) · `not listed` (no row) · `code` (listed as code) · `needs_human` (asks)\n\n" +
		"| Skip reason | Meaning |\n|---|---|\n| `out_of_scope` | not tracked |\n\n" +
		"| Field | Covers |\n|---|---|\n| `price` | rates |\n| `lifecycle` | dates |\n\n"
	const defaults = "| Default | Field | When | Bot does | Why | Outcome |\n|---|---|---|---|---|---|\n"
	const asks = "| Ask | When | Question | Why |\n|---|---|---|---|\n"

	t.Run("reads defaults, asks, fields and offered regions", func(t *testing.T) {
		t.Parallel()

		rules, problems := ParseRules([]byte(preamble +
			"| Host | Offered | Not offered |\n|---|---|---|\n| bedrock | `global` `us` | `in` (asked 2026-10-15) |\n\n" + defaults +
			"| price-follow | `price` | A host rate differs. | Set it. | Rates move. | change |\n" +
			"| too-old | – | Released over 90 days ago. | Leave it out. | Left out on purpose. | skip:`out_of_scope` (a note, with a comma) |\n\n" + asks +
			"| no-rule | No rule fits. | What should change? | A person decides. |\n"))
		assert.Empty(t, problems)
		assert.Equal(t, map[string]int{"price-follow": 18, "too-old": 19, "no-rule": 23}, rules.Lines)
		assert.Equal(t, map[string][]string{"price-follow": {"change"}, "too-old": {"skip:`out_of_scope`"}, "no-rule": {"needs_human"}}, rules.Outcomes)
		assert.Equal(t, map[string][]string{"price-follow": {"price"}}, rules.RuleFields)
		assert.Equal(t, map[string]string{"no-rule": "What should change?"}, rules.Questions)
		assert.Equal(t, map[string]string{"out_of_scope": "not tracked"}, rules.Reasons)
		assert.Equal(t, map[string]string{"price": "rates", "lifecycle": "dates"}, rules.Fields)
		assert.Equal(t, map[string][]string{"bedrock": {"global", "us"}}, rules.Offered)
	})

	tests := []struct {
		name        string
		rows        string
		wantProblem string
	}{
		{"letter ID rejected", defaults + "| T1 | `price` | A. | B. | C. | change |\n", "bad rule ID T1"},
		{"slug without a hyphen", defaults + "| t1 | `price` | A. | B. | C. | change |\n", "bad rule ID t1"},
		{"duplicate ID", defaults + "| price-follow | `price` | A. | B. | C. | change |\n\n" + asks + "| price-follow | A. | B? | C. |\n", "duplicate rule ID price-follow"},
		{"short default row", defaults + "| price-follow | `price` | A. | B. | change |\n", "price-follow: a Default row has 6 cells"},
		{"short ask row", asks + "| no-rule | A. | B? |\n", "no-rule: an Ask row has 4 cells"},
		{"empty cell", defaults + "| price-follow | `price` | A. |  | C. | change |\n", "price-follow: empty cell"},
		{"unknown outcome", defaults + "| price-follow | `price` | A. | B. | C. | change, maybe |\n", "price-follow: unknown outcome maybe"},
		{"unknown skip reason", defaults + "| too-old | – | A. | B. | C. | skip:`typo` |\n", "too-old: unknown skip reason `typo`"},
		{"a default can't ask a person", defaults + "| price-follow | `price` | A. | B. | C. | change, needs_human |\n", "price-follow: a Default can't send a row to a person; make it an Ask"},
		{"an ask needs a question", asks + "| no-rule | A. |  | C. |\n", "no-rule: empty cell"},
		{"unknown field", defaults + "| price-follow | `colour` | A. | B. | C. | change |\n", "price-follow: unknown field colour"},
		{"a change needs a field", defaults + "| price-follow | – | A. | B. | C. | change |\n", "price-follow: a rule that makes a change names its field"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, problems := ParseRules([]byte(preamble + tt.rows))
			assert.Contains(t, strings.Join(problems, "\n"), tt.wantProblem)
		})
	}
}

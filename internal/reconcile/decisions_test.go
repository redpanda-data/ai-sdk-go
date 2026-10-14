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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseDecisions(t *testing.T) {
	t.Parallel()

	const file = `# a comment
decisions:
  - id: DEC-001
    offering: bedrock:eu.anthropic.claude-fable-5
    field: lifecycle
    kind: keep
    lock: {"stage": "ga"}
    reason: "No end of life is announced."
    decided: 2026-10-13
    review_by: 2027-01-13
    pr: 259
  - id: DEC-002
    offering: vertex:claude-example
    field: new_model
    kind: hold
    reason: "Not offered to users."
    decided: 2026-10-14
    review_by: 2027-01-14
`

	t.Run("reads entries and their lines", func(t *testing.T) {
		t.Parallel()

		entries, err := ParseDecisions([]byte(file))
		require.NoError(t, err)
		require.Len(t, entries, 2)

		assert.Equal(t, Decision{
			ID: "DEC-001", Offering: "bedrock:eu.anthropic.claude-fable-5", Field: "lifecycle", Kind: "keep",
			Lock: map[string]any{"stage": "ga"}, Reason: "No end of life is announced.",
			Decided: "2026-10-13", ReviewBy: "2027-01-13", PR: 259, Line: 3,
		}, entries[0])
		assert.Equal(t, 12, entries[1].Line)
	})

	t.Run("rejects an unknown key", func(t *testing.T) {
		t.Parallel()

		_, err := ParseDecisions([]byte(file + "    colour: red\n"))
		assert.ErrorContains(t, err, "colour")
	})

	t.Run("an empty file has no decisions", func(t *testing.T) {
		t.Parallel()

		entries, err := ParseDecisions(nil)
		require.NoError(t, err)
		assert.Empty(t, entries)
	})
}

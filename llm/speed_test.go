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

package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeSpeed(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		raw  string
		want Speed
	}{
		{"empty stays empty (unreported)", "", ""},
		{"whitespace collapses to empty", "  \t", ""},

		// Standard synonyms.
		{"standard", "standard", SpeedStandard},
		{"default maps to standard", "default", SpeedStandard},
		{"mixed case standard", "Standard", SpeedStandard},
		{"padded standard", "  STANDARD  ", SpeedStandard},

		// Fast synonyms — the point of the semantic design. "rapid" is
		// not aliased today (no producer); it would pass through as
		// Speed("rapid") until we add it to the table.
		{"anthropic fast", "fast", SpeedFast},
		{"bedrock optimized", "optimized", SpeedFast},
		{"mixed case fast", "Fast", SpeedFast},
		{"unaliased rapid preserved verbatim", "rapid", Speed("rapid")},

		// Dash-to-underscore normalization for unknowns.
		{"unknown value preserved", "turbo_mode", Speed("turbo_mode")},
		{"unknown dash normalized", "turbo-mode", Speed("turbo_mode")},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, c.want, NormalizeSpeed(c.raw))
		})
	}
}

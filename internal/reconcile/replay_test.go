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
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestExactReplay checks the replay check on small catalogues: a bot
// commit replayed onto a newer base passes only when it is exactly the
// original change, and anything else the base or the replay moved is a
// reason to run the agent instead.
func TestExactReplay(t *testing.T) {
	t.Parallel()

	// A catalogue with Bedrock copies of Opus 4.7, 4.8 and 4.9 at these
	// input rates (0 leaves the copy out), Sonnet 5.5's Bedrock copy, and
	// Opus 4.7's facts.
	catalogue := func(opus47, opus48, opus49, sonnet int, knowledge string, derived string) Snapshot {
		t.Helper()

		offering := func(id string, input int) string {
			return `{"id": "global.anthropic.claude-` + id + `", "model": "anthropic/claude-` + id + `", "pricing": {"input": ` + strconv.Itoa(input) + `}, "derived": {"note": "` + derived + `"}}`
		}

		var offerings []string

		for _, entry := range []struct {
			id    string
			input int
		}{{"opus-4-7", opus47}, {"opus-4-8", opus48}, {"opus-4-9", opus49}, {"sonnet-5-5", sonnet}} {
			if entry.input > 0 {
				offerings = append(offerings, offering(entry.id, entry.input))
			}
		}

		snap, err := ParseSnapshot([]byte(`{"facts": {"anthropic/claude-opus-4-7": {"knowledge": "` + knowledge + `"}},
			"providers": [{"provider": "aws.bedrock", "offerings": [` + strings.Join(offerings, ", ") + `]}]}`))
		require.NoError(t, err)

		return snap
	}

	// The bot raised Opus 4.7's Bedrock rate from 5 to 6 on a catalogue
	// with 4.7 and 4.8, where Sonnet 5.5 is 3.
	parent := catalogue(5, 5, 0, 3, "2025-01", "")
	bot := catalogue(6, 5, 0, 3, "2025-01", "")
	asked := Report{NeedsHuman: []NeedsHuman{{Offering: "bedrock:global.anthropic.claude-sonnet-5-5", Field: "price", Rule: "no-rule"}}}

	tests := []struct {
		name         string
		parent, bot  Snapshot
		base, replay Snapshot
		report       Report
		want         []string
	}{
		{
			name: "a clean replay onto the same base", parent: parent, bot: bot,
			base: parent, replay: bot,
		},
		{
			name: "an unrelated change on the base", parent: parent, bot: bot,
			base: catalogue(5, 5, 0, 4, "2025-01", ""), replay: catalogue(6, 5, 0, 4, "2025-01", ""),
		},
		{
			// The case that sank patch replays: main dropped 4.7 and added
			// 4.9, and the same hunk then applied cleanly to 4.8.
			name: "a change that lands on a neighbour", parent: parent, bot: bot,
			base: catalogue(0, 5, 5, 3, "2025-01", ""), replay: catalogue(0, 6, 5, 3, "2025-01", ""),
			want: []string{
				"bedrock:global.anthropic.claude-opus-4-7: the base changed it after the bot's commit",
				"bedrock:global.anthropic.claude-opus-4-7: the replay doesn't make the bot's change",
				"bedrock:global.anthropic.claude-opus-4-8: the replay changes it, and the bot's commit didn't",
			},
		},
		{
			name: "the base already holds the bot's change", parent: parent, bot: bot,
			base: bot, replay: bot,
			want: []string{"bedrock:global.anthropic.claude-opus-4-7: the base changed it after the bot's commit"},
		},
		{
			name: "a question about an offering the base changed", parent: parent, bot: bot, report: asked,
			base: catalogue(5, 5, 0, 4, "2025-01", ""), replay: catalogue(6, 5, 0, 4, "2025-01", ""),
			want: []string{"bedrock:global.anthropic.claude-sonnet-5-5: a report row names it, and the base changed it after the bot's commit"},
		},
		{
			name: "a question about an offering the base left alone", parent: parent, bot: bot, report: asked,
			base: catalogue(5, 5, 0, 3, "2025-02", ""), replay: catalogue(6, 5, 0, 3, "2025-02", ""),
		},
		{
			name: "the bot's facts change, replayed", parent: parent, bot: catalogue(5, 5, 0, 3, "2025-02", ""),
			base: catalogue(5, 5, 5, 3, "2025-01", ""), replay: catalogue(5, 5, 5, 3, "2025-02", ""),
		},
		{
			name: "the bot's facts change, lost", parent: parent, bot: catalogue(5, 5, 0, 3, "2025-02", ""),
			base: parent, replay: parent,
			want: []string{"anthropic/claude-opus-4-7 (facts): the replay doesn't make the bot's change"},
		},
		{
			name: "derived values that moved with their neighbours", parent: parent, bot: bot,
			base: parent, replay: catalogue(6, 5, 0, 3, "2025-01", "regenerated"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, ReplayMismatches(tt.parent, tt.bot, tt.base, tt.replay, tt.report, "anthropic"))
		})
	}
}

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

package bedrock

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFinalizeToolArgs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		in       string
		want     string
		wantOK   bool
		wantNone bool
	}{
		{name: "empty coerces to object", in: "", want: "{}", wantOK: true},
		{name: "valid passes through", in: `{"q":"SELECT 1"}`, want: `{"q":"SELECT 1"}`, wantOK: true},
		{name: "valid empty object", in: `{}`, want: `{}`, wantOK: true},
		{name: "truncated is dropped", in: `{"q":`, wantOK: false, wantNone: true},
		{name: "garbage is dropped", in: `not json`, wantOK: false, wantNone: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			acc := &contentBlockAccumulator{toolArgs: tc.in}

			got, ok := acc.finalizeToolArgs()
			assert.Equal(t, tc.wantOK, ok)

			if tc.wantNone {
				assert.Nil(t, got)
			} else {
				assert.Equal(t, tc.want, string(got))
			}
		})
	}
}

// TestBuildFinalParts_DropsPartialToolUse covers the streaming wedge: when
// Bedrock's Converse stream is cut off (typically StopReasonMaxTokens) while
// accumulating ContentBlockDelta toolUse input deltas, the accumulator holds
// truncated JSON like `{"q":`. Without the finalize guard, buildFinalParts
// used to hand that block back as a tool_use with coerced `{}` args (hiding
// the fact that the call was partial) and earlier bedrock revisions would
// have shipped the truncated bytes verbatim — poisoning session replay the
// same way the anthropic bug did.
func TestBuildFinalParts_DropsPartialToolUse(t *testing.T) {
	t.Parallel()

	m := &Model{}

	blocks := map[int]*contentBlockAccumulator{
		0: {
			index:     0,
			blockType: blockTypeToolUse,
			toolUse:   &toolUseData{ID: "tool_ok", Name: "query"},
			toolArgs:  `{"q":"SELECT 1"}`,
		},
		1: {
			index:     1,
			blockType: blockTypeToolUse,
			toolUse:   &toolUseData{ID: "tool_partial", Name: "query"},
			toolArgs:  `{"q":`, // stream ended mid-delta
		},
	}

	parts := m.buildFinalParts(blocks)

	require.Len(t, parts, 1, "partial tool_use must not leak into final parts")
	require.NotNil(t, parts[0].ToolRequest)
	assert.Equal(t, "tool_ok", parts[0].ToolRequest.ID)
	assert.JSONEq(t, `{"q":"SELECT 1"}`, string(parts[0].ToolRequest.Arguments))
}

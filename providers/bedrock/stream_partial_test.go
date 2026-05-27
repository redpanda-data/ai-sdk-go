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

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestBuildFinalParts_DropsPartialToolUse covers the streaming wedge: when
// Bedrock's Converse stream is cut off (typically StopReasonMaxTokens) while
// accumulating ContentBlockDelta toolUse input deltas, the accumulator holds
// truncated JSON like `{"q":`. buildFinalParts must drop that block rather
// than ship the partial accumulation downstream — otherwise the truncated
// bytes reach session state and poison every subsequent replay the same way
// the anthropic bug did.
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
	tr, ok := parts[0].(*llm.ToolRequestPart)
	require.True(t, ok)
	assert.Equal(t, "tool_ok", tr.ID)
	assert.JSONEq(t, `{"q":"SELECT 1"}`, string(tr.Arguments))
}

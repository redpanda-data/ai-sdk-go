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

package openaicompat

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapFinishReason_TruncationWithToolCalls locks in the rule that
// truncation signals win over the hasToolCalls-based ToolCalls upgrade.
// See providers/anthropic/response_mapper_test.go for the full rationale.
func TestMapFinishReason_TruncationWithToolCalls(t *testing.T) {
	t.Parallel()

	mapper := &ResponseMapper{}

	cases := []struct {
		name         string
		reason       string
		hasToolCalls bool
		want         llm.FinishReason
	}{
		{"stop + tool calls promotes to ToolCalls", "stop", true, llm.FinishReasonToolCalls},
		{"stop without tool calls stays Stop", "stop", false, llm.FinishReasonStop},
		{"tool_calls + tool calls stays ToolCalls", "tool_calls", true, llm.FinishReasonToolCalls},
		{"length + tool calls stays Length", "length", true, llm.FinishReasonLength},
		{"length without tool calls stays Length", "length", false, llm.FinishReasonLength},
		{"content_filter + tool calls stays ContentFilter", "content_filter", true, llm.FinishReasonContentFilter},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, err := mapper.mapFinishReason(tc.reason, tc.hasToolCalls)
			require.NoError(t, err)
			assert.Equal(t, tc.want, got)
		})
	}
}

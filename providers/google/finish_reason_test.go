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

package google

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"google.golang.org/genai"

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
		reason       genai.FinishReason
		hasToolCalls bool
		want         llm.FinishReason
	}{
		{"stop + tool calls promotes to ToolCalls", genai.FinishReasonStop, true, llm.FinishReasonToolCalls},
		{"stop without tool calls stays Stop", genai.FinishReasonStop, false, llm.FinishReasonStop},
		{"max_tokens + tool calls stays Length", genai.FinishReasonMaxTokens, true, llm.FinishReasonLength},
		{"max_tokens without tool calls stays Length", genai.FinishReasonMaxTokens, false, llm.FinishReasonLength},
		{"safety + tool calls stays ContentFilter", genai.FinishReasonSafety, true, llm.FinishReasonContentFilter},
		{"recitation + tool calls stays ContentFilter", genai.FinishReasonRecitation, true, llm.FinishReasonContentFilter},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := mapper.mapFinishReason(tc.reason, tc.hasToolCalls)
			assert.Equal(t, tc.want, got)
		})
	}
}

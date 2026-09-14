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

package a2a

import (
	"encoding/json"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMessageFromLLM_ToolSearchPart: hosted tool search is part of the model's
// turn and reaches A2A clients as a typed DataPart instead of vanishing.
func TestMessageFromLLM_ToolSearchPart(t *testing.T) {
	t.Parallel()

	msg := llm.NewMessage(llm.RoleAssistant,
		llm.NewTextPart("Let me find a tool."),
		&llm.ToolSearchPart{
			Provider: "anthropic",
			Data:     json.RawMessage(`{"type":"tool_search_tool_result"}`),
			Tools:    []string{"jira__create_issue"},
		},
	)

	converted := MessageFromLLM(msg)
	require.Len(t, converted.Parts, 2)

	data, ok := converted.Parts[1].(a2a.DataPart)
	require.True(t, ok, "tool search becomes a DataPart")
	assert.Equal(t, "tool_search", data.Metadata["data_type"])
	assert.Equal(t, "anthropic", data.Data["provider"])
	assert.Equal(t, []any{"jira__create_issue"}, data.Data["tools"])
}

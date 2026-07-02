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
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapMessages_EmptyAssistantContentIsRepaired is the Gemini parity
// regression for the empty-content replay bug. Unlike openaicompat — which
// omits an empty assistant turn by construction — the google mapper appends a
// Content for every assistant turn, so an empty turn (a max_tokens cut whose
// only block was a dropped partial tool_use, or a reasoning-only turn whose
// parts mapParts skips) would produce a Content with empty Parts, which Gemini
// rejects. The guard substitutes a single non-whitespace text part.
func TestMapMessages_EmptyAssistantContentIsRepaired(t *testing.T) {
	t.Parallel()

	rm := NewRequestMapper(&Config{ModelName: "gemini-2.5-flash"})

	messages := []llm.Message{
		{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run the tools")}},
		{Role: llm.RoleAssistant, Content: []llm.Part{}}, // poisoned truncated turn
		{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("continue")}},
	}

	contents, _, err := rm.mapMessages(messages)
	require.NoError(t, err)

	require.Len(t, contents, 3, "all three history messages must map through")

	assistant := contents[1]
	require.Equal(t, genai.RoleModel, assistant.Role)
	require.NotEmpty(t, assistant.Parts,
		"empty assistant turn mapped to a Content with empty Parts — Gemini rejects this")
	require.Len(t, assistant.Parts, 1, "repaired turn must carry exactly one part")

	part := assistant.Parts[0]
	assert.NotEmpty(t, part.Text, "substituted part must be a non-empty text part")
}

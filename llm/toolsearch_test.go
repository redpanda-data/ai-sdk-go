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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolSearchPartRoundTrip(t *testing.T) {
	t.Parallel()

	part := &ToolSearchPart{Provider: "openai", Data: json.RawMessage(`{"type":"tool_search_output","call_id":null}`), Tools: []string{"fetch"}}
	raw, err := MarshalPart(part)
	require.NoError(t, err)
	restored, err := UnmarshalPart(raw)
	require.NoError(t, err)
	require.Equal(t, part, restored)
	cloned, ok := ClonePart(part).(*ToolSearchPart)
	require.True(t, ok)

	cloned.Data[0] = '['
	cloned.Tools[0] = "changed"

	require.Equal(t, byte('{'), part.Data[0])
	require.Equal(t, "fetch", part.Tools[0])
	message := NewMessage(RoleAssistant, part)
	require.Empty(t, message.ToolRequests())
}

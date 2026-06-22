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

package tool_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestWithCallIDRoundTrip(t *testing.T) {
	t.Parallel()

	id, ok := tool.CallIDFromContext(context.Background())
	assert.False(t, ok)
	assert.Empty(t, id)

	ctx := tool.WithCallID(context.Background(), "call_42")
	id, ok = tool.CallIDFromContext(ctx)
	assert.True(t, ok)
	assert.Equal(t, "call_42", id)
}

// TestRegistryExecuteExposesToolCallID verifies the Registry makes the
// originating LLM tool-call id (req.ID) visible to the tool's Execute via the
// context, so transport tools can forward it downstream.
func TestRegistryExecuteExposesToolCallID(t *testing.T) {
	t.Parallel()

	var seen string

	var seenOK bool

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(&mockTool{
		name: "probe",
		execFunc: func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
			seen, seenOK = tool.CallIDFromContext(ctx)

			return json.RawMessage(`{"ok":true}`), nil
		},
	}))

	_, err := registry.Execute(t.Context(), &llm.ToolRequestPart{
		ID:        "call_xyz",
		Name:      "probe",
		Arguments: json.RawMessage(`{}`),
	})
	require.NoError(t, err)

	assert.True(t, seenOK, "tool-call id should be present on the execution context")
	assert.Equal(t, "call_xyz", seen)
}

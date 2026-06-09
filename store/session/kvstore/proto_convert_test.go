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

package kvstore

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
)

func TestProtoConvert_SessionStateRoundTrip(t *testing.T) {
	t.Parallel()

	expiry := time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC)

	original := &session.State{
		ID: "sess-1",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("deploy the service")),
			llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call_1", "deploy", json.RawMessage(`{"env":"prod"}`)),
			),
		},
		Metadata: map[string]any{"tenant": "acme"},
		PendingToolCalls: map[string]session.PendingToolCall{
			"call_1": {
				SchemaVersion: session.PendingToolCallSchemaVersion,
				ID:            "call_1",
				Name:          "deploy",
				Arguments:     json.RawMessage(`{"env":"prod"}`),
				Reason:        "tool_result",
				Resume:        "tool_response",
				Message:       "Deploying to prod…",
				State:         json.RawMessage(`{"deploy_id":"d-42"}`),
				CorrelationID: "d-42",
				CreatedAt:     time.Date(2026, 6, 9, 11, 0, 0, 0, time.UTC),
				ExpiresAt:     &expiry,
				LastOutput:    json.RawMessage(`{"status":"in_progress"}`),
				Progress: []session.ProgressEntry{
					{At: time.Date(2026, 6, 9, 11, 30, 0, 0, time.UTC), Payload: json.RawMessage(`{"pct":50}`)},
				},
				Metadata: map[string]any{"source": "webhook"},
			},
		},
		ResumeReceipts: map[string]session.ResumeReceipt{
			"call_0": {
				CallID:     "call_0",
				ResultHash: "def456",
				ResolvedAt: time.Date(2026, 6, 9, 10, 0, 0, 0, time.UTC),
				Metadata:   map[string]any{"by": "operator"},
			},
		},
	}

	pb, err := toProtoSessionState(original)
	require.NoError(t, err)

	got, err := FromProtoSessionState(pb)
	require.NoError(t, err)

	assert.Equal(t, original, got)
}

func TestProtoConvert_LegacyToolResponseError(t *testing.T) {
	t.Parallel()

	// The legacy wire format stored the failure message in Error with an
	// empty Result. It must decode to the {"error": ...} payload the new
	// code produces, so Result is always valid JSON.
	got, err := fromProtoPart(&llmpb.Part{
		Kind: llmpb.PartKind_PART_KIND_TOOL_RESPONSE,
		Data: &llmpb.Part_ToolResponse{
			ToolResponse: &llmpb.ToolResponse{
				Id:    "call_1",
				Name:  "deploy",
				Error: "boom",
			},
		},
	})
	require.NoError(t, err)

	assert.Equal(t, llm.NewToolErrorPart("call_1", "deploy", "boom"), got)

	resp, ok := got.(*llm.ToolResponsePart)
	require.True(t, ok)
	assert.JSONEq(t, `{"error":"boom"}`, string(resp.Result))
}

func TestProtoConvert_SessionStateRoundTrip_EmptyPending(t *testing.T) {
	t.Parallel()

	original := &session.State{
		ID:       "sess-2",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}

	pb, err := toProtoSessionState(original)
	require.NoError(t, err)
	assert.Nil(t, pb.PendingToolCalls)
	assert.Nil(t, pb.ResumeReceipts)

	got, err := FromProtoSessionState(pb)
	require.NoError(t, err)
	assert.Equal(t, original, got)
}

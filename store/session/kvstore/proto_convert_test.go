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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestProtoSessionState_ConversationIDRoundTrip guards the Schema Registry
// persistence path: the protobuf conversion is field-by-field, so a State
// field that is not wired through both converters silently disappears after a
// save/load.
func TestProtoSessionState_ConversationIDRoundTrip(t *testing.T) {
	t.Parallel()

	orig := &session.State{
		ID:             "agent-tool-child-1",
		ConversationID: "root-conv",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello")),
		},
		Metadata: map[string]any{"locale": "en"},
	}

	pb, err := toProtoSessionState(orig)
	require.NoError(t, err)
	assert.Equal(t, "root-conv", pb.ConversationId)

	got, err := FromProtoSessionState(pb)
	require.NoError(t, err)

	assert.Equal(t, orig.ID, got.ID)
	assert.Equal(t, "root-conv", got.ConversationID)
	assert.Equal(t, orig.Metadata, got.Metadata)
	require.Len(t, got.Messages, 1)
}

func TestProtoSessionState_EmptyConversationID(t *testing.T) {
	t.Parallel()

	orig := &session.State{ID: "root-1"}

	pb, err := toProtoSessionState(orig)
	require.NoError(t, err)
	assert.Empty(t, pb.ConversationId)

	got, err := FromProtoSessionState(pb)
	require.NoError(t, err)
	assert.Empty(t, got.ConversationID)
	assert.Equal(t, "root-1", session.ConversationID(got))
}

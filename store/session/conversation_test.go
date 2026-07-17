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

package session_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func TestConversationID_Resolution(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		s    *session.State
		want string
	}{
		{
			name: "nil session",
			s:    nil,
			want: "",
		},
		{
			name: "own conversation falls back to session id",
			s:    &session.State{ID: "sess-1"},
			want: "sess-1",
		},
		{
			name: "override wins over session id",
			s:    &session.State{ID: "agent-tool-child-1", ConversationID: "root-conv"},
			want: "root-conv",
		},
		{
			name: "empty session entirely",
			s:    &session.State{},
			want: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, session.ConversationID(tt.s))
		})
	}
}

func TestState_ConversationIDJSONRoundTrip(t *testing.T) {
	t.Parallel()

	orig := &session.State{ID: "child-1", ConversationID: "root-conv"}

	data, err := json.Marshal(orig)
	require.NoError(t, err)
	assert.Contains(t, string(data), `"conversation_id":"root-conv"`)

	var got session.State

	require.NoError(t, json.Unmarshal(data, &got))
	assert.Equal(t, "root-conv", got.ConversationID)

	// A session without an override omits the field entirely.
	data, err = json.Marshal(&session.State{ID: "root-1"})
	require.NoError(t, err)
	assert.NotContains(t, string(data), "conversation_id")
}

func TestState_CloneCopiesConversationID(t *testing.T) {
	t.Parallel()

	orig := &session.State{ID: "child-1", ConversationID: "root-conv"}

	clone := orig.Clone()
	assert.Equal(t, "root-conv", clone.ConversationID)
}

func TestInMemoryStore_ListIncludesConversationID(t *testing.T) {
	t.Parallel()

	store := session.NewInMemoryStore()
	ctx := t.Context()

	require.NoError(t, store.Save(ctx, &session.State{ID: "root-1"}))
	require.NoError(t, store.Save(ctx, &session.State{ID: "child-1", ConversationID: "root-1"}))

	resp, err := store.List(ctx, nil)
	require.NoError(t, err)
	require.Len(t, resp.Sessions, 2)

	byID := make(map[string]session.Summary, len(resp.Sessions))
	for _, s := range resp.Sessions {
		byID[s.ID] = s
	}

	assert.Empty(t, byID["root-1"].ConversationID)
	assert.Equal(t, "root-1", byID["child-1"].ConversationID)
}

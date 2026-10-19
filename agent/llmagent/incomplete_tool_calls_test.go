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

package llmagent

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestDetectIncompleteToolCalls(t *testing.T) {
	t.Parallel()

	user := llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))
	toolReq := llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call_1", "lookup", json.RawMessage(`{}`)))
	toolResp := llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("call_1", "lookup", json.RawMessage(`{}`), false))
	text := llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("done"))

	tests := []struct {
		name         string
		msgs         []llm.Message
		wantIDs      []string
		wantInsertAt int
	}{
		{name: "empty", msgs: nil},
		{name: "single user", msgs: []llm.Message{user}},
		{name: "trailing text assistant", msgs: []llm.Message{user, text}},
		{name: "runner shape: requests then new user message", msgs: []llm.Message{user, toolReq, user}, wantIDs: []string{"call_1"}, wantInsertAt: 2},
		{name: "resumed shape: requests are the tail", msgs: []llm.Message{user, toolReq}, wantIDs: []string{"call_1"}, wantInsertAt: 2},
		{name: "resumed shape with only the request", msgs: []llm.Message{toolReq}, wantIDs: []string{"call_1"}, wantInsertAt: 1},
		{name: "completed tool call", msgs: []llm.Message{user, toolReq, toolResp}},
		{name: "completed then new user message", msgs: []llm.Message{user, toolReq, toolResp, user}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reqs, insertAt := detectIncompleteToolCalls(tt.msgs)

			ids := make([]string, 0, len(reqs))
			for _, r := range reqs {
				ids = append(ids, r.ID)
			}

			if len(tt.wantIDs) == 0 {
				require.Empty(t, ids)

				return
			}

			assert.Equal(t, tt.wantIDs, ids)
			assert.Equal(t, tt.wantInsertAt, insertAt)
		})
	}
}

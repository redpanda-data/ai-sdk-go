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

package llmagent_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestRun_ContextOverflowSurfacesDistinctly verifies that a genuine
// context-window overflow (input too large for the model) propagates as the
// dedicated agent.FinishReasonContextOverflow, distinct from the output
// truncation signalled by FinishReasonLength.
func TestRun_ContextOverflowSurfacesDistinctly(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message:      llm.Message{Role: llm.RoleAssistant, Content: []llm.Part{}},
				FinishReason: llm.FinishReasonContextOverflow,
			}, nil
		})

	ag, err := llmagent.New("test-agent", "You are a helpful assistant", model)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "test-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("do the thing"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	events := collectEvents(t, ag.Run(t.Context(), inv))

	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonContextOverflow, endEvent.FinishReason)
}

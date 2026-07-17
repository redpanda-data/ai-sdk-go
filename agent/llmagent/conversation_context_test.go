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
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// conversationCapturingInterceptor records the conversation id visible on ctx
// during tool interception, proving the value is scoped to the whole executor
// chain, not just the final tool execution.
type conversationCapturingInterceptor struct {
	gotConversationID string
}

func (i *conversationCapturingInterceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponsePart, error) {
	i.gotConversationID = agent.ConversationIDFromContext(ctx)
	return next(ctx, info)
}

// runConversationIDCapture runs a one-tool-call agent turn against sess and
// returns the conversation id observed by the tool and by a tool interceptor.
func runConversationIDCapture(t *testing.T, sess *session.State) (string, string) {
	t.Helper()

	var toolSaw string

	captureTool := &mockTool{
		name: "capture",
		definition: llm.ToolDefinition{
			Name:        "capture",
			Description: "captures the conversation id from ctx",
		},
	}
	captureTool.executeFn = func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
		toolSaw = agent.ConversationIDFromContext(ctx)
		return json.RawMessage(`{}`), nil
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(captureTool))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstTurn()).
		Times(1).
		ThenRespondWithToolCall("capture", map[string]any{})
	model.When(fakellm.LastMessageHasToolResponse("capture")).
		ThenStreamText("done", fakellm.StreamConfig{})

	interceptor := &conversationCapturingInterceptor{}

	ag, err := llmagent.New(
		"capture-agent",
		"You are a test assistant",
		model,
		llmagent.WithTools(registry),
		llmagent.WithInterceptors(interceptor),
	)
	require.NoError(t, err)

	inv := agent.NewInvocationMetadata(sess, agent.Info{})
	collectEvents(t, ag.Run(t.Context(), inv))

	return toolSaw, interceptor.gotConversationID
}

// TestRun_ToolContextCarriesConversationID verifies the llmagent wiring: the
// conversation grouping id must reach the tool (and tool interceptors) via
// ctx, so agenttool can group sub-agents without manufacturing the context in
// tests.
func TestRun_ToolContextCarriesConversationID(t *testing.T) {
	t.Parallel()

	sess := &session.State{
		ID:       "root-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}

	toolSaw, interceptorSaw := runConversationIDCapture(t, sess)

	// A root session is its own conversation.
	assert.Equal(t, "root-session", toolSaw)
	assert.Equal(t, "root-session", interceptorSaw)
}

// TestRun_ToolContextCarriesConversationOverride verifies that a session with
// a ConversationID override (a sub-agent session) exposes the override — the
// root conversation — to tools, not its own storage id.
func TestRun_ToolContextCarriesConversationOverride(t *testing.T) {
	t.Parallel()

	sess := &session.State{
		ID:             "agent-tool-child-1",
		ConversationID: "root-conv",
		Messages:       []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}

	toolSaw, interceptorSaw := runConversationIDCapture(t, sess)

	assert.Equal(t, "root-conv", toolSaw)
	assert.Equal(t, "root-conv", interceptorSaw)
}

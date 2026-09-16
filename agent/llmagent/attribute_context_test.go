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

// attributeCapturingInterceptor records the attributes visible on ctx during
// tool interception, not just during the tool execution itself.
type attributeCapturingInterceptor struct {
	gotAttributes map[string]string
}

func (i *attributeCapturingInterceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponsePart, error) {
	i.gotAttributes = agent.AttributesFromContext(ctx)

	return next(ctx, info)
}

// ctx is the only channel into Tool.Execute, so it is the only way agenttool
// can hand the invocation's attributes to a sub-agent.
func TestRun_ToolContextCarriesAttributes(t *testing.T) {
	t.Parallel()

	var toolSaw map[string]string

	captureTool := &mockTool{
		name: "capture",
		definition: llm.ToolDefinition{
			Name:        "capture",
			Description: "captures the caller attributes from ctx",
		},
	}
	captureTool.executeFn = func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
		toolSaw = agent.AttributesFromContext(ctx)

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

	interceptor := &attributeCapturingInterceptor{}

	ag, err := llmagent.New(
		"capture-agent",
		"You are a test assistant",
		model,
		llmagent.WithTools(registry),
		llmagent.WithInterceptors(interceptor),
	)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "root-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}
	inv := agent.NewInvocationMetadata(sess, agent.Info{}, agent.WithAttributes(
		map[string]string{agent.AttrUserID: "alice@example.test", "user.tier": "premium"}))

	collectEvents(t, ag.Run(t.Context(), inv))

	want := map[string]string{agent.AttrUserID: "alice@example.test", "user.tier": "premium"}
	assert.Equal(t, want, toolSaw)
	assert.Equal(t, want, interceptor.gotAttributes)
}

func TestRun_ToolContextCarriesNoAttributesWhenNoneAsserted(t *testing.T) {
	t.Parallel()

	var toolSaw map[string]string

	captureTool := &mockTool{
		name: "capture",
		definition: llm.ToolDefinition{
			Name:        "capture",
			Description: "captures the caller attributes from ctx",
		},
	}
	captureTool.executeFn = func(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
		toolSaw = agent.AttributesFromContext(ctx)

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

	ag, err := llmagent.New(
		"capture-agent",
		"You are a test assistant",
		model,
		llmagent.WithTools(registry),
	)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "root-session",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}

	collectEvents(t, ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})))

	assert.Empty(t, toolSaw)
}

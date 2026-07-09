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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
)

func TestGlobalInstructionPropagation(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// 1. Setup Child Agent
	childFake := fakellm.NewFakeModel()
	childFake.When(fakellm.Any()).ThenRespondText("Child response")
	
	childAgent, err := llmagent.New("child", "I am the child.", childFake)
	require.NoError(t, err)

	// 2. Setup Parent Agent with Child as a Tool
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(agenttool.New(childAgent)))

	parentFake := fakellm.NewFakeModel()
	// Rule to make the parent call the child tool
	parentFake.When(fakellm.UserMessageContains("delegate")).
		ThenRespondWithToolCall("child", map[string]any{"task": "do something"})
	// Rule for the second turn (after tool result)
	parentFake.When(fakellm.Any()).ThenRespondText("Parent done")

	parentAgent, err := llmagent.New("parent", "I am the parent.", parentFake, llmagent.WithTools(registry))
	require.NoError(t, err)

	// 3. Execute with Global Instructions in Context
	gctx := agent.ContextWithGlobalInstructions(ctx, "CRITICAL: ALWAYS USE JSON.")
	
	sess := &session.State{ID: "parent-sess"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("delegate to child")))
	inv := agent.NewInvocationMetadata(sess, parentAgent.Info())

	for evt, err := range parentAgent.Run(gctx, inv) {
		require.NoError(t, err)
		_ = evt
	}

	// 4. Verify Parent Prompt
	parentCalls := parentFake.Calls()
	require.NotEmpty(t, parentCalls)
	parentSystemMsg := findSystemMessage(parentCalls[0].Request)
	assert.Contains(t, parentSystemMsg.TextContent(), "CRITICAL: ALWAYS USE JSON.")

	// 5. Verify Child Prompt (Propagation check)
	childCalls := childFake.Calls()
	require.NotEmpty(t, childCalls, "Child agent should have been called as a tool")
	childSystemMsg := findSystemMessage(childCalls[0].Request)
	assert.Contains(t, childSystemMsg.TextContent(), "CRITICAL: ALWAYS USE JSON.", 
		"Child agent should have inherited global instructions from parent context")
}

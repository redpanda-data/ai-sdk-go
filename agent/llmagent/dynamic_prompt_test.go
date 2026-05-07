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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func TestDynamicSystemPrompt(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Base prompt with various placeholders
	basePrompt := "Hello {user_name}! Today is {current_date}. Your role is {role}. App: {app_name}."

	// Setup fake model to capture the request
	fake := fakellm.NewFakeModel()
	fake.When(fakellm.Any()).ThenRespondText("Done")

	// Create agent
	a, err := llmagent.New("test-agent", basePrompt, fake)
	require.NoError(t, err)

	// Case 1: Session Metadata
	t.Run("Session Metadata", func(t *testing.T) {
		sess := &session.State{
			ID: "test-sess",
			Metadata: map[string]any{
				"user_name": "Alice",
				"role":      "Assistant",
				"app_name":  "TestApp",
			},
		}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")))
		inv := agent.NewInvocationMetadata(sess, a.Info())

		for evt, err := range a.Run(ctx, inv) {
			require.NoError(t, err)
			_ = evt // skip events
		}

		// Inspect the captured request
		reqs := fake.Calls()
		require.Len(t, reqs, 1)
		
		systemMsg := findSystemMessage(reqs[0].Request)
		require.NotNil(t, systemMsg)
		
		expectedDate := time.Now().UTC().Format("2006-01-02")
		assert.Contains(t, systemMsg.TextContent(), "Hello Alice!")
		assert.Contains(t, systemMsg.TextContent(), "Today is "+expectedDate)
		assert.Contains(t, systemMsg.TextContent(), "Your role is Assistant")
		assert.Contains(t, systemMsg.TextContent(), "App: TestApp")
	})

	// Case 2: Invocation Metadata Overrides Session
	t.Run("Invocation Override", func(t *testing.T) {
		fake.ResetCalls()
		sess := &session.State{
			ID: "test-sess",
			Metadata: map[string]any{
				"user_name": "Alice",
			},
		}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")))
		inv := agent.NewInvocationMetadata(sess, a.Info())
		inv.SetMetadata("user_name", "Bob") // Override session

		for evt, err := range a.Run(ctx, inv) {
			require.NoError(t, err)
			_ = evt
		}

		reqs := fake.Calls()
		systemMsg := findSystemMessage(reqs[0].Request)
		assert.Contains(t, systemMsg.TextContent(), "Hello Bob!")
	})

	// Case 3: Global Instructions from Context
	t.Run("Global Instructions", func(t *testing.T) {
		fake.ResetCalls()
		sess := &session.State{ID: "test-sess"}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")))
		inv := agent.NewInvocationMetadata(sess, a.Info())

		// Add global instructions to context
		gctx := agent.ContextWithGlobalInstructions(ctx, "Be extremely polite.")

		for evt, err := range a.Run(gctx, inv) {
			require.NoError(t, err)
			_ = evt
		}

		reqs := fake.Calls()
		systemMsg := findSystemMessage(reqs[0].Request)
		
		assert.Contains(t, systemMsg.TextContent(), "---")
		assert.Contains(t, systemMsg.TextContent(), "## Global Instructions")
		assert.Contains(t, systemMsg.TextContent(), "Be extremely polite.")
	})

	// Case 4: JSON Safety
	t.Run("JSON Safety", func(t *testing.T) {
		fake.ResetCalls()
		jsonPrompt := `Format: {"name": "{user_name}", "data": { "key": "val" }}`
		a2, _ := llmagent.New("json-agent", jsonPrompt, fake)
		
		sess := &session.State{
			ID: "test-sess",
			Metadata: map[string]any{"user_name": "Alice"},
		}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")))
		inv := agent.NewInvocationMetadata(sess, a2.Info())

		for evt, err := range a2.Run(ctx, inv) {
			require.NoError(t, err)
			_ = evt
		}

		reqs := fake.Calls()
		systemMsg := findSystemMessage(reqs[0].Request)
		
		// Should replace {user_name} but NOT touch { "key": "val" }
		assert.Contains(t, systemMsg.TextContent(), `"name": "Alice"`)
		assert.Contains(t, systemMsg.TextContent(), `"data": { "key": "val" }`)
	})
}

func findSystemMessage(req *llm.Request) *llm.Message {
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem {
			return &m
		}
	}
	return nil
}

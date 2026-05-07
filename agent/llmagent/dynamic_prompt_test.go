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
	"fmt"
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

func TestDynamicInstructionProvider(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// Setup fake model to capture the request
	fake := fakellm.NewFakeModel()
	fake.When(fakellm.Any()).ThenRespondText("Done")

	// Create agent with InstructionProvider
	a, err := llmagent.New("test-agent", "Fallback prompt", fake,
		llmagent.WithInstructionProvider(func(ctx context.Context, inv *agent.InvocationMetadata) (string, error) {
			user := "Unknown"
			if v, ok := inv.Metadata()["user_name"]; ok {
				user = v.(string)
			} else if sess := inv.Session(); sess != nil && sess.Metadata != nil {
				if v, ok := sess.Metadata["user_name"]; ok {
					user = v.(string)
				}
			}

			role := "Assistant"
			if v, ok := inv.Metadata()["role"]; ok {
				role = v.(string)
			}

			date := time.Now().UTC().Format("2006-01-02")
			return fmt.Sprintf("Hello %s! Today is %s. Your role is %s.", user, date, role), nil
		}),
	)
	require.NoError(t, err)

	// Case 1: Session Metadata
	t.Run("Session Metadata", func(t *testing.T) {
		sess := &session.State{
			ID: "test-sess",
			Metadata: map[string]any{
				"user_name": "Alice",
			},
		}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")))
		inv := agent.NewInvocationMetadata(sess, a.Info())

		for evt, err := range a.Run(ctx, inv) {
			require.NoError(t, err)
			_ = evt
		}

		reqs := fake.Calls()
		require.Len(t, reqs, 1)
		
		systemMsg := findSystemMessage(reqs[0].Request)
		require.NotNil(t, systemMsg)
		
		expectedDate := time.Now().UTC().Format("2006-01-02")
		assert.Contains(t, systemMsg.TextContent(), "Hello Alice!")
		assert.Contains(t, systemMsg.TextContent(), "Today is "+expectedDate)
		assert.Contains(t, systemMsg.TextContent(), "Your role is Assistant")
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
		inv.SetMetadata("user_name", "Bob")
		inv.SetMetadata("role", "Expert")

		for evt, err := range a.Run(ctx, inv) {
			require.NoError(t, err)
			_ = evt
		}

		reqs := fake.Calls()
		systemMsg := findSystemMessage(reqs[0].Request)
		assert.Contains(t, systemMsg.TextContent(), "Hello Bob!")
		assert.Contains(t, systemMsg.TextContent(), "Your role is Expert")
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
}

func findSystemMessage(req *llm.Request) *llm.Message {
	for _, m := range req.Messages {
		if m.Role == llm.RoleSystem {
			return &m
		}
	}
	return nil
}

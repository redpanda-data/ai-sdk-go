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

// Package agenttool wraps agents as tools for hierarchical composition and context isolation.
//
// AgentTool enables parent agents to delegate subtasks to child agents with fresh sessions.
// Each invocation creates an isolated context, useful for:
//   - Context management: offload subtasks without polluting main agent context
//   - Tool access: child agents can have different tools than parent
//   - Focused execution: each subtask gets clean context
//
// Usage:
//
//	// Create assistant with tools
//	assistant := llmagent.New("assistant", "You are helpful...", model,
//	    llmagent.WithTools(toolRegistry))
//
//	// Main agent delegates via agenttool
//	mainTools := tool.NewRegistry(tool.RegistryConfig{})
//	mainTools.Register(agenttool.New(assistant))
package agenttool

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// AgentTool wraps an Agent as a Tool, enabling hierarchical agent composition.
// Each invocation creates a fresh session for the agent.
type AgentTool struct {
	agent agent.Agent
}

// New creates a new AgentTool that wraps the given agent as a tool.
func New(a agent.Agent) tool.Tool {
	return &AgentTool{agent: a}
}

// Name implements tool.Tool.
func (at *AgentTool) Name() string { return at.agent.Info().Name }

// Description implements tool.Tool.
func (at *AgentTool) Description() string { return at.agent.Info().Description }

// InputSchema implements tool.Tool.
func (at *AgentTool) InputSchema() json.RawMessage {
	schema := at.agent.InputSchema()

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		// Programming error: agent's InputSchema contains unmarshalable types
		// (channels, funcs, etc.). Surface a permissive fallback so the
		// registry can still expose the tool — the agent author can fix
		// the schema independently of tool wiring.
		return json.RawMessage(`{"type":"object"}`)
	}

	return schemaJSON
}

// ToolSpec implements tool.SpecProvider. Declaring AsyncHandoff makes
// the registry validate the handoff pauses this wrapper emits and adds
// the async hint to the model-visible description.
func (at *AgentTool) ToolSpec() tool.Spec {
	return tool.Spec{
		Name:        at.Name(),
		Description: at.Description(),
		InputSchema: at.InputSchema(),
		Async:       tool.AsyncHandoff(),
	}
}

// Result represents the output from an agent tool execution.
type Result struct {
	Result string `json:"result"`
}

// Execute implements tool.Tool by running the agent with a fresh session.
//
// Input Handling:
//   - Args are passed as JSON in a user message (e.g., {"query": "search X"})
//   - The child agent receives this as text and parses it naturally.
//
// Output Structure:
//   - Returns {"result": "<text>"} on terminal completion.
//   - If the child agent pauses (FinishReasonPaused), the parent tool
//     returns a Handoff await so the runner can drive the child via
//     Runner.Resume on the next iteration.
//
// Session Isolation:
//   - Each invocation creates a fresh session.
func (at *AgentTool) Execute(ctx context.Context, call tool.Call) (tool.Execution, error) {
	info := at.agent.Info()

	sess := &session.State{
		ID:       fmt.Sprintf("agent-tool-%s-%d", info.Name, time.Now().UnixNano()),
		Messages: []llm.Message{},
		Metadata: map[string]any{},
	}

	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart(string(call.Args)))
	sess.Messages = append(sess.Messages, userMsg)

	inv := agent.NewInvocationMetadata(sess, info)

	var (
		result    string
		paused    bool
		pauseInfo agent.InvocationEndEvent
	)

	for evt, err := range at.agent.Run(ctx, inv) {
		if err != nil {
			return tool.Execution{}, fmt.Errorf("agent execution failed: %w", err)
		}

		switch e := evt.(type) {
		case agent.MessageEvent:
			result = e.Response.Message.TextContent()
		case agent.InvocationEndEvent:
			if e.FinishReason == agent.FinishReasonPaused {
				paused = true
				pauseInfo = e
			}
		}
	}

	if paused {
		// Encode just enough child context for the parent to resume on
		// re-entry. Persisting the child's full session state by value
		// would bloat parent session storage; we store the session ID and
		// the surfaced pending-call summaries.
		state, err := json.Marshal(map[string]any{
			"child_session_id": sess.ID,
			"pending_calls":    pauseInfo.PendingCalls,
		})
		if err != nil {
			return tool.Execution{}, fmt.Errorf("encode handoff state: %w", err)
		}

		placeholder, err := json.Marshal(Result{Result: "Child agent paused awaiting input."})
		if err != nil {
			return tool.Execution{}, fmt.Errorf("encode handoff placeholder: %w", err)
		}

		return tool.Execution{
			Output: placeholder,
			Await: &tool.Await{
				Reason: tool.AwaitReasonHandoff,
				Resume: tool.ResumeWithReentry,
				State:  state,
			},
		}, nil
	}

	if result == "" {
		result = "Task completed with no text output."
	}

	output, err := json.Marshal(Result{Result: result})
	if err != nil {
		return tool.Execution{}, fmt.Errorf("marshal agent tool result: %w", err)
	}

	return tool.Execution{Output: output}, nil
}

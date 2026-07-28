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

	"github.com/rs/xid"

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

// Definition implements tool.Tool by using the agent's existing metadata.
func (at *AgentTool) Definition() llm.ToolDefinition {
	info := at.agent.Info()
	schema := at.agent.InputSchema()

	schemaJSON, err := json.Marshal(schema)
	if err != nil {
		// Programming error: agent's InputSchema contains unmarshalable types (channels, funcs, etc.)
		// This is caught at tool registration time, not by user input or parent agent calls.
		return llm.ToolDefinition{
			Name:        info.Name,
			Description: fmt.Sprintf("[SCHEMA ERROR] %s - Invalid InputSchema implementation: %v", info.Description, err),
			Parameters:  json.RawMessage(`{"type":"object"}`),
		}
	}

	return llm.ToolDefinition{
		Name:        info.Name,
		Description: info.Description,
		Parameters:  schemaJSON,
	}
}

// Result represents the output from an agent tool execution.
type Result struct {
	Result string `json:"result"`
	// Truncated is true when the sub-agent's turn stopped at its output-token
	// limit (agent.FinishReasonLength) rather than finishing naturally. The
	// result then holds only the partial content produced before the cut, so the
	// parent can tell an incomplete answer apart from a complete one — the
	// agent-as-tool analogue of the A2A executor's `truncated` marker.
	Truncated bool `json:"truncated,omitempty"`
}

// Execute implements tool.Tool by running the agent with a fresh session.
//
// Input Handling:
//   - Args are passed as JSON in a user message (e.g., {"query": "search X"})
//   - The child agent receives this as text and parses it naturally
//   - Modern LLMs reliably handle JSON parsing from text
//   - Alternative approaches (text wrapping, schema validation) add complexity
//     without proven value - the LLM handles malformed inputs by asking for clarification
//
// Output Structure:
//   - Returns {"result": "<text>"}
//   - Only the last assistant message is captured as the result
//   - Token usage is tracked separately via interceptors on the agent
//
// Session Isolation:
//   - The sub-agent always runs with a fresh, in-memory Messages slice and its
//     state is never loaded or persisted, so it cannot observe the parent's
//     conversation history. Context is isolated regardless of the session id.
//   - For context sharing, pass relevant information explicitly in args
//
// Session ID & conversation grouping:
//   - The sub-agent always gets its own freshly minted, collision-resistant
//     session id (xid). It never reuses the parent's id, so it cannot collide
//     with the parent's or a sibling sub-agent's session if it ever reaches a
//     store — there is no "safe only while unpersisted" caveat.
//   - When invoked as part of a parent agent's turn (the calling agent set the
//     conversation grouping id on ctx, see agent.ContextWithConversationID),
//     the sub-agent records it as the session's ConversationID (transitively,
//     the root conversation). Observability uses that to group the
//     parent→sub-agent tree under one conversation (gen_ai.conversation.id)
//     without overloading the storage id.
func (at *AgentTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	info := at.agent.Info()

	// Args are passed as JSON text (e.g., {"query": "..."}); the child agent
	// receives them in a user message and parses them naturally.
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart(string(args)))

	// The sub-agent gets its own unique storage session id; conversation
	// grouping does NOT rely on it. The parent's conversation id (transitively,
	// the root conversation) is carried separately in ConversationID — empty
	// when there is no calling conversation — so the otel plugin groups this
	// sub-agent under the same gen_ai.conversation.id as the parent.
	sess := &session.State{
		ID:             fmt.Sprintf("agent-tool-%s-%s", info.Name, xid.New().String()),
		ConversationID: agent.ConversationIDFromContext(ctx),
		Messages:       []llm.Message{userMsg},
		Metadata:       map[string]any{},
	}

	inv := agent.NewInvocationMetadata(sess, info)

	// Scope the grouping id to the child run so nested sub-agents group under
	// the same root even when the child agent implementation does not set it
	// before its own tool calls.
	ctx = agent.ContextWithConversationID(ctx, session.ConversationID(sess))

	// Run agent, collecting the last assistant message as the result and the
	// finish reason so a truncated turn can be flagged to the parent.
	var (
		result       string
		finishReason agent.FinishReason
	)

	for evt, err := range at.agent.Run(ctx, inv) {
		if err != nil {
			return nil, fmt.Errorf("agent execution failed: %w", err)
		}

		switch e := evt.(type) {
		case agent.MessageEvent:
			// Capture last assistant message as result.
			result = e.Response.Message.TextContent()
		case agent.InvocationEndEvent:
			finishReason = e.FinishReason
		}
	}

	if result == "" {
		result = "Task completed with no text output."
	}

	// Output truncation is non-fatal: the sub-agent stopped at its output-token
	// cap with a partial answer. Deliver the partial content but flag it so the
	// parent does not mistake it for a complete result.
	truncated := finishReason == agent.FinishReasonLength

	output := Result{
		Result:    result,
		Truncated: truncated,
	}

	return json.Marshal(output)
}

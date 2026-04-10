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
//   - Each invocation creates a fresh session (no context sharing)
//   - This prevents context pollution and keeps parent/child boundaries clear
//   - For context sharing, pass relevant information explicitly in args
func (at *AgentTool) Execute(ctx context.Context, args json.RawMessage) (tool.Result, error) {
	info := at.agent.Info()

	// 1. Create fresh session with unique ID to prevent collisions in state stores
	sess := &session.State{
		ID:       fmt.Sprintf("agent-tool-%s-%d", info.Name, time.Now().UnixNano()),
		Messages: []llm.Message{},
		Metadata: map[string]any{},
	}

	// 2. Convert args to user message
	// Args are passed as JSON text (e.g., {"query": "..."})
	// Child agent receives this in a user message and parses it naturally
	userMsg := llm.NewMessage(llm.RoleUser, llm.NewTextPart(string(args)))
	sess.Messages = append(sess.Messages, userMsg)

	// 3. Create invocation metadata
	inv := agent.NewInvocationMetadata(sess, info)

	// 4. Run agent and collect response
	var result string

	for evt, err := range at.agent.Run(ctx, inv) {
		if err != nil {
			return tool.Result{}, fmt.Errorf("agent execution failed: %w", err)
		}

		// Capture last assistant message as result
		if msgEvt, ok := evt.(agent.MessageEvent); ok {
			result = msgEvt.Response.Message.TextContent()
		}
	}

	// 5. Return result
	if result == "" {
		result = "Task completed with no text output."
	}

	output := Result{
		Result: result,
	}

	encoded, err := json.Marshal(output)
	if err != nil {
		return tool.Result{}, err
	}

	return tool.Result{Output: encoded}, nil
}

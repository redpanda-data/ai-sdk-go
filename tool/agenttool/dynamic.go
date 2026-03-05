package agenttool

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// AgentFactory creates a fresh agent from tool call args.
// Returns the agent and the user message to send (extracted from args).
type AgentFactory func(ctx context.Context, args json.RawMessage) (agent.Agent, json.RawMessage, error)

// DynamicAgentTool wraps an AgentFactory as a Tool. On each invocation, the
// LLM picks a model and system prompt from the tool schema, and the factory
// creates a fresh agent configured accordingly.
type DynamicAgentTool struct {
	schema  json.RawMessage
	factory AgentFactory
}

// NewDynamic creates a DynamicAgentTool. The models slice defines the allowed
// model enum in the tool schema. The factory receives the raw args on each
// invocation and must return the configured agent plus the user message.
func NewDynamic(models []string, factory AgentFactory) tool.Tool {
	modelEnum := make([]any, len(models))
	for i, m := range models {
		modelEnum[i] = m
	}

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"system_prompt": map[string]any{
				"type":        "string",
				"description": "The system prompt that defines the subagent's behavior and role.",
			},
			"model": map[string]any{
				"type":        "string",
				"description": "The model to use for this subagent.",
				"enum":        modelEnum,
			},
			"message": map[string]any{
				"type":        "string",
				"description": "The task to send to the subagent.",
			},
		},
		"required": []string{"system_prompt", "model", "message"},
	}

	schemaJSON, _ := json.Marshal(schema)

	return &DynamicAgentTool{
		schema:  schemaJSON,
		factory: factory,
	}
}

// Definition implements tool.Tool.
func (dt *DynamicAgentTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "dynamic_subagent",
		Description: "Create and run a subagent with a chosen model and system prompt.",
		Parameters:  dt.schema,
	}
}

// Execute implements tool.Tool by calling the factory to create a fresh agent,
// then running it with an isolated session.
func (dt *DynamicAgentTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	a, userMsg, err := dt.factory(ctx, args)
	if err != nil {
		return nil, fmt.Errorf("agent factory failed: %w", err)
	}

	return executeAgent(ctx, a, userMsg)
}

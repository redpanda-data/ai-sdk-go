package tool

import (
	"context"
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Tool represents any executable tool - MCP tools, custom functions, external APIs, etc.
// This interface provides the minimum contract that all tools must implement.
//
// Tools should focus on their core functionality and delegate streaming,
// error handling, and lifecycle management to the Registry.
//
// Tool defines the interface for LLM-callable tools that can be executed by AI agents.
type Tool interface {
	// Definition returns the tool's schema for LLM consumption
	// This includes name, description, and parameter JSON schema
	Definition() llm.ToolDefinition

	// Execute performs the tool's main operation synchronously
	// Input and output are JSON for maximum flexibility across tool types
	Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

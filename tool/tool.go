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

	// Execute performs the tool's main operation.
	//
	// Tools always return an Output that is persisted in the conversation history.
	// When the returned Result includes Pending metadata, the runtime pauses and
	// exposes a typed continuation to the caller.
	Execute(ctx context.Context, args json.RawMessage) (Result, error)
}

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

package llm

import "encoding/json"

// Request represents a standardized request to any AI model.
// This structure contains only the universal concepts that work across
// all providers, ensuring maximum compatibility and portability.
type Request struct {
	// Messages contains the conversation history and current input.
	// This is the primary content for the model to process.
	Messages []Message `json:"messages"`

	// Tools defines the functions/tools available for the model to call.
	// Only used if the model supports tool calling (check Capabilities.Tools).
	Tools []ToolDefinition `json:"tools,omitempty"`

	// ToolChoice controls how the model should use available tools.
	// This field is ignored if Tools is empty or the model doesn't support tools.
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// ResponseFormat specifies the desired output structure.
	// Only used if the model supports structured output (check Capabilities.StructuredOutput).
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	// Metadata provides additional context for tracing, logging, and debugging.
	// This data flows through but does not affect model behavior.
	Metadata map[string]string `json:"metadata,omitempty"`
}

// ToolDefinition describes a function/tool available to the model.
// This provides the model with the information needed to decide when and how to call tools.
type ToolDefinition struct {
	// Name is the unique identifier for this tool
	Name string `json:"name"`

	// Description explains what this tool does and when to use it.
	// This helps the model make good decisions about tool usage.
	Description string `json:"description"`

	// Parameters defines the input schema for this tool as a JSON Schema.
	// This tells the model what arguments are expected and their types.
	Parameters json.RawMessage `json:"parameters"`

	// Type specifies the tool category for observability.
	// Used for OpenTelemetry gen_ai.tool.type attribute.
	Type ToolKind `json:"type,omitempty"`
}

// ToolKind classifies where and how a tool executes for observability.
type ToolKind string

// Tool kind constants for OpenTelemetry semantic conventions.
// These describe where/how the tool executes.
const (
	// ToolKindFunction: Local execution - agent generates parameters,
	// local code executes the logic (built-in tools, user-provided functions).
	ToolKindFunction ToolKind = "function"

	// ToolKindExtension: Agent-side remote execution - agent calls
	// external APIs or services (e.g., MCP server tools).
	ToolKindExtension ToolKind = "extension"

	// ToolKindDatastore: Specialized data retrieval tools
	// (e.g., vector databases, knowledge bases).
	ToolKindDatastore ToolKind = "datastore"
)

// ToolChoice controls how the model should interact with available tools.
type ToolChoice struct {
	// Type specifies the tool selection strategy.
	Type ToolChoiceType `json:"type"`

	// Name specifies a particular tool when Type is ToolChoiceSpecific.
	// This forces the model to use only the named tool.
	Name *string `json:"name,omitempty"`
}

// ToolChoiceType selects the tool-invocation strategy a model should use.
type ToolChoiceType string

// Common ToolChoiceType values.
const (
	ToolChoiceAuto     ToolChoiceType = "auto"     // Model decides whether and which tools to use
	ToolChoiceNone     ToolChoiceType = "none"     // Model should not use any tools
	ToolChoiceRequired ToolChoiceType = "required" // Model must use at least one tool
	ToolChoiceSpecific ToolChoiceType = "specific" // Model must use the tool specified in Name
)

// ResponseFormat controls the structure of the model's output.
// This provides three levels of output control, from free-form to strictly constrained.
type ResponseFormat struct {
	// Type specifies the output format constraint level.
	Type ResponseFormatType `json:"type"`

	// JSONSchema provides the schema when Type is ResponseFormatJSONSchema.
	// This constrains the model to generate valid JSON matching the exact schema.
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// ResponseFormatType selects how strictly a model's output is constrained.
type ResponseFormatType string

// ResponseFormatType values provide increasing levels of output structure control:
//
// text: Natural language output with no constraints (default)
// json_object: Valid JSON output with any structure the model chooses
// json_schema: Valid JSON output that must exactly match the provided schema.
const (
	// ResponseFormatText produces natural language output with no structural constraints.
	// This is the default behavior. Use explicitly when you need to override provider
	// defaults or switch dynamically from structured to unstructured output.
	ResponseFormatText ResponseFormatType = "text"

	// ResponseFormatJSONObject guarantees valid JSON output but allows any JSON structure.
	// The model can choose the JSON format and field names. Good for data extraction
	// where you need JSON but don't have rigid structure requirements.
	// Example output: {"name": "John", "skills": ["Go", "Python"]}.
	ResponseFormatJSONObject ResponseFormatType = "json_object"

	// ResponseFormatJSONSchema enforces both valid JSON and exact schema compliance.
	// The model output must exactly match the provided JSONSchema. Use this when
	// you need predictable JSON structure for API integration or data processing.
	// Example: {"sentiment": "positive", "confidence": 0.87} matching your schema.
	ResponseFormatJSONSchema ResponseFormatType = "json_schema"
)

// JSONSchema defines a constraint for structured JSON output.
// The SDK automatically configures providers for maximum schema compliance
// when structured output is requested.
type JSONSchema struct {
	// Name is an identifier for this schema
	Name string `json:"name"`

	// Description explains what this schema represents
	Description string `json:"description,omitempty"`

	// Schema is the JSON Schema definition as a JSON object.
	// This defines the structure the model's output must match.
	Schema json.RawMessage `json:"schema"`
}

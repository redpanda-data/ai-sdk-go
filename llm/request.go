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

	// Options contains provider-specific configuration.
	// Examples: temperature, max_tokens, top_p, etc.
	// Each provider will type-assert this to their own configuration struct.
	Options any `json:"options,omitempty"`

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

	// Metadata provides additional information about the tool.
	// This can include provider-specific configuration or documentation.
	Metadata map[string]any `json:"metadata,omitempty"`

	// Type specifies the tool category for observability.
	// Values: "function" (default), "extension", "datastore"
	// Used for OpenTelemetry gen_ai.tool.type attribute.
	Type string `json:"type,omitempty"`
}

// ToolChoice controls how the model should interact with available tools.
type ToolChoice struct {
	// Type specifies the tool selection strategy.
	// Valid values: "auto", "none", "required", "specific"
	Type string `json:"type"`

	// Name specifies a particular tool when Type is "specific".
	// This forces the model to use only the named tool.
	Name *string `json:"name,omitempty"`
}

// Common ToolChoice values.
const (
	ToolChoiceAuto     = "auto"     // Model decides whether and which tools to use
	ToolChoiceNone     = "none"     // Model should not use any tools
	ToolChoiceRequired = "required" // Model must use at least one tool
	ToolChoiceSpecific = "specific" // Model must use the tool specified in Name
)

// Tool type constants for OpenTelemetry semantic conventions.
// These describe where/how the tool executes.
const (
	// ToolTypeFunction: Local execution - agent generates parameters,
	// local code executes the logic (built-in tools, user-provided functions).
	ToolTypeFunction = "function"

	// ToolTypeExtension: Agent-side remote execution - agent calls
	// external APIs or services (e.g., MCP server tools).
	ToolTypeExtension = "extension"

	// ToolTypeDatastore: Specialized data retrieval tools
	// (e.g., vector databases, knowledge bases).
	ToolTypeDatastore = "datastore"
)

// ResponseFormat controls the structure of the model's output.
// This provides three levels of output control, from free-form to strictly constrained.
type ResponseFormat struct {
	// Type specifies the output format constraint level.
	// Valid values: "text", "json_object", "json_schema"
	Type string `json:"type"`

	// JSONSchema provides the schema when Type is "json_schema".
	// This constrains the model to generate valid JSON matching the exact schema.
	// Only used with ResponseFormatJSONSchema.
	JSONSchema *JSONSchema `json:"json_schema,omitempty"`
}

// ResponseFormat types provide increasing levels of output structure control:
//
// text: Natural language output with no constraints (default)
// json_object: Valid JSON output with any structure the model chooses
// json_schema: Valid JSON output that must exactly match the provided schema.
const (
	// ResponseFormatText produces natural language output with no structural constraints.
	// This is the default behavior. Use explicitly when you need to override provider
	// defaults or switch dynamically from structured to unstructured output.
	ResponseFormatText = "text"

	// ResponseFormatJSONObject guarantees valid JSON output but allows any JSON structure.
	// The model can choose the JSON format and field names. Good for data extraction
	// where you need JSON but don't have rigid structure requirements.
	// Example output: {"name": "John", "skills": ["Go", "Python"]}.
	ResponseFormatJSONObject = "json_object"

	// ResponseFormatJSONSchema enforces both valid JSON and exact schema compliance.
	// The model output must exactly match the provided JSONSchema. Use this when
	// you need predictable JSON structure for API integration or data processing.
	// Example: {"sentiment": "positive", "confidence": 0.87} matching your schema.
	ResponseFormatJSONSchema = "json_schema"
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

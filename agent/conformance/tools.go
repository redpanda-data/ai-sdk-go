package conformance

import (
	"context"
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// CalculatorTool is a standard test tool that adds two numbers.
// Used across all agent conformance tests to ensure consistent behavior.
type CalculatorTool struct{}

// NewCalculatorTool returns a new calculator tool instance.
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{}
}

func (*CalculatorTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "add_numbers",
		Description: "Adds two numbers together and returns the result",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"a": {
					"type": "number",
					"description": "The first number to add"
				},
				"b": {
					"type": "number",
					"description": "The second number to add"
				}
			},
			"required": ["a", "b"]
		}`),
	}
}

func (*CalculatorTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var params struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}

	if err := json.Unmarshal(args, &params); err != nil {
		return nil, err
	}

	result := params.A + params.B

	response := map[string]any{
		"result": result,
		"a":      params.A,
		"b":      params.B,
	}

	return json.Marshal(response)
}

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

func (*CalculatorTool) IsAsynchronous() bool { return false }

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

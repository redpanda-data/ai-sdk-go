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

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// CalculatorTool is a standard test tool that adds two numbers.
// Used across all agent conformance tests to ensure consistent behavior.
type CalculatorTool struct{}

// NewCalculatorTool returns a new calculator tool instance.
func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{}
}

// Name implements tool.Tool.
func (*CalculatorTool) Name() string { return "add_numbers" }

// Description implements tool.Tool.
func (*CalculatorTool) Description() string {
	return "Adds two numbers together and returns the result"
}

// InputSchema implements tool.Tool.
func (*CalculatorTool) InputSchema() json.RawMessage {
	return json.RawMessage(`{
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
    }`)
}

// Execute implements tool.Tool.
func (*CalculatorTool) Execute(_ context.Context, call tool.Call) (tool.Execution, error) {
	var params struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
	}

	if err := json.Unmarshal(call.Args, &params); err != nil {
		return tool.Execution{}, err
	}

	output, err := json.Marshal(map[string]any{
		"result": params.A + params.B,
		"a":      params.A,
		"b":      params.B,
	})
	if err != nil {
		return tool.Execution{}, err
	}

	return tool.Execution{Output: output}, nil
}

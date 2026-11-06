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

//nolint:exhaustive // Inventory admits only the three direct Fireworks protocols before dispatch.
package fireworks_test

import (
	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func contractToolConfig(c contractCase, request map[string]any) {
	parameters := map[string]any{"type": "object", "properties": map[string]any{"key": map[string]any{"type": "string", "enum": []string{"x"}}}, "required": []string{"key"}, "additionalProperties": false}

	switch c.API {
	case catalog.WireProtocolAnthropicMessages:
		request[contractTools] = []any{map[string]any{"name": "lookup", "description": "Look up a secret word by key.", "input_schema": parameters}}
		request["tool_choice"] = map[string]any{"type": "tool", "name": "lookup"}
	case catalog.WireProtocolOpenAIChatCompletions:
		request[contractTools] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "description": "Look up a secret word by key.", "parameters": parameters, "strict": true}}}
		request["tool_choice"] = map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}
	case catalog.WireProtocolOpenAIResponses:
		request[contractTools] = []any{map[string]any{"type": "function", "name": "lookup", "description": "Look up a secret word by key.", "parameters": parameters, "strict": true}}
		request["tool_choice"] = map[string]any{"type": "function", "name": "lookup"}
	}
}

func contractFormatConfig(c contractCase, request map[string]any) {
	shape := c
	if c.Feature == contractInvalidFormat {
		shape.Feature = contractJSON
	}

	if c.Feature == contractInvalidSchema {
		shape.Feature = contractStructured
	}

	if shape.Feature == contractJSON || shape.Feature == contractStructured {
		contractOutputConfig(shape, request)
	}
}

func contractOutputConfig(c contractCase, request map[string]any) {
	format := map[string]any{"type": "json_object"}

	if c.Feature == contractStructured {
		schema := map[string]any{"type": "object", "properties": map[string]any{"ok": map[string]any{"type": "boolean", "enum": []bool{true}}}, "required": []string{"ok"}, "additionalProperties": false}
		if c.Options.FormatProof {
			schema = map[string]any{"type": "object", "properties": map[string]any{"gate": map[string]any{"type": "string", "enum": []string{contractSchemaMarker}}, "count": map[string]any{"type": "integer", "enum": []int{73}}}, "required": []string{"gate", "count"}, "additionalProperties": false}
		}

		switch c.API {
		case catalog.WireProtocolOpenAIResponses:
			format = map[string]any{"type": "json_schema", "name": "contract_output", "strict": true, "schema": schema}
		case catalog.WireProtocolAnthropicMessages:
			format = map[string]any{"type": "json_schema", "schema": schema}
		case catalog.WireProtocolOpenAIChatCompletions:
			format = map[string]any{"type": "json_schema", "json_schema": map[string]any{"name": "contract_output", "strict": true, "schema": schema}}
		}
	}

	switch c.API {
	case catalog.WireProtocolOpenAIResponses:
		request["text"] = map[string]any{"format": format}
	case catalog.WireProtocolAnthropicMessages:
		request["output_config"] = map[string]any{"format": format}
	case catalog.WireProtocolOpenAIChatCompletions:
		request["response_format"] = format
	}
}

func contractEffort(c contractCase) string {
	if c.Effort != "" {
		return c.Effort
	}

	for _, o := range fireworks.Catalog().All() {
		if o.ID != c.Model {
			continue
		}

		support, _ := o.SupportForWireAPI(catalog.WireAPI{Protocol: c.API, Transport: c.Transport})
		if len(support.Reasoning.Efforts) == 0 {
			return ""
		}

		if c.Feature != contractReasoningHistory {
			return string(support.Reasoning.Efforts[0])
		}

		for _, effort := range support.Reasoning.Efforts {
			if effort != contractEffortNone {
				return string(effort)
			}
		}

		return string(support.Reasoning.Efforts[0])
	}

	return ""
}

func contractEffortConfig(c contractCase, request map[string]any) {
	effort := contractEffort(c)
	if effort == "" {
		return
	}

	switch c.API {
	case catalog.WireProtocolOpenAIResponses:
		request[contractReasoning] = map[string]any{"effort": effort}
	case catalog.WireProtocolAnthropicMessages:
		output := contractObject(request["output_config"])
		if output == nil {
			output = map[string]any{}
			request["output_config"] = output
		}

		if effort == contractEffortNone {
			request["thinking"] = map[string]any{"type": "disabled"}
		} else {
			output["effort"] = effort
		}
	case catalog.WireProtocolOpenAIChatCompletions:
		request["reasoning_effort"] = effort
	}
}

func contractMaxTokens(c contractCase) int {
	maxTokens := 512
	if c.Feature == contractReasoning || c.Feature == contractReasoningHistory {
		maxTokens = 1024
	}

	if c.Feature == contractIncomplete {
		maxTokens = 1
	}

	if c.Feature == contractCancel {
		maxTokens = 16
	}

	if c.Options.MaxOutputTokens > 0 {
		maxTokens = c.Options.MaxOutputTokens
	}

	return maxTokens
}

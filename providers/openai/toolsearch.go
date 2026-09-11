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

package openai

import (
	"encoding/json"

	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	outputTypeToolSearchCall   = "tool_search_call"
	outputTypeToolSearchOutput = "tool_search_output"
)

// Reuse the ordinary schema mapper; native loading only changes their exposure.
func nativeSearchTools(defs []llm.ToolDefinition, mapped []responses.ToolUnionParam) []responses.ToolUnionParam {
	tools := make([]responses.ToolUnionParam, 0, len(mapped)+1)
	groups := make(map[string]*responses.NamespaceToolParam)

	for i, def := range defs {
		fn := mapped[i].OfFunction

		fn.DeferLoading = param.NewOpt(def.Deferred)
		if def.Group.Name == "" {
			tools = append(tools, mapped[i])
			continue
		}

		group := groups[def.Group.Name]
		if group == nil {
			group = &responses.NamespaceToolParam{Name: def.Group.Name, Description: def.Group.Description}
			groups[def.Group.Name] = group
			tools = append(tools, responses.ToolUnionParam{OfNamespace: group})
		}

		group.Tools = append(group.Tools, responses.NamespaceToolToolUnionParam{OfFunction: &responses.NamespaceToolToolFunctionParam{
			Name: fn.Name, Description: fn.Description, Parameters: fn.Parameters, Strict: fn.Strict, DeferLoading: fn.DeferLoading,
		}})
	}

	return append(tools, responses.ToolUnionParam{OfToolSearch: &responses.ToolSearchToolParam{}})
}

func mapFunctionCall(fc responses.ResponseFunctionToolCall) *llm.ToolRequestPart {
	part := llm.NewToolRequestPart(fc.CallID, fc.Name, normalizeToolArguments(fc.Arguments))
	if fc.Namespace != "" {
		part.Metadata = map[string]any{"openai_namespace": fc.Namespace}
	}

	return part
}

func mapToolSearch(item responses.ResponseOutputItemUnion) *llm.ToolSearchPart {
	part := &llm.ToolSearchPart{Provider: providerName, Data: json.RawMessage(item.RawJSON())}
	if item.Type == outputTypeToolSearchOutput {
		for _, tool := range item.AsToolSearchOutput().Tools {
			switch tool.Type {
			case "function":
				part.Tools = append(part.Tools, tool.Name)
			case "namespace":
				for _, function := range tool.AsNamespace().Tools {
					part.Tools = append(part.Tools, function.Name)
				}
			}
		}
	}

	return part
}

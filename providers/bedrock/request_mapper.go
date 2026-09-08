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

package bedrock

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/redpanda-data/ai-sdk-go/internal/jsonschema"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// RequestMapper handles conversion from llm.Request to Bedrock Converse API format.
type RequestMapper struct {
	config *Config
}

// NewRequestMapper creates a new RequestMapper with the given configuration.
func NewRequestMapper(config *Config) *RequestMapper {
	return &RequestMapper{config: config}
}

// ToConverseInput converts an llm.Request to a bedrockruntime.ConverseInput.
func (rm *RequestMapper) ToConverseInput(req *llm.Request) (*bedrockruntime.ConverseInput, error) {
	p, err := rm.buildConverseParams(req)
	if err != nil {
		return nil, err
	}

	return &bedrockruntime.ConverseInput{
		ModelId:                      aws.String(rm.config.APIModelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.infConfig,
		AdditionalModelRequestFields: p.thinking,
		ToolConfig:                   p.tools,
		OutputConfig:                 p.outputConfig,
	}, nil
}

// ToConverseStreamInput converts an llm.Request to a bedrockruntime.ConverseStreamInput.
func (rm *RequestMapper) ToConverseStreamInput(req *llm.Request) (*bedrockruntime.ConverseStreamInput, error) {
	p, err := rm.buildConverseParams(req)
	if err != nil {
		return nil, err
	}

	return &bedrockruntime.ConverseStreamInput{
		ModelId:                      aws.String(rm.config.APIModelID),
		Messages:                     p.messages,
		System:                       p.system,
		InferenceConfig:              p.infConfig,
		AdditionalModelRequestFields: p.thinking,
		ToolConfig:                   p.tools,
		OutputConfig:                 p.outputConfig,
	}, nil
}

// converseParams holds the resolved fields shared by Converse and ConverseStream.
type converseParams struct {
	messages  []types.Message
	system    []types.SystemContentBlock
	infConfig *types.InferenceConfiguration
	thinking  document.Interface
	tools     *types.ToolConfiguration
	// outputConfig carries structured-output constraints (Converse
	// outputConfig.textFormat).
	outputConfig *types.OutputConfig
}

// buildConverseParams resolves all common request parameters once.
func (rm *RequestMapper) buildConverseParams(req *llm.Request) (*converseParams, error) {
	messages, system, err := rm.mapMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	p := &converseParams{
		messages:  messages,
		system:    system,
		infConfig: rm.buildInferenceConfig(),
	}

	if rm.config.EnableThinking || rm.config.ReasoningEffort != nil {
		p.thinking = rm.buildThinkingFields()
	}

	if len(req.Tools) > 0 {
		toolConfig, err := rm.mapToolConfig(req.Tools, req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		p.tools = toolConfig
	}

	// Bedrock's Converse API rejects a request whose message history contains
	// toolUse/toolResult blocks unless toolConfig is set on that same request,
	// even when the current turn has nothing to offer — e.g. a tool used
	// earlier in the conversation has since been removed from the agent.
	// Mirror langchain-aws's fix for the same constraint: flatten the stale
	// blocks to text instead of failing the whole request. Reconstructing a
	// synthetic toolConfig from the tool names in history was considered and
	// rejected: it would keep the blocks structured, but it re-advertises a
	// tool the agent no longer has, inviting the model to call it again.
	// https://github.com/langchain-ai/langchain-aws/pull/595
	if p.tools == nil && hasToolUseOrResultBlocks(p.messages) {
		p.messages = convertToolBlocksToText(p.messages)
	}

	if req.ResponseFormat != nil {
		if err := applyResponseFormat(p, req.ResponseFormat); err != nil {
			return nil, fmt.Errorf("%w: response format mapping failed: %w", llm.ErrRequestMapping, err)
		}
	}

	return p, nil
}

// applyResponseFormat sets Converse's outputConfig.textFormat from a
// unified ResponseFormat. A text format leaves p.outputConfig nil.
//
// Converse only accepts a JSON schema, so ResponseFormatJSONObject is
// rejected rather than silently downgraded to unconstrained text.
func applyResponseFormat(p *converseParams, format *llm.ResponseFormat) error {
	switch format.Type {
	case llm.ResponseFormatText, "":
		return nil

	case llm.ResponseFormatJSONSchema:
		if format.JSONSchema == nil {
			return errors.New("JSONSchema is required when Type is json_schema")
		}

		var schema map[string]any
		if err := json.Unmarshal(format.JSONSchema.Schema, &schema); err != nil {
			return fmt.Errorf("invalid JSON schema: %w", err)
		}

		// Converse enforces the same structured-output subset as the
		// Anthropic API (additionalProperties: false on every object, no
		// constraint keywords), so adapt before sending.
		jsonschema.AdaptForStructuredOutput(schema)

		adapted, err := json.Marshal(schema)
		if err != nil {
			return fmt.Errorf("invalid JSON schema: %w", err)
		}

		p.outputConfig = &types.OutputConfig{
			TextFormat: &types.OutputFormat{
				Type: types.OutputFormatTypeJsonSchema,
				Structure: &types.OutputFormatStructureMemberJsonSchema{
					Value: types.JsonSchemaDefinition{
						Schema: aws.String(string(adapted)),
					},
				},
			},
		}

		return nil

	case llm.ResponseFormatJSONObject:
		return fmt.Errorf("%w: Bedrock requires a schema for JSON output: use %q with a JSONSchema",
			llm.ErrUnsupportedFeature, llm.ResponseFormatJSONSchema)

	default:
		return fmt.Errorf("unsupported response format type: %s", format.Type)
	}
}

// buildInferenceConfig creates the InferenceConfiguration from config options.
func (rm *RequestMapper) buildInferenceConfig() *types.InferenceConfiguration {
	var cfg types.InferenceConfiguration
	hasConfig := false

	if rm.config.Temperature != nil {
		v := float32(*rm.config.Temperature)
		cfg.Temperature = &v
		hasConfig = true
	}

	if rm.config.TopP != nil {
		v := float32(*rm.config.TopP)
		cfg.TopP = &v
		hasConfig = true
	}

	if rm.config.MaxTokens != nil {
		cfg.MaxTokens = rm.config.MaxTokens
		hasConfig = true
	}

	if len(rm.config.Stop) > 0 {
		cfg.StopSequences = rm.config.Stop
		hasConfig = true
	}

	if !hasConfig {
		return nil
	}

	return &cfg
}

// buildThinkingFields returns the additionalModelRequestFields document for
// the configured thinking mode.
func (rm *RequestMapper) buildThinkingFields() document.Interface {
	// A manual budget (WithThinking) requests classic extended thinking.
	if rm.config.BudgetTokens > 0 {
		return document.NewLazyDocument(map[string]any{
			"thinking": map[string]any{
				"type":          "enabled",
				"budget_tokens": rm.config.BudgetTokens,
			},
		})
	}

	// Otherwise thinking is adaptive: the model decides how long to think,
	// optionally biased by the reasoning effort (WithReasoningEffort).
	fields := map[string]any{
		"thinking": map[string]any{
			"type": "adaptive",
		},
	}
	if rm.config.ReasoningEffort != nil {
		fields["output_config"] = map[string]any{
			"effort": *rm.config.ReasoningEffort,
		}
	}

	return document.NewLazyDocument(fields)
}

// mapMessages converts llm.Messages to Bedrock Converse types, separating system messages.
// TODO: only text/tool/reasoning parts are mapped. The catalog
// advertises image (and on some models audio/video/document) input
// because the models accept it; llm.Part has no binary part yet.
func (rm *RequestMapper) mapMessages(messages []llm.Message) ([]types.Message, []types.SystemContentBlock, error) {
	var apiMessages []types.Message
	var system []types.SystemContentBlock

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleSystem:
			for _, part := range msg.Content {
				if tp, ok := part.(*llm.TextPart); ok {
					system = append(system, &types.SystemContentBlockMemberText{
						Value: tp.Text,
					})
				}
			}

		case llm.RoleUser:
			apiMsg, err := rm.mapUserMessage(msg)
			if err != nil {
				return nil, nil, err
			}

			apiMessages = append(apiMessages, apiMsg)

		case llm.RoleAssistant:
			apiMsg, err := rm.mapAssistantMessage(msg)
			if err != nil {
				return nil, nil, err
			}

			apiMessages = append(apiMessages, apiMsg)

		default:
			return nil, nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	// If caching is enabled, insert CachePointBlocks after the last system block
	// and after the last content block of the last message.
	if rm.config.EnableCaching {
		cachePoint := types.CachePointBlock{Type: types.CachePointTypeDefault}
		if len(system) > 0 {
			system = append(system, &types.SystemContentBlockMemberCachePoint{Value: cachePoint})
		}

		if len(apiMessages) > 0 {
			lastMsg := &apiMessages[len(apiMessages)-1]
			lastMsg.Content = append(lastMsg.Content, &types.ContentBlockMemberCachePoint{Value: cachePoint})
		}
	}

	return apiMessages, system, nil
}

// mapUserMessage converts a user message to Bedrock Converse format.
func (rm *RequestMapper) mapUserMessage(msg llm.Message) (types.Message, error) {
	apiMsg := types.Message{
		Role: types.ConversationRoleUser,
	}

	for _, part := range msg.Content {
		switch p := part.(type) {
		case *llm.TextPart:
			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberText{
				Value: p.Text,
			})

		case *llm.ToolResponsePart:
			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberToolResult{
				Value: rm.mapToolResultBlock(p),
			})

		default:
			return apiMsg, fmt.Errorf("unsupported part type in user message: %T", part)
		}
	}

	return apiMsg, nil
}

// mapAssistantMessage converts an assistant message to Bedrock Converse format.
func (rm *RequestMapper) mapAssistantMessage(msg llm.Message) (types.Message, error) {
	apiMsg := types.Message{
		Role: types.ConversationRoleAssistant,
	}

	for _, part := range msg.Content {
		switch p := part.(type) {
		case *llm.TextPart:
			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberText{
				Value: p.Text,
			})

		case *llm.ToolRequestPart:
			// Parse arguments to a generic map for document.Interface
			var input map[string]any
			if err := json.Unmarshal(p.Arguments, &input); err != nil {
				return apiMsg, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: aws.String(p.ID),
					Name:      aws.String(p.Name),
					Input:     document.NewLazyDocument(input),
				},
			})

		case *llm.ReasoningPart:
			// Pass reasoning traces back as reasoning content blocks
			if p.Text != "" || p.Signature != "" {
				apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text:      aws.String(p.Text),
							Signature: aws.String(p.Signature),
						},
					},
				})
			}

		default:
			return apiMsg, fmt.Errorf("unsupported part type in assistant message: %T", part)
		}
	}

	// A truncated turn can reach us with no content parts (e.g. stop_reason=
	// max_tokens where the model's only block was a partial tool_use the
	// streaming finalizer dropped). Bedrock's Converse API — Anthropic-shaped —
	// rejects an assistant message with a content-less body, so replaying such a
	// persisted turn fails the whole request. Substitute a single minimal text
	// block, matching the anthropic mapper. This runs before the cache-point
	// insertion stage in mapMessages, so an empty assistant turn can never
	// become a cachePoint-only content block (which is still invalid).
	//
	// The placeholder must be non-empty and non-whitespace: a whitespace-only or
	// trailing-whitespace final assistant turn can be rejected or stripped back
	// to empty, so a non-whitespace token is robust in any position.
	if len(apiMsg.Content) == 0 {
		apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberText{
			Value: "(truncated)",
		})
	}

	return apiMsg, nil
}

// mapToolResultBlock converts a tool response to a Bedrock ToolResultBlock.
func (rm *RequestMapper) mapToolResultBlock(resp *llm.ToolResponsePart) types.ToolResultBlock {
	block := types.ToolResultBlock{
		ToolUseId: aws.String(resp.ID),
	}

	if resp.IsError {
		block.Status = types.ToolResultStatusError
	} else {
		block.Status = types.ToolResultStatusSuccess
	}

	block.Content = []types.ToolResultContentBlock{
		&types.ToolResultContentBlockMemberText{Value: string(resp.Result)},
	}

	return block
}

// mapToolConfig converts tool definitions and choice to Bedrock ToolConfiguration.
func (rm *RequestMapper) mapToolConfig(tools []llm.ToolDefinition, choice *llm.ToolChoice) (*types.ToolConfiguration, error) {
	apiTools := make([]types.Tool, 0, len(tools))

	for _, tool := range tools {
		var schemaMap map[string]any
		if err := json.Unmarshal(tool.Parameters, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse tool schema for %s: %w", tool.Name, err)
		}

		apiTools = append(apiTools, &types.ToolMemberToolSpec{
			Value: types.ToolSpecification{
				Name:        aws.String(tool.Name),
				Description: aws.String(tool.Description),
				InputSchema: &types.ToolInputSchemaMemberJson{
					Value: document.NewLazyDocument(schemaMap),
				},
			},
		})
	}

	// Cache the tool definitions.
	//
	// Bedrock's toolConfig is a separate top-level field from system and messages,
	// so a CachePointBlock in either of those does not cover it: without a marker
	// here the whole tool schema is re-sent uncached on every single call. For a
	// tool-heavy server that is the largest STABLE block in the request — measured
	// against a ServiceNow MCP, six generated tools are ~6 KB of JSON Schema, and
	// an agent making nine calls paid for all of it nine times.
	//
	// The marker goes last, after every tool, because a cache point covers the
	// prefix up to itself and the tool set is fixed for the life of the agent.
	if rm.config.EnableCaching && len(apiTools) > 0 {
		apiTools = append(apiTools, &types.ToolMemberCachePoint{
			Value: types.CachePointBlock{Type: types.CachePointTypeDefault},
		})
	}

	config := &types.ToolConfiguration{
		Tools: apiTools,
	}

	if choice != nil {
		tc, err := rm.mapToolChoice(choice)
		if err != nil {
			return nil, fmt.Errorf("tool choice mapping failed: %w", err)
		}

		config.ToolChoice = tc
	}

	return config, nil
}

// mapToolChoice converts llm.ToolChoice to Bedrock ToolChoice.
func (rm *RequestMapper) mapToolChoice(choice *llm.ToolChoice) (types.ToolChoice, error) {
	switch choice.Type {
	case llm.ToolChoiceAuto:
		return &types.ToolChoiceMemberAuto{
			Value: types.AutoToolChoice{},
		}, nil

	case llm.ToolChoiceRequired:
		return &types.ToolChoiceMemberAny{
			Value: types.AnyToolChoice{},
		}, nil

	case llm.ToolChoiceNone:
		return nil, errors.New("bedrock does not support tool_choice=none; omit tools from the request instead")

	case llm.ToolChoiceSpecific:
		if choice.Name == nil || *choice.Name == "" {
			return nil, errors.New("tool name required for ToolChoiceSpecific")
		}

		return &types.ToolChoiceMemberTool{
			Value: types.SpecificToolChoice{
				Name: choice.Name,
			},
		}, nil

	default:
		return nil, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}
}

// hasToolUseOrResultBlocks reports whether any message contains a ToolUse or
// ToolResult content block.
func hasToolUseOrResultBlocks(messages []types.Message) bool {
	for _, msg := range messages {
		for _, block := range msg.Content {
			switch block.(type) {
			case *types.ContentBlockMemberToolUse, *types.ContentBlockMemberToolResult:
				return true
			}
		}
	}

	return false
}

// convertToolBlocksToText flattens ToolUse/ToolResult content blocks to plain
// text, preserving only the information needed to keep the transcript
// readable.
func convertToolBlocksToText(messages []types.Message) []types.Message {
	converted := make([]types.Message, len(messages))

	for i, msg := range messages {
		out := types.Message{Role: msg.Role}

		for _, block := range msg.Content {
			switch b := block.(type) {
			case *types.ContentBlockMemberToolUse:
				out.Content = append(out.Content, &types.ContentBlockMemberText{
					Value: toolUseText(b.Value),
				})

			case *types.ContentBlockMemberToolResult:
				out.Content = append(out.Content, &types.ContentBlockMemberText{
					Value: toolResultText(b.Value),
				})

			default:
				out.Content = append(out.Content, block)
			}
		}

		converted[i] = out
	}

	return converted
}

// toolUseText renders a ToolUseBlock as "[Called {name} with parameters: {json}]",
// or "[Called {name}]" when there are no parameters.
func toolUseText(tu types.ToolUseBlock) string {
	name := "function"
	if tu.Name != nil {
		name = *tu.Name
	}

	if raw := marshalDocument(tu.Input); len(raw) > 0 && string(raw) != "{}" && string(raw) != "null" {
		return fmt.Sprintf("[Called %s with parameters: %s]", name, raw)
	}

	return fmt.Sprintf("[Called %s]", name)
}

// toolResultText renders a ToolResultBlock as "[Tool output: {content}]", or
// "[Tool error: {content}]" for a failed call, with "(empty)" standing in for
// a void result. It must never return "": an empty return would drop the
// block entirely, and a ToolResult is often the only content in its message —
// mapUserMessage never mixes it with anything mapAssistantMessage-side would
// need — so a dropped block can leave the message with no content at all,
// which Bedrock rejects exactly like it rejects the empty-assistant-turn case
// this file already guards elsewhere.
//
// Only Text and Json content blocks are handled: mapToolResultBlock, the only
// producer of ToolResultBlock in this package, never emits anything else.
func toolResultText(tr types.ToolResultBlock) string {
	var sb strings.Builder

	for _, c := range tr.Content {
		switch v := c.(type) {
		case *types.ToolResultContentBlockMemberText:
			sb.WriteString(v.Value)

		case *types.ToolResultContentBlockMemberJson:
			sb.Write(marshalDocument(v.Value))
		}
	}

	content := sb.String()
	if strings.TrimSpace(content) == "" {
		content = "(empty)"
	}

	if tr.Status == types.ToolResultStatusError {
		return fmt.Sprintf("[Tool error: %s]", content)
	}

	return fmt.Sprintf("[Tool output: %s]", content)
}

// marshalDocument returns nil, rather than an error, on a marshal failure:
// callers treat that as "no parameters"/"no content", the same fallback used
// when a document is absent entirely.
func marshalDocument(doc document.Interface) []byte {
	if doc == nil {
		return nil
	}

	raw, err := doc.MarshalSmithyDocument()
	if err != nil {
		return nil
	}

	return raw
}

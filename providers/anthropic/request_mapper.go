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

package anthropic

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// RequestMapper handles conversion from unified Request to Anthropic API format.
type RequestMapper struct {
	config       *Config
	schemaMapper *SchemaMapper
}

// NewRequestMapper creates a new RequestMapper with the given configuration.
func NewRequestMapper(config *Config) *RequestMapper {
	return &RequestMapper{
		config:       config,
		schemaMapper: NewSchemaMapper(),
	}
}

// ToProvider converts our unified Request to Anthropic Beta Messages API format.
func (rm *RequestMapper) ToProvider(req *llm.Request) (anthropic.BetaMessageNewParams, error) {
	// Determine which model name to use (custom override if set, otherwise the configured model)
	modelName := rm.config.ModelName
	if rm.config.CustomModelName != "" {
		modelName = rm.config.CustomModelName
	}

	// Create base request for Beta Messages API
	apiReq := anthropic.BetaMessageNewParams{
		Model: anthropic.Model(modelName),
	}

	// MaxTokens is set by provider config (required by Anthropic API)
	apiReq.MaxTokens = int64(rm.config.MaxTokens)

	// Map messages and system prompt
	messages, systemPrompt, err := rm.mapMessages(req.Messages)
	if err != nil {
		return apiReq, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	apiReq.Messages = messages
	if len(systemPrompt) > 0 {
		apiReq.System = systemPrompt
	}

	// Apply configuration parameters
	if rm.config.Temperature != nil {
		apiReq.Temperature = param.NewOpt(*rm.config.Temperature)
	}

	if rm.config.TopP != nil {
		apiReq.TopP = param.NewOpt(*rm.config.TopP)
	}

	if rm.config.TopK != nil {
		apiReq.TopK = param.NewOpt(int64(*rm.config.TopK))
	}

	if len(rm.config.Stop) > 0 {
		apiReq.StopSequences = rm.config.Stop
	}

	// Apply tool definitions if provided
	// Note: Anthropic doesn't support response_format (JSON mode or structured output)
	// Users should use tool calling directly for structured output
	if len(req.Tools) > 0 {
		tools, err := rm.mapToolDefinitions(req.Tools)
		if err != nil {
			return apiReq, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		apiReq.Tools = tools

		// Apply tool choice if specified
		if req.ToolChoice != nil {
			toolChoice, err := rm.mapToolChoice(req.ToolChoice)
			if err != nil {
				return apiReq, fmt.Errorf("%w: tool choice mapping failed: %w", llm.ErrRequestMapping, err)
			}

			apiReq.ToolChoice = toolChoice
		}
	}

	// Enable extended thinking if configured
	if rm.config.EnableThinking {
		switch {
		case rm.config.ThinkingBudget != nil:
			// Explicit budget: manual thinking with user-specified tokens
			apiReq.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(*rm.config.ThinkingBudget)
		case rm.config.AdaptiveThinking:
			// Model supports adaptive thinking: let the API decide the budget
			apiReq.Thinking = anthropic.BetaThinkingConfigParamUnion{
				OfAdaptive: &anthropic.BetaThinkingConfigAdaptiveParam{},
			}
		default:
			// Legacy fallback: 25% of max tokens with minimum of 1024
			budgetTokens := max(int64(rm.config.MaxTokens/4), 1024)
			apiReq.Thinking = anthropic.BetaThinkingConfigParamOfEnabled(budgetTokens)
		}
	}

	// Apply effort if configured
	if rm.config.Effort != nil {
		apiReq.OutputConfig = anthropic.BetaOutputConfigParam{
			Effort: anthropic.BetaOutputConfigEffort(*rm.config.Effort),
		}
	}

	// Apply speed if configured
	if rm.config.Speed != nil {
		apiReq.Speed = anthropic.BetaMessageNewParamsSpeed(*rm.config.Speed)
	}

	return apiReq, nil
}

// mapMessages converts our unified messages to Anthropic format.
// It separates system messages from user/assistant messages.
func (rm *RequestMapper) mapMessages(messages []llm.Message) ([]anthropic.BetaMessageParam, []anthropic.BetaTextBlockParam, error) {
	apiMessages := make([]anthropic.BetaMessageParam, 0, len(messages))

	var systemBlocks []anthropic.BetaTextBlockParam

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleSystem:
			// System messages go into the separate system parameter
			for _, part := range msg.Content {
				if tp, ok := part.(*llm.TextPart); ok {
					systemBlocks = append(systemBlocks, anthropic.BetaTextBlockParam{
						Type: constant.Text(""),
						Text: tp.Text,
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

	// If caching is enabled, set cache_control on system blocks and last message
	if rm.config.EnableCaching {
		// Mark the last system block for caching if we have any
		if len(systemBlocks) > 0 {
			lastIdx := len(systemBlocks) - 1
			block := systemBlocks[lastIdx]
			block.CacheControl = anthropic.NewBetaCacheControlEphemeralParam()
			systemBlocks[lastIdx] = block
		}

		// Also mark the last text block of the last message
		if len(apiMessages) > 0 {
			lastMsg := &apiMessages[len(apiMessages)-1]
			for i := len(lastMsg.Content) - 1; i >= 0; i-- {
				if lastMsg.Content[i].OfText != nil {
					content := lastMsg.Content[i]
					content.OfText.CacheControl = anthropic.NewBetaCacheControlEphemeralParam()
					lastMsg.Content[i] = content

					break
				}
			}
		}
	}

	return apiMessages, systemBlocks, nil
}

// mapUserMessage converts a user message to Anthropic format.
func (rm *RequestMapper) mapUserMessage(msg llm.Message) (anthropic.BetaMessageParam, error) {
	apiMsg := anthropic.BetaMessageParam{
		Role: anthropic.BetaMessageParamRoleUser,
	}

	for _, part := range msg.Content {
		switch p := part.(type) {
		case *llm.TextPart:
			apiMsg.Content = append(apiMsg.Content, anthropic.BetaContentBlockParamUnion{
				OfText: &anthropic.BetaTextBlockParam{
					Type: constant.Text(""),
					Text: p.Text,
				},
			})

		case *llm.ToolResponsePart:
			block, err := rm.mapToolResultBlock(p)
			if err != nil {
				return apiMsg, err
			}

			apiMsg.Content = append(apiMsg.Content, block)

		default:
			return apiMsg, fmt.Errorf("unsupported part type in user message: %T", part)
		}
	}

	return apiMsg, nil
}

// mapAssistantMessage converts an assistant message to Anthropic format.
func (rm *RequestMapper) mapAssistantMessage(msg llm.Message) (anthropic.BetaMessageParam, error) {
	apiMsg := anthropic.BetaMessageParam{
		Role: anthropic.BetaMessageParamRoleAssistant,
	}

	for _, part := range msg.Content {
		switch p := part.(type) {
		case *llm.TextPart:
			apiMsg.Content = append(apiMsg.Content, anthropic.BetaContentBlockParamUnion{
				OfText: &anthropic.BetaTextBlockParam{
					Type: constant.Text(""),
					Text: p.Text,
				},
			})

		case *llm.ToolRequestPart:
			// Parse arguments as map for input field
			var input map[string]any
			if err := json.Unmarshal(p.Arguments, &input); err != nil {
				return apiMsg, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			apiMsg.Content = append(apiMsg.Content, anthropic.BetaContentBlockParamUnion{
				OfToolUse: &anthropic.BetaToolUseBlockParam{
					Type:  constant.ToolUse(""),
					ID:    p.ID,
					Name:  p.Name,
					Input: input,
				},
			})

		case *llm.ReasoningPart:
			// Map reasoning to thinking block
			apiMsg.Content = append(apiMsg.Content, anthropic.BetaContentBlockParamUnion{
				OfThinking: &anthropic.BetaThinkingBlockParam{
					Type:      constant.Thinking(""),
					Thinking:  p.Text,
					Signature: p.Signature,
				},
			})

		default:
			return apiMsg, fmt.Errorf("unsupported part type in assistant message: %T", part)
		}
	}

	// A truncated turn can reach us with no content parts: e.g. stop_reason=
	// max_tokens where the model's only block was a partial tool_use whose
	// accumulated JSON args didn't parse, so the streaming finalizer dropped it
	// (see providers/anthropic/model.go). Anthropic's Messages API rejects an
	// assistant message with an empty content array
	// ("messages.N.content: Field required"), so replaying such a persisted turn
	// 400s the whole request. Substitute a single minimal text block rather than
	// dropping the message: dropping would leave the surrounding user turns
	// adjacent and break Anthropic's required user/assistant alternation on
	// replay. A truncated turn carries no recoverable content, so a placeholder
	// loses nothing.
	//
	// The placeholder must be non-empty (Anthropic rejects empty text blocks)
	// AND non-whitespace: Anthropic rejects a final assistant turn that ends in
	// trailing whitespace, and some API/SDK versions strip a whitespace-only
	// text block back to empty. A non-whitespace token is therefore robust
	// whether the repaired turn is mid-conversation or the last message.
	if len(apiMsg.Content) == 0 {
		apiMsg.Content = append(apiMsg.Content, anthropic.BetaContentBlockParamUnion{
			OfText: &anthropic.BetaTextBlockParam{
				Type: constant.Text(""),
				Text: "(truncated)",
			},
		})
	}

	return apiMsg, nil
}

// mapToolResultBlock converts a tool response to Anthropic's tool_result format.
func (rm *RequestMapper) mapToolResultBlock(part *llm.ToolResponsePart) (anthropic.BetaContentBlockParamUnion, error) {
	if part == nil {
		return anthropic.BetaContentBlockParamUnion{}, errors.New("nil ToolResponsePart")
	}

	content := []anthropic.BetaToolResultBlockParamContentUnion{
		{OfText: &anthropic.BetaTextBlockParam{
			Type: constant.Text(""),
			Text: string(part.Result),
		}},
	}

	return anthropic.BetaContentBlockParamUnion{
		OfToolResult: &anthropic.BetaToolResultBlockParam{
			Type:      constant.ToolResult(""),
			ToolUseID: part.ID,
			Content:   content,
			IsError:   param.NewOpt(part.IsError),
		},
	}, nil
}

// mapToolDefinitions converts our tool definitions to Anthropic format.
func (rm *RequestMapper) mapToolDefinitions(tools []llm.ToolDefinition) ([]anthropic.BetaToolUnionParam, error) {
	apiTools := make([]anthropic.BetaToolUnionParam, 0, len(tools))

	for _, tool := range tools {
		// Parse the JSON schema
		var schemaMap map[string]any
		if err := json.Unmarshal(tool.Parameters, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse tool schema for %s: %w", tool.Name, err)
		}

		// Adapt the schema for Anthropic
		schema := rm.schemaMapper.AdaptSchemaForAnthropic(schemaMap)

		// Use Anthropic SDK helper which properly transforms and filters the schema
		inputSchema := anthropic.BetaToolInputSchema(schema)

		apiTool := anthropic.BetaToolUnionParam{
			OfTool: &anthropic.BetaToolParam{
				Name:        tool.Name,
				Description: param.NewOpt(tool.Description),
				InputSchema: inputSchema,
			},
		}

		apiTools = append(apiTools, apiTool)
	}

	return apiTools, nil
}

// mapToolChoice converts our tool choice to Anthropic format.
func (rm *RequestMapper) mapToolChoice(choice *llm.ToolChoice) (anthropic.BetaToolChoiceUnionParam, error) {
	switch choice.Type {
	case llm.ToolChoiceAuto:
		return anthropic.BetaToolChoiceUnionParam{
			OfAuto: &anthropic.BetaToolChoiceAutoParam{
				Type:                   constant.Auto(""),
				DisableParallelToolUse: param.NewOpt(false),
			},
		}, nil

	case llm.ToolChoiceRequired:
		// Map "required" to "any" in Anthropic
		return anthropic.BetaToolChoiceUnionParam{
			OfAny: &anthropic.BetaToolChoiceAnyParam{
				Type:                   constant.Any(""),
				DisableParallelToolUse: param.NewOpt(false),
			},
		}, nil

	case llm.ToolChoiceNone:
		// Anthropic doesn't have an explicit "none" - we handle this by not passing tools
		return anthropic.BetaToolChoiceUnionParam{}, errors.New("ToolChoiceNone should be handled by not passing tools")

	case llm.ToolChoiceSpecific:
		if choice.Name == nil || *choice.Name == "" {
			return anthropic.BetaToolChoiceUnionParam{}, errors.New("tool name required for ToolChoiceSpecific")
		}

		return anthropic.BetaToolChoiceUnionParam{
			OfTool: &anthropic.BetaToolChoiceToolParam{
				Type:                   constant.Tool(""),
				Name:                   *choice.Name,
				DisableParallelToolUse: param.NewOpt(true),
			},
		}, nil

	default:
		return anthropic.BetaToolChoiceUnionParam{}, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}
}

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
	"math"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/internal/sampling"
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
	}, nil
}

// converseParams holds the resolved fields shared by Converse and ConverseStream.
type converseParams struct {
	messages  []types.Message
	system    []types.SystemContentBlock
	infConfig *types.InferenceConfiguration
	thinking  document.Interface
	tools     *types.ToolConfiguration
}

// buildConverseParams resolves all common request parameters once.
func (rm *RequestMapper) buildConverseParams(req *llm.Request) (*converseParams, error) {
	if req.Sampling != nil {
		if err := rm.validateSamplingOverride(req.Sampling); err != nil {
			return nil, err
		}
	}

	messages, system, err := rm.mapMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	p := &converseParams{
		messages:  messages,
		system:    system,
		infConfig: rm.buildInferenceConfig(req.Sampling),
	}

	if rm.config.EnableThinking {
		p.thinking = rm.buildThinkingFields()
	}

	if len(req.Tools) > 0 {
		toolConfig, err := rm.mapToolConfig(req.Tools, req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		p.tools = toolConfig
	}

	return p, nil
}

// buildInferenceConfig creates the InferenceConfiguration from config and
// per-request sampling overrides.
func (rm *RequestMapper) buildInferenceConfig(sampling *llm.SamplingParams) *types.InferenceConfiguration {
	// Resolve effective sampling knobs: Request.Sampling overrides per-field;
	// Config provides defaults for fields not set on the request.
	temperature := rm.config.Temperature
	topP := rm.config.TopP
	maxTokens := rm.config.MaxTokens
	stop := rm.config.Stop

	if sampling != nil {
		temperature = llm.CoalesceFloat64(sampling.Temperature, temperature)
		topP = llm.CoalesceFloat64(sampling.TopP, topP)
		stop = llm.CoalesceStrings(sampling.StopSequences, stop)

		if sampling.MaxOutputTokens != nil {
			v := int32(*sampling.MaxOutputTokens) //nolint:gosec // bounds checked in validateSamplingOverride
			maxTokens = &v
		}
	}

	var cfg types.InferenceConfiguration
	hasConfig := false

	if temperature != nil {
		v := float32(*temperature)
		cfg.Temperature = &v
		hasConfig = true
	}

	if topP != nil {
		v := float32(*topP)
		cfg.TopP = &v
		hasConfig = true
	}

	if maxTokens != nil {
		cfg.MaxTokens = maxTokens
		hasConfig = true
	}

	if len(stop) > 0 {
		cfg.StopSequences = stop
		hasConfig = true
	}

	if !hasConfig {
		return nil
	}

	return &cfg
}

// validateSamplingOverride rejects per-request sampling values that fall
// outside the model's constraints. Only fields the Bedrock Converse API
// actually consumes are validated.
func (rm *RequestMapper) validateSamplingOverride(s *llm.SamplingParams) error {
	if s.Temperature != nil {
		if err := rm.config.Constraints.ValidateTemperature(*s.Temperature); err != nil {
			return fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
		}
	}

	if s.TopP != nil && (*s.TopP < 0 || *s.TopP > 1) {
		return fmt.Errorf("%w: top_p must be 0.0-1.0, got %f", llm.ErrRequestMapping, *s.TopP)
	}

	if s.MaxOutputTokens != nil && *s.MaxOutputTokens > math.MaxInt32 {
		return fmt.Errorf("%w: max_output_tokens %d exceeds int32 range", llm.ErrRequestMapping, *s.MaxOutputTokens)
	}

	if err := sampling.ValidateMaxOutputTokens(s.MaxOutputTokens, rm.config.Constraints.MaxOutputTokens); err != nil {
		return fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	if len(s.StopSequences) > 4 {
		return fmt.Errorf("%w: maximum 4 stop sequences allowed, got %d", llm.ErrRequestMapping, len(s.StopSequences))
	}

	return nil
}

// buildThinkingFields returns the additionalModelRequestFields document for
// enabling extended thinking with the configured budget.
func (rm *RequestMapper) buildThinkingFields() document.Interface {
	return document.NewLazyDocument(map[string]any{
		"thinking": map[string]any{
			"type":          "enabled",
			"budget_tokens": rm.config.BudgetTokens,
		},
	})
}

// mapMessages converts llm.Messages to Bedrock Converse types, separating system messages.
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
			if p.Text != "" {
				apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text:      aws.String(p.Text),
							Signature: aws.String(p.ID),
						},
					},
				})
			}

		default:
			return apiMsg, fmt.Errorf("unsupported part type in assistant message: %T", part)
		}
	}

	return apiMsg, nil
}

// mapToolResultBlock converts a tool response to a Bedrock ToolResultBlock.
func (rm *RequestMapper) mapToolResultBlock(resp *llm.ToolResponsePart) types.ToolResultBlock {
	block := types.ToolResultBlock{
		ToolUseId: aws.String(resp.ID),
	}

	if resp.Error != "" {
		block.Status = types.ToolResultStatusError
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberText{Value: resp.Error},
		}
	} else {
		block.Status = types.ToolResultStatusSuccess
		block.Content = []types.ToolResultContentBlock{
			&types.ToolResultContentBlockMemberText{Value: string(resp.Result)},
		}
	}

	return block
}

// mapToolConfig converts tool definitions and choice to Bedrock ToolConfiguration.
func (rm *RequestMapper) mapToolConfig(tools []llm.ToolDefinition, choice *llm.ToolChoice) (*types.ToolConfiguration, error) {
	apiTools := make([]types.Tool, 0, len(tools))

	for _, tool := range tools {
		schemaBytes, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool schema for %s: %w", tool.Name, err)
		}

		var schemaMap map[string]any
		if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
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

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

package openaicompat

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// RequestMapper handles conversion from unified Request to OpenAI Chat Completion API format.
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

// ToProvider converts our unified Request to OpenAI Chat Completion API format.
func (rm *RequestMapper) ToProvider(req *llm.Request) (openai.ChatCompletionNewParams, error) {
	// Create base request for Chat Completion API
	apiReq := openai.ChatCompletionNewParams{
		Model: rm.config.ModelName,
	}

	// Map messages to Chat API format
	messages, err := rm.mapMessages(req.Messages)
	if err != nil {
		return apiReq, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	apiReq.Messages = messages

	// Resolve sampling knobs: Request.Sampling overrides per-field; Config
	// provides defaults for fields not set on the request.
	if req.Sampling != nil {
		if err := rm.validateSamplingOverride(req.Sampling); err != nil {
			return apiReq, err
		}
	}

	temperature := rm.config.Temperature
	topP := rm.config.TopP
	maxTokens := rm.config.MaxTokens
	frequencyPenalty := rm.config.FrequencyPenalty
	presencePenalty := rm.config.PresencePenalty
	stop := rm.config.Stop
	var seed *int64

	if rm.config.Seed != nil {
		v := int64(*rm.config.Seed)
		seed = &v
	}

	if req.Sampling != nil {
		temperature = llm.CoalesceFloat64(req.Sampling.Temperature, temperature)
		topP = llm.CoalesceFloat64(req.Sampling.TopP, topP)
		maxTokens = llm.CoalesceInt(req.Sampling.MaxOutputTokens, maxTokens)
		frequencyPenalty = llm.CoalesceFloat64(req.Sampling.FrequencyPenalty, frequencyPenalty)
		presencePenalty = llm.CoalesceFloat64(req.Sampling.PresencePenalty, presencePenalty)
		stop = llm.CoalesceStrings(req.Sampling.StopSequences, stop)
		seed = llm.CoalesceInt64(req.Sampling.Seed, seed)
	}

	if temperature != nil {
		apiReq.Temperature = openai.Float(*temperature)
	}

	if maxTokens != nil {
		apiReq.MaxTokens = openai.Int(int64(*maxTokens))
	}

	if topP != nil {
		apiReq.TopP = openai.Float(*topP)
	}

	if frequencyPenalty != nil {
		apiReq.FrequencyPenalty = openai.Float(*frequencyPenalty)
	}

	if presencePenalty != nil {
		apiReq.PresencePenalty = openai.Float(*presencePenalty)
	}

	if seed != nil {
		apiReq.Seed = openai.Int(*seed)
	}

	if rm.config.LogProbs != nil && *rm.config.LogProbs {
		apiReq.Logprobs = openai.Bool(true)
		// Default top_logprobs to 5 when logprobs is enabled
		apiReq.TopLogprobs = openai.Int(5)
	}

	if len(stop) > 0 {
		// Stop can be a string or array of strings
		apiReq.Stop = openai.ChatCompletionNewParamsStopUnion{
			OfStringArray: stop,
		}
	}

	// Apply tool definitions if provided
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

	// Apply response format for structured output
	if req.ResponseFormat != nil {
		responseFormat, err := rm.mapResponseFormat(req.ResponseFormat)
		if err != nil {
			return apiReq, fmt.Errorf("%w: response format mapping failed: %w", llm.ErrRequestMapping, err)
		}

		apiReq.ResponseFormat = responseFormat
	}

	return apiReq, nil
}

// mapMessages converts messages to Chat Completion API format.
// This method handles all message types and consolidates parts into chat messages.
func (rm *RequestMapper) mapMessages(messages []llm.Message) ([]openai.ChatCompletionMessageParamUnion, error) {
	apiMessages := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages))

	for _, msg := range messages {
		// Group parts by type for this message
		var (
			textParts     []llm.Part
			toolRequests  []llm.Part
			toolResponses []llm.Part
		)

		for _, part := range msg.Content {
			switch {
			case part.IsText():
				textParts = append(textParts, *part)
			case part.IsToolRequest():
				toolRequests = append(toolRequests, *part)
			case part.IsToolResponse():
				toolResponses = append(toolResponses, *part)
			case part.IsReasoning():
				// Chat API doesn't have native reasoning support
				// Skip reasoning parts
				continue
			default:
				return nil, fmt.Errorf("unknown part kind: %q", part.Kind.String())
			}
		}

		// Map based on role and content types
		switch msg.Role {
		case llm.RoleSystem:
			if len(textParts) > 0 {
				apiMsg := rm.mapSystemMessage(textParts)
				apiMessages = append(apiMessages, apiMsg)
			}

		case llm.RoleUser:
			// Handle tool responses first (takes precedence)
			if len(toolResponses) > 0 {
				for _, part := range toolResponses {
					apiMsg, err := rm.mapToolMessage(&part)
					if err != nil {
						return nil, err
					}

					apiMessages = append(apiMessages, apiMsg)
				}
			} else if len(textParts) > 0 {
				apiMsg := rm.mapUserMessage(textParts)
				apiMessages = append(apiMessages, apiMsg)
			}

		case llm.RoleAssistant:
			if len(toolRequests) > 0 {
				// Assistant message with tool calls
				apiMsg, err := rm.mapAssistantMessageWithTools(textParts, toolRequests)
				if err != nil {
					return nil, err
				}

				apiMessages = append(apiMessages, apiMsg)
			} else if len(textParts) > 0 {
				// Assistant message with text only
				apiMsg := rm.mapAssistantMessage(textParts)
				apiMessages = append(apiMessages, apiMsg)
			}

		default:
			return nil, fmt.Errorf("unsupported message role: %q", msg.Role)
		}
	}

	return apiMessages, nil
}

// mapSystemMessage converts system text parts to a system message.
func (rm *RequestMapper) mapSystemMessage(parts []llm.Part) openai.ChatCompletionMessageParamUnion {
	// Concatenate all text parts
	var text strings.Builder

	for i, part := range parts {
		if i > 0 {
			text.WriteString("\n")
		}

		text.WriteString(part.Text)
	}

	return openai.ChatCompletionMessageParamUnion{
		OfSystem: &openai.ChatCompletionSystemMessageParam{
			Role:    constant.System(""),
			Content: openai.ChatCompletionSystemMessageParamContentUnion{OfString: param.NewOpt(text.String())},
		},
	}
}

// mapUserMessage converts user text parts to a user message.
func (rm *RequestMapper) mapUserMessage(parts []llm.Part) openai.ChatCompletionMessageParamUnion {
	// If single text part, use simple string content
	if len(parts) == 1 && parts[0].IsText() {
		return openai.ChatCompletionMessageParamUnion{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Role:    constant.User(""),
				Content: openai.ChatCompletionUserMessageParamContentUnion{OfString: param.NewOpt(parts[0].Text)},
			},
		}
	}

	// Multiple parts or complex content - use content array
	contentParts := make([]openai.ChatCompletionContentPartUnionParam, 0, len(parts))
	for _, part := range parts {
		contentParts = append(contentParts, openai.ChatCompletionContentPartUnionParam{
			OfText: &openai.ChatCompletionContentPartTextParam{
				Type: constant.Text(""),
				Text: part.Text,
			},
		})
	}

	return openai.ChatCompletionMessageParamUnion{
		OfUser: &openai.ChatCompletionUserMessageParam{
			Role: constant.User(""),
			Content: openai.ChatCompletionUserMessageParamContentUnion{
				OfArrayOfContentParts: contentParts,
			},
		},
	}
}

// mapAssistantMessage converts assistant text parts to an assistant message.
func (rm *RequestMapper) mapAssistantMessage(parts []llm.Part) openai.ChatCompletionMessageParamUnion {
	// Concatenate all text parts
	var (
		text      string
		textSb264 strings.Builder
	)

	for i, part := range parts {
		if i > 0 {
			textSb264.WriteString("\n")
		}

		textSb264.WriteString(part.Text)
	}

	text += textSb264.String()

	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: &openai.ChatCompletionAssistantMessageParam{
			Role:    constant.Assistant(""),
			Content: openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(text)},
		},
	}
}

// mapAssistantMessageWithTools converts assistant message with tool calls.
func (rm *RequestMapper) mapAssistantMessageWithTools(textParts []llm.Part, toolParts []llm.Part) (openai.ChatCompletionMessageParamUnion, error) {
	// Concatenate text parts for content
	var (
		text      string
		textSb283 strings.Builder
	)

	for i, part := range textParts {
		if i > 0 {
			textSb283.WriteString("\n")
		}

		textSb283.WriteString(part.Text)
	}

	text = textSb283.String()

	// Map tool calls
	toolCalls := make([]openai.ChatCompletionMessageToolCallUnionParam, 0, len(toolParts))
	for _, part := range toolParts {
		if part.ToolRequest == nil {
			return openai.ChatCompletionMessageParamUnion{}, errors.New("tool request part has nil ToolRequest")
		}

		toolCalls = append(toolCalls, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID:   part.ToolRequest.ID,
				Type: constant.Function(""),
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      part.ToolRequest.Name,
					Arguments: string(part.ToolRequest.Arguments),
				},
			},
		})
	}

	msg := &openai.ChatCompletionAssistantMessageParam{
		Role:      constant.Assistant(""),
		ToolCalls: toolCalls,
	}

	// Only include content if there's text
	if text != "" {
		msg.Content = openai.ChatCompletionAssistantMessageParamContentUnion{OfString: param.NewOpt(text)}
	}

	return openai.ChatCompletionMessageParamUnion{
		OfAssistant: msg,
	}, nil
}

// mapToolMessage converts a tool response to a tool message.
func (rm *RequestMapper) mapToolMessage(part *llm.Part) (openai.ChatCompletionMessageParamUnion, error) {
	if part.ToolResponse == nil {
		return openai.ChatCompletionMessageParamUnion{}, errors.New("tool response part has nil ToolResponse")
	}

	var content string

	if part.ToolResponse.Error != "" {
		// If there was an error, include it in the content
		errorResult := map[string]any{
			"error": part.ToolResponse.Error,
		}

		errorBytes, err := json.Marshal(errorResult)
		if err != nil {
			return openai.ChatCompletionMessageParamUnion{}, fmt.Errorf("failed to marshal tool error: %w", err)
		}

		content = string(errorBytes)
	} else {
		// Use the successful result
		content = string(part.ToolResponse.Result)
	}

	return openai.ChatCompletionMessageParamUnion{
		OfTool: &openai.ChatCompletionToolMessageParam{
			Role:       constant.Tool(""),
			Content:    openai.ChatCompletionToolMessageParamContentUnion{OfString: param.NewOpt(content)},
			ToolCallID: part.ToolResponse.ID,
		},
	}, nil
}

// mapToolDefinitions converts tool definitions to Chat API format.
func (rm *RequestMapper) mapToolDefinitions(tools []llm.ToolDefinition) ([]openai.ChatCompletionToolUnionParam, error) {
	apiTools := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))

	for _, tool := range tools {
		// Parse the parameters JSON schema
		var parametersMap map[string]any
		if len(tool.Parameters) > 0 {
			err := json.Unmarshal(tool.Parameters, &parametersMap)
			if err != nil {
				return nil, fmt.Errorf("invalid parameters JSON for tool %s: %w", tool.Name, err)
			}
		} else {
			// If no parameters provided, use an empty object schema
			parametersMap = map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			}
		}

		// Transform schema for OpenAI strict mode
		params := rm.schemaMapper.AdaptSchemaForOpenAI(parametersMap)

		apiTools = append(apiTools, openai.ChatCompletionToolUnionParam{
			OfFunction: &openai.ChatCompletionFunctionToolParam{
				Type: constant.Function(""),
				Function: shared.FunctionDefinitionParam{
					Name:        tool.Name,
					Description: param.NewOpt(tool.Description),
					Parameters:  params,
					Strict:      param.NewOpt(true), // Enable strict mode for reliable parameter validation
				},
			},
		})
	}

	return apiTools, nil
}

// mapToolChoice converts tool choice to Chat API format.
func (rm *RequestMapper) mapToolChoice(choice *llm.ToolChoice) (openai.ChatCompletionToolChoiceOptionUnionParam, error) {
	switch choice.Type {
	case llm.ToolChoiceAuto:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openai.ChatCompletionToolChoiceOptionAutoAuto)),
		}, nil

	case llm.ToolChoiceNone:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openai.ChatCompletionToolChoiceOptionAutoNone)),
		}, nil

	case llm.ToolChoiceRequired:
		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfAuto: param.NewOpt(string(openai.ChatCompletionToolChoiceOptionAutoRequired)),
		}, nil

	case llm.ToolChoiceSpecific:
		if choice.Name == nil || *choice.Name == "" {
			return openai.ChatCompletionToolChoiceOptionUnionParam{}, errors.New("specific tool choice requires function name")
		}

		return openai.ChatCompletionToolChoiceOptionUnionParam{
			OfFunctionToolChoice: &openai.ChatCompletionNamedToolChoiceParam{
				Type: constant.Function(""),
				Function: openai.ChatCompletionNamedToolChoiceFunctionParam{
					Name: *choice.Name,
				},
			},
		}, nil

	default:
		return openai.ChatCompletionToolChoiceOptionUnionParam{}, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}
}

// mapResponseFormat converts response format to Chat API format.
func (rm *RequestMapper) mapResponseFormat(format *llm.ResponseFormat) (openai.ChatCompletionNewParamsResponseFormatUnion, error) {
	switch format.Type {
	case llm.ResponseFormatText:
		textParam := shared.NewResponseFormatTextParam()

		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfText: &textParam,
		}, nil

	case llm.ResponseFormatJSONObject:
		jsonObjectParam := shared.NewResponseFormatJSONObjectParam()

		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONObject: &jsonObjectParam,
		}, nil

	case llm.ResponseFormatJSONSchema:
		if format.JSONSchema == nil {
			return openai.ChatCompletionNewParamsResponseFormatUnion{}, errors.New("json_schema format requires schema")
		}

		// Parse the original schema
		var originalSchema map[string]any

		err := json.Unmarshal(format.JSONSchema.Schema, &originalSchema)
		if err != nil {
			return openai.ChatCompletionNewParamsResponseFormatUnion{}, fmt.Errorf("invalid JSON schema: %w", err)
		}

		// Transform schema for OpenAI strict mode
		schemaMap := rm.schemaMapper.AdaptSchemaForOpenAI(originalSchema)

		jsonSchemaParam := shared.ResponseFormatJSONSchemaParam{
			Type: constant.JSONSchema(""),
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        format.JSONSchema.Name,
				Description: param.NewOpt(format.JSONSchema.Description),
				Schema:      schemaMap,
				Strict:      param.NewOpt(true),
			},
		}

		return openai.ChatCompletionNewParamsResponseFormatUnion{
			OfJSONSchema: &jsonSchemaParam,
		}, nil

	default:
		return openai.ChatCompletionNewParamsResponseFormatUnion{}, fmt.Errorf("unsupported response format type: %s", format.Type)
	}
}

// validateSamplingOverride rejects per-request sampling values that fall
// outside the model's constraints. Only fields the Chat Completion API
// consumes are validated.
func (rm *RequestMapper) validateSamplingOverride(s *llm.SamplingParams) error {
	if s.Temperature != nil {
		if err := rm.config.Constraints.ValidateTemperature(*s.Temperature); err != nil {
			return fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
		}
	}

	if s.TopP != nil && (*s.TopP < 0 || *s.TopP > 1) {
		return fmt.Errorf("%w: top_p must be 0.0-1.0, got %f", llm.ErrRequestMapping, *s.TopP)
	}

	if s.MaxOutputTokens != nil && *s.MaxOutputTokens < 1 {
		return fmt.Errorf("%w: max_output_tokens must be positive, got %d", llm.ErrRequestMapping, *s.MaxOutputTokens)
	}

	if s.PresencePenalty != nil && (*s.PresencePenalty < -2.0 || *s.PresencePenalty > 2.0) {
		return fmt.Errorf("%w: presence_penalty must be -2.0 to 2.0, got %f", llm.ErrRequestMapping, *s.PresencePenalty)
	}

	if s.FrequencyPenalty != nil && (*s.FrequencyPenalty < -2.0 || *s.FrequencyPenalty > 2.0) {
		return fmt.Errorf("%w: frequency_penalty must be -2.0 to 2.0, got %f", llm.ErrRequestMapping, *s.FrequencyPenalty)
	}

	if len(s.StopSequences) > 4 {
		return fmt.Errorf("%w: maximum 4 stop sequences allowed, got %d", llm.ErrRequestMapping, len(s.StopSequences))
	}

	return nil
}

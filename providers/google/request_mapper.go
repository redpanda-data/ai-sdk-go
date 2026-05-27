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

package google

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	mimeTypeJSON                = "application/json"
	metadataKeyThoughtSignature = "gemini_thought_signature"
)

// RequestMapper handles conversion from unified Request to Google API format.
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

// ToProvider converts our unified Request to Google API format.
func (rm *RequestMapper) ToProvider(req *llm.Request) ([]*genai.Content, *genai.GenerateContentConfig, error) {
	// Map messages to Content
	contents, systemInstruction, err := rm.mapMessages(req.Messages)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	// Create base config
	config := &genai.GenerateContentConfig{}

	// Set system instruction if present
	if systemInstruction != nil {
		config.SystemInstruction = systemInstruction
	}

	// Resolve sampling knobs: Request.Sampling overrides per-field; Config
	// provides defaults for fields not set on the request.
	if req.Sampling != nil {
		if err := rm.validateSamplingOverride(req.Sampling); err != nil {
			return nil, nil, err
		}
	}

	temperature := rm.config.Temperature
	topP := rm.config.TopP
	topK := rm.config.TopK
	maxTokens := rm.config.MaxTokens
	stop := rm.config.Stop
	presencePenalty := rm.config.PresencePenalty
	frequencyPenalty := rm.config.FrequencyPenalty

	if req.Sampling != nil {
		temperature = llm.CoalesceFloat64(req.Sampling.Temperature, temperature)
		topP = llm.CoalesceFloat64(req.Sampling.TopP, topP)
		stop = llm.CoalesceStrings(req.Sampling.StopSequences, stop)

		if req.Sampling.TopK != nil {
			v := int32(*req.Sampling.TopK) //nolint:gosec // bounds checked in validateSamplingOverride
			topK = &v
		}

		if req.Sampling.MaxOutputTokens != nil {
			v := int32(*req.Sampling.MaxOutputTokens) //nolint:gosec // bounds checked in validateSamplingOverride
			maxTokens = &v
		}

		if req.Sampling.PresencePenalty != nil {
			v := float32(*req.Sampling.PresencePenalty)
			presencePenalty = &v
		}

		if req.Sampling.FrequencyPenalty != nil {
			v := float32(*req.Sampling.FrequencyPenalty)
			frequencyPenalty = &v
		}
	}

	if temperature != nil {
		temp := float32(*temperature)
		config.Temperature = &temp
	}

	if topP != nil {
		v := float32(*topP)
		config.TopP = &v
	}

	if topK != nil {
		v := float32(*topK)
		config.TopK = &v
	}

	if maxTokens != nil {
		config.MaxOutputTokens = *maxTokens
	}

	if len(stop) > 0 {
		config.StopSequences = stop
	}

	if presencePenalty != nil {
		config.PresencePenalty = presencePenalty
	}

	if frequencyPenalty != nil {
		config.FrequencyPenalty = frequencyPenalty
	}

	// Apply response format from request
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSONObject:
			config.ResponseMIMEType = mimeTypeJSON

		case llm.ResponseFormatJSONSchema:
			if req.ResponseFormat.JSONSchema != nil {
				config.ResponseMIMEType = mimeTypeJSON

				schemaBytes, err := json.Marshal(req.ResponseFormat.JSONSchema.Schema)
				if err != nil {
					return nil, nil, fmt.Errorf("%w: marshal response schema: %w", llm.ErrRequestMapping, err)
				}

				var schemaMap map[string]any
				if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
					return nil, nil, fmt.Errorf("%w: failed to parse response schema: %w", llm.ErrRequestMapping, err)
				}

				config.ResponseJsonSchema = schemaMap
			}
		}
	}

	// Apply thinking config if enabled
	if rm.config.EnableThinking {
		config.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
			ThinkingBudget:  rm.config.ThinkingBudget,
		}
	}

	// Apply tool definitions if provided
	if len(req.Tools) > 0 {
		tools, err := rm.mapToolDefinitions(req.Tools)
		if err != nil {
			return nil, nil, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		config.Tools = tools

		// Apply tool choice if specified
		if req.ToolChoice != nil {
			toolConfig, err := rm.mapToolChoice(req.ToolChoice)
			if err != nil {
				return nil, nil, fmt.Errorf("%w: tool choice mapping failed: %w", llm.ErrRequestMapping, err)
			}

			config.ToolConfig = toolConfig
		}
	}

	return contents, config, nil
}

// mapMessages converts our unified messages to Gemini Content format.
// It separates system messages from user/assistant messages.
func (rm *RequestMapper) mapMessages(messages []llm.Message) ([]*genai.Content, *genai.Content, error) {
	var contents []*genai.Content
	var systemInstruction *genai.Content

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleSystem:
			// System messages go into the system instruction
			parts, err := rm.mapParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}

			systemInstruction = &genai.Content{
				Role:  "", // System instruction doesn't have a role
				Parts: parts,
			}

		case llm.RoleUser:
			parts, err := rm.mapParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}

			contents = append(contents, &genai.Content{
				Role:  genai.RoleUser,
				Parts: parts,
			})

		case llm.RoleAssistant:
			parts, err := rm.mapParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}

			contents = append(contents, &genai.Content{
				Role:  genai.RoleModel,
				Parts: parts,
			})

		default:
			return nil, nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	return contents, systemInstruction, nil
}

// mapParts converts unified Parts to Gemini Parts.
func (rm *RequestMapper) mapParts(parts []llm.Part) ([]*genai.Part, error) {
	geminiParts := make([]*genai.Part, 0, len(parts))

	for _, part := range parts {
		switch p := part.(type) {
		case *llm.TextPart:
			geminiParts = append(geminiParts, genai.NewPartFromText(p.Text))

		case *llm.ToolRequestPart:
			// Parse arguments as map for function call
			var args map[string]any
			if err := json.Unmarshal(p.Arguments, &args); err != nil {
				return nil, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			geminiPart := genai.NewPartFromFunctionCall(
				p.Name,
				args,
			)

			// Restore thought signature preserved from previous response (required for Gemini 3 Pro)
			if p.Metadata != nil {
				if sig, ok := p.Metadata[metadataKeyThoughtSignature].([]byte); ok {
					geminiPart.ThoughtSignature = sig
				}
			}

			geminiParts = append(geminiParts, geminiPart)

		case *llm.ToolResponsePart:
			// Parse result as map for function response
			var response map[string]any
			if p.Error != "" {
				// If there was an error, wrap it in the response
				response = map[string]any{
					"error": p.Error,
				}
			} else {
				if err := json.Unmarshal(p.Result, &response); err != nil {
					// If unmarshaling fails, wrap the raw result
					response = map[string]any{
						"result": string(p.Result),
					}
				}
			}

			geminiParts = append(geminiParts, genai.NewPartFromFunctionResponse(
				p.ID,
				response,
			))

		case *llm.ReasoningPart:
			// Gemini thinking is handled automatically by the ThinkingConfig
			// and returned in the response. We don't need to include it in the request.
			// Skip reasoning parts in the input.
			continue

		default:
			return nil, fmt.Errorf("unsupported part type: %T", part)
		}
	}

	return geminiParts, nil
}

// mapToolDefinitions converts our tool definitions to Gemini format.
func (rm *RequestMapper) mapToolDefinitions(tools []llm.ToolDefinition) ([]*genai.Tool, error) {
	functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	for _, tool := range tools {
		// Parse the JSON schema
		schemaBytes, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal tool schema for %s: %w", tool.Name, err)
		}

		var schemaMap map[string]any
		if err := json.Unmarshal(schemaBytes, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse tool schema for %s: %w", tool.Name, err)
		}

		// Adapt the schema for Google (though Google uses standard JSON Schema)
		schema := rm.schemaMapper.AdaptSchemaForGoogle(schemaMap)

		funcDecl := &genai.FunctionDeclaration{
			Name:        tool.Name,
			Description: tool.Description,
			// Use JSON Schema format (preferred for flexibility)
			ParametersJsonSchema: schema,
		}

		functionDeclarations = append(functionDeclarations, funcDecl)
	}

	// Gemini expects tools to be wrapped in a Tool struct
	return []*genai.Tool{
		{
			FunctionDeclarations: functionDeclarations,
		},
	}, nil
}

// mapToolChoice converts our tool choice to Gemini format.
func (rm *RequestMapper) mapToolChoice(choice *llm.ToolChoice) (*genai.ToolConfig, error) {
	config := &genai.ToolConfig{}

	switch choice.Type {
	case llm.ToolChoiceAuto:
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAuto,
		}

	case llm.ToolChoiceRequired:
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeAny,
		}

	case llm.ToolChoiceNone:
		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode: genai.FunctionCallingConfigModeNone,
		}

	case llm.ToolChoiceSpecific:
		if choice.Name == nil || *choice.Name == "" {
			return nil, errors.New("tool name required for ToolChoiceSpecific")
		}

		config.FunctionCallingConfig = &genai.FunctionCallingConfig{
			Mode:                 genai.FunctionCallingConfigModeAny,
			AllowedFunctionNames: []string{*choice.Name},
		}

	default:
		return nil, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}

	return config, nil
}

// validateSamplingOverride rejects per-request sampling values that fall
// outside the model's constraints. Only fields the Google API actually
// consumes are validated here.
func (rm *RequestMapper) validateSamplingOverride(s *llm.SamplingParams) error {
	if s.Temperature != nil {
		if err := rm.config.Constraints.ValidateTemperature(*s.Temperature); err != nil {
			return fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
		}
	}

	if s.TopP != nil && (*s.TopP < 0 || *s.TopP > 1) {
		return fmt.Errorf("%w: top_p must be 0.0-1.0, got %f", llm.ErrRequestMapping, *s.TopP)
	}

	if s.TopK != nil && (*s.TopK < 1 || *s.TopK > math.MaxInt32) {
		return fmt.Errorf("%w: top_k must be in [1, %d], got %d", llm.ErrRequestMapping, math.MaxInt32, *s.TopK)
	}

	if s.MaxOutputTokens != nil && (*s.MaxOutputTokens < 1 || *s.MaxOutputTokens > math.MaxInt32) {
		return fmt.Errorf("%w: max_output_tokens must be in [1, %d], got %d", llm.ErrRequestMapping, math.MaxInt32, *s.MaxOutputTokens)
	}

	if len(s.StopSequences) > 5 {
		return fmt.Errorf("%w: maximum 5 stop sequences allowed, got %d", llm.ErrRequestMapping, len(s.StopSequences))
	}

	if s.PresencePenalty != nil && (*s.PresencePenalty < -2.0 || *s.PresencePenalty > 2.0) {
		return fmt.Errorf("%w: presence_penalty must be -2.0 to 2.0, got %f", llm.ErrRequestMapping, *s.PresencePenalty)
	}

	if s.FrequencyPenalty != nil && (*s.FrequencyPenalty < -2.0 || *s.FrequencyPenalty > 2.0) {
		return fmt.Errorf("%w: frequency_penalty must be -2.0 to 2.0, got %f", llm.ErrRequestMapping, *s.FrequencyPenalty)
	}

	return nil
}

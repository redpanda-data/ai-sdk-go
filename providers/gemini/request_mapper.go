package gemini

import (
	"encoding/json"
	"errors"
	"fmt"

	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	mimeTypeJSON = "application/json"
)

// RequestMapper handles conversion from unified Request to Gemini API format.
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

// ToProvider converts our unified Request to Gemini API format.
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

	// Apply configuration parameters
	if rm.config.Temperature != nil {
		temp := float32(*rm.config.Temperature)
		config.Temperature = &temp
	}

	if rm.config.TopP != nil {
		topP := float32(*rm.config.TopP)
		config.TopP = &topP
	}

	if rm.config.TopK != nil {
		topK := float32(*rm.config.TopK)
		config.TopK = &topK
	}

	if rm.config.MaxTokens != nil {
		config.MaxOutputTokens = *rm.config.MaxTokens
	}

	if len(rm.config.Stop) > 0 {
		config.StopSequences = rm.config.Stop
	}

	if rm.config.PresencePenalty != nil {
		config.PresencePenalty = rm.config.PresencePenalty
	}

	if rm.config.FrequencyPenalty != nil {
		config.FrequencyPenalty = rm.config.FrequencyPenalty
	}

	// Apply response format from request (takes precedence over config)
	//nolint:nestif // Response format handling requires nested structure
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case llm.ResponseFormatJSONObject:
			config.ResponseMIMEType = mimeTypeJSON

		case llm.ResponseFormatJSONSchema:
			if req.ResponseFormat.JSONSchema != nil {
				config.ResponseMIMEType = mimeTypeJSON
				// Convert JSONSchema.Schema (json.RawMessage) to interface{}
				var schemaMap map[string]any
				if err := json.Unmarshal(req.ResponseFormat.JSONSchema.Schema, &schemaMap); err != nil {
					return nil, nil, fmt.Errorf("%w: failed to parse response schema: %w", llm.ErrRequestMapping, err)
				}

				config.ResponseJsonSchema = schemaMap
			}
		}
	} else {
		// Fallback to config-based settings if no request format specified
		if rm.config.ResponseMimeType != nil {
			config.ResponseMIMEType = *rm.config.ResponseMimeType
		}

		if rm.config.ResponseSchema != nil {
			config.ResponseJsonSchema = *rm.config.ResponseSchema
		}
	}

	// Apply thinking config if enabled
	if rm.config.EnableThinking {
		config.ThinkingConfig = &genai.ThinkingConfig{
			IncludeThoughts: true,
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

		case llm.RoleTool:
			// Tool responses are added to the conversation with RoleUser
			parts, err := rm.mapParts(msg.Content)
			if err != nil {
				return nil, nil, err
			}

			contents = append(contents, &genai.Content{
				Role:  genai.RoleUser,
				Parts: parts,
			})

		default:
			return nil, nil, fmt.Errorf("unsupported message role: %s", msg.Role)
		}
	}

	return contents, systemInstruction, nil
}

// mapParts converts unified Parts to Gemini Parts.
func (rm *RequestMapper) mapParts(parts []*llm.Part) ([]*genai.Part, error) {
	geminiParts := make([]*genai.Part, 0, len(parts))

	for _, part := range parts {
		switch {
		case part.IsText():
			geminiParts = append(geminiParts, genai.NewPartFromText(part.Text))

		case part.IsToolRequest():
			if part.ToolRequest == nil {
				return nil, errors.New("tool request part has nil ToolRequest")
			}

			// Parse arguments as map for function call
			var args map[string]any
			if err := json.Unmarshal(part.ToolRequest.Arguments, &args); err != nil {
				return nil, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			geminiParts = append(geminiParts, genai.NewPartFromFunctionCall(
				part.ToolRequest.Name,
				args,
			))

		case part.IsToolResponse():
			if part.ToolResponse == nil {
				return nil, errors.New("tool response part has nil ToolResponse")
			}

			// Parse result as map for function response
			var response map[string]any
			if part.ToolResponse.Error != "" {
				// If there was an error, wrap it in the response
				response = map[string]any{
					"error": part.ToolResponse.Error,
				}
			} else {
				if err := json.Unmarshal(part.ToolResponse.Result, &response); err != nil {
					// If unmarshaling fails, wrap the raw result
					response = map[string]any{
						"result": string(part.ToolResponse.Result),
					}
				}
			}

			geminiParts = append(geminiParts, genai.NewPartFromFunctionResponse(
				part.ToolResponse.ID,
				response,
			))

		case part.IsReasoning():
			// Gemini thinking is handled automatically by the ThinkingConfig
			// and returned in the response. We don't need to include it in the request.
			// Skip reasoning parts in the input.
			continue

		default:
			return nil, fmt.Errorf("unsupported part type: %s", part.Kind)
		}
	}

	return geminiParts, nil
}

// mapToolDefinitions converts our tool definitions to Gemini format.
func (rm *RequestMapper) mapToolDefinitions(tools []llm.ToolDefinition) ([]*genai.Tool, error) {
	functionDeclarations := make([]*genai.FunctionDeclaration, 0, len(tools))

	for _, tool := range tools {
		// Parse the JSON schema
		var schemaMap map[string]any
		if err := json.Unmarshal(tool.Parameters, &schemaMap); err != nil {
			return nil, fmt.Errorf("failed to parse tool schema for %s: %w", tool.Name, err)
		}

		// Adapt the schema for Gemini (though Gemini uses standard JSON Schema)
		schema := rm.schemaMapper.AdaptSchemaForGemini(schemaMap)

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

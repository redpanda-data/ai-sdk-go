package openai

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// RequestMapper handles conversion from unified Request to OpenAI API format.
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

// ToProvider converts our unified Request to OpenAI Responses API format.
func (rm *RequestMapper) ToProvider(req *llm.Request) (responses.ResponseNewParams, error) {
	// Create base request for Responses API
	apiReq := responses.ResponseNewParams{
		Model: rm.config.ModelName,
	}

	// Map input using message format
	inputItems, err := rm.mapMessagesToInputItems(req.Messages)
	if err != nil {
		return apiReq, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	apiReq.Input = responses.ResponseNewParamsInputUnion{
		OfInputItemList: inputItems,
	}

	// Apply configuration parameters
	if rm.config.Temperature != nil {
		apiReq.Temperature = openai.Float(*rm.config.Temperature)
	}

	if rm.config.MaxTokens != nil {
		apiReq.MaxOutputTokens = openai.Int(int64(*rm.config.MaxTokens))
	}

	// Apply reasoning parameters for reasoning models
	// The OpenAI SDK uses awkward discriminated unions - we need to initialize
	// the OfReasoning struct once and then set both fields on it
	if rm.config.ReasoningEffort != nil || rm.config.ReasoningSummary != nil {
		reasoningConfig := shared.ReasoningParam{}

		if rm.config.ReasoningEffort != nil {
			// The field expects a param.Opt[string]. We use param.NewOpt to wrap our value.
			reasoningConfig.Effort = shared.ReasoningEffort(*rm.config.ReasoningEffort)
		}

		if rm.config.ReasoningSummary != nil {
			reasoningConfig.Summary = shared.ReasoningSummary(*rm.config.ReasoningSummary)
		}

		apiReq.Reasoning = reasoningConfig
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
		textConfig, err := rm.mapResponseFormat(req.ResponseFormat)
		if err != nil {
			return apiReq, fmt.Errorf("%w: response format mapping failed: %w", llm.ErrRequestMapping, err)
		}

		apiReq.Text = textConfig
	}

	return apiReq, nil
}

// mapMessagesToInputItems converts messages to the Responses API input format.
// This method handles all message types including regular messages, tool requests, and tool responses.
func (rm *RequestMapper) mapMessagesToInputItems(messages []llm.Message) ([]responses.ResponseInputItemUnionParam, error) {
	items := make([]responses.ResponseInputItemUnionParam, 0, len(messages)*2) // Pre-allocate for potential expansion

	for _, msg := range messages {
		// Process each part in the message content
		for _, part := range msg.Content {
			switch {
			case part.IsText():
				// Text parts cannot be in RoleTool messages
				if msg.Role == llm.RoleTool {
					return nil, fmt.Errorf("%w: RoleTool messages cannot contain text parts", llm.ErrRequestMapping)
				}

				item, err := rm.mapTextMessage(part, msg.Role)
				if err != nil {
					return nil, err
				}

				items = append(items, item)

			case part.IsToolRequest():
				// Tool requests must be in RoleAssistant messages
				if msg.Role != llm.RoleAssistant {
					return nil, fmt.Errorf("%w: tool request parts require RoleAssistant, got %s", llm.ErrRequestMapping, msg.Role)
				}

				item, err := rm.mapToolRequestMessage(part)
				if err != nil {
					return nil, err
				}

				items = append(items, item)

			case part.IsToolResponse():
				// Tool responses must be in RoleTool messages
				if msg.Role != llm.RoleTool {
					return nil, fmt.Errorf("%w: tool response parts require RoleTool, got %s", llm.ErrRequestMapping, msg.Role)
				}

				item, err := rm.mapToolResponseMessage(part)
				if err != nil {
					return nil, err
				}

				items = append(items, item)

			case part.IsReasoning():
				item, err := rm.mapReasoningMessage(part)
				if err != nil {
					return nil, err
				}

				items = append(items, item)

			default:
				return nil, fmt.Errorf("failed mapping unknown part kind: %q", part.Kind.String())
			}
		}
	}

	return items, nil
}

// mapTextMessage converts a text part to an API message input item.
func (rm *RequestMapper) mapTextMessage(part *llm.Part, role llm.MessageRole) (responses.ResponseInputItemUnionParam, error) {
	apiRole, err := rm.mapRoleToAPI(role)
	if err != nil {
		return responses.ResponseInputItemUnionParam{}, err
	}

	apiMsg := responses.EasyInputMessageParam{
		Role: apiRole,
		Content: responses.EasyInputMessageContentUnionParam{
			OfString: param.NewOpt(part.Text),
		},
	}

	return responses.ResponseInputItemUnionParam{
		OfMessage: &apiMsg,
	}, nil
}

// mapToolRequestMessage converts a tool request part to a function call input item.
func (*RequestMapper) mapToolRequestMessage(part *llm.Part) (responses.ResponseInputItemUnionParam, error) {
	if part.ToolRequest == nil {
		return responses.ResponseInputItemUnionParam{}, errors.New("tool request part has nil ToolRequest")
	}

	functionCall := &responses.ResponseFunctionToolCallParam{
		CallID:    part.ToolRequest.ID,
		Name:      part.ToolRequest.Name,
		Arguments: string(part.ToolRequest.Arguments),
		Type:      constant.FunctionCall(""),
	}

	return responses.ResponseInputItemUnionParam{
		OfFunctionCall: functionCall,
	}, nil
}

// mapToolResponseMessage converts a tool response part to a function call output input item.
func (*RequestMapper) mapToolResponseMessage(part *llm.Part) (responses.ResponseInputItemUnionParam, error) {
	if part.ToolResponse == nil {
		return responses.ResponseInputItemUnionParam{}, errors.New("tool response part has nil ToolResponse")
	}

	var output string

	if part.ToolResponse.Error != "" {
		// If there was an error, include it in the output
		errorResult := map[string]any{
			"error": part.ToolResponse.Error,
		}

		errorBytes, err := json.Marshal(errorResult)
		if err != nil {
			return responses.ResponseInputItemUnionParam{}, fmt.Errorf("failed to marshal tool error: %w", err)
		}

		output = string(errorBytes)
	} else {
		// Use the successful result
		output = string(part.ToolResponse.Result)
	}

	functionOutput := &responses.ResponseInputItemFunctionCallOutputParam{
		CallID: part.ToolResponse.ID,
		Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
			OfString: param.NewOpt(output),
		},
		Type: constant.FunctionCallOutput(""),
	}

	return responses.ResponseInputItemUnionParam{
		OfFunctionCallOutput: functionOutput,
	}, nil
}

// mapReasoningMessage converts a reasoning part to a reasoning input item.
func (*RequestMapper) mapReasoningMessage(part *llm.Part) (responses.ResponseInputItemUnionParam, error) {
	if part.ReasoningTrace == nil {
		return responses.ResponseInputItemUnionParam{}, errors.New("reasoning part has nil ReasoningTrace")
	}

	reasoning := &responses.ResponseReasoningItemParam{
		ID:   part.ReasoningTrace.ID,
		Type: constant.Reasoning(""),
		Summary: []responses.ResponseReasoningItemSummaryParam{
			{
				Text: part.ReasoningTrace.Text,
				Type: constant.SummaryText(""),
			},
		},
	}

	return responses.ResponseInputItemUnionParam{
		OfReasoning: reasoning,
	}, nil
}

// mapResponseFormat converts unified ResponseFormat to OpenAI Responses API format.
func (rm *RequestMapper) mapResponseFormat(format *llm.ResponseFormat) (responses.ResponseTextConfigParam, error) {
	textConfig := responses.ResponseTextConfigParam{}

	switch format.Type {
	case llm.ResponseFormatText:
		// Default is text format
		textParam := shared.NewResponseFormatTextParam()
		textConfig.Format = responses.ResponseFormatTextConfigUnionParam{
			OfText: &textParam,
		}

	case llm.ResponseFormatJSONObject:
		// JSON Mode - generates valid JSON but without schema constraints
		jsonObjectParam := shared.NewResponseFormatJSONObjectParam()
		textConfig.Format = responses.ResponseFormatTextConfigUnionParam{
			OfJSONObject: &jsonObjectParam,
		}

	case llm.ResponseFormatJSONSchema:
		// Structured Outputs - generates JSON matching exact schema
		// NOTE: OpenAI requires all parameters to be "required" and uses union types like ["string", "null"]
		// for optional fields. We automatically transform standard JSON schemas to meet these requirements.
		// See: https://platform.openai.com/docs/guides/structured-outputs#all-fields-must-be-required
		if format.JSONSchema == nil {
			return responses.ResponseTextConfigParam{}, errors.New("JSONSchema is required when Type is json_schema")
		}

		// Parse the original schema
		var originalSchema map[string]any

		err := json.Unmarshal(format.JSONSchema.Schema, &originalSchema)
		if err != nil {
			return responses.ResponseTextConfigParam{}, fmt.Errorf("invalid JSON schema: %w", err)
		}

		// Transform the schema for OpenAI compatibility using the schema mapper
		schemaMap := rm.schemaMapper.AdaptSchemaForOpenAI(originalSchema)

		textConfig.Format = responses.ResponseFormatTextConfigUnionParam{
			OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
				Type:        "json_schema",
				Name:        format.JSONSchema.Name,
				Schema:      schemaMap, // Use the transformed schema
				Description: param.NewOpt(format.JSONSchema.Description),
				Strict:      param.NewOpt(true), // Always enable strict mode for maximum reliability
			},
		}

	default:
		return responses.ResponseTextConfigParam{}, fmt.Errorf("unsupported response format type: %s", format.Type)
	}

	return textConfig, nil
}

func (*RequestMapper) mapRoleToAPI(role llm.MessageRole) (responses.EasyInputMessageRole, error) {
	switch role {
	case llm.RoleUser:
		return responses.EasyInputMessageRoleUser, nil
	case llm.RoleAssistant:
		return responses.EasyInputMessageRoleAssistant, nil
	case llm.RoleSystem:
		return responses.EasyInputMessageRoleSystem, nil
	case llm.RoleTool:
		// RoleTool is not a direct input message role in the same way.
		// Tool responses are handled via specific tool output items.
		return "", fmt.Errorf("unsupported message role for OpenAI Responses API: %s", role)
	default:
		return "", fmt.Errorf("unsupported message role for OpenAI Responses API: %s", role)
	}
}

// mapToolDefinitions converts unified tool definitions to OpenAI Responses API function tools.
// Each tool definition is mapped to a FunctionTool variant in the ToolUnionParam.
func (rm *RequestMapper) mapToolDefinitions(tools []llm.ToolDefinition) ([]responses.ToolUnionParam, error) {
	apiTools := make([]responses.ToolUnionParam, 0, len(tools))

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

		// Apply OpenAI schema transformation to ensure compatibility
		// This ensures all properties are marked as required and uses union types for optional fields
		adaptedSchema := rm.schemaMapper.AdaptSchemaForOpenAI(parametersMap)

		// Create function tool parameter
		functionTool := &responses.FunctionToolParam{
			Type:        constant.Function(""),
			Name:        tool.Name,
			Description: param.NewOpt(tool.Description),
			Parameters:  adaptedSchema,      // Use the OpenAI-adapted schema
			Strict:      param.NewOpt(true), // Enable strict parameter validation for reliability
		}

		// Create the tool union parameter with the function variant
		toolUnion := responses.ToolUnionParam{
			OfFunction: functionTool,
		}

		apiTools = append(apiTools, toolUnion)
	}

	return apiTools, nil
}

// mapToolChoice converts unified ToolChoice to OpenAI Responses API format.
func (*RequestMapper) mapToolChoice(choice *llm.ToolChoice) (responses.ResponseNewParamsToolChoiceUnion, error) {
	var toolChoice responses.ResponseNewParamsToolChoiceUnion

	switch choice.Type {
	case llm.ToolChoiceAuto:
		toolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsAuto)
	case llm.ToolChoiceNone:
		toolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsNone)
	case llm.ToolChoiceRequired:
		toolChoice.OfToolChoiceMode = param.NewOpt(responses.ToolChoiceOptionsRequired)
	case llm.ToolChoiceSpecific:
		if choice.Name == nil {
			return toolChoice, errors.New("tool name is required when ToolChoice type is 'specific'")
		}
		// For specific tool choice, we need to use the OfFunctionTool variant
		toolChoice.OfFunctionTool = &responses.ToolChoiceFunctionParam{
			Type: "function",
			Name: *choice.Name,
		}
	default:
		return toolChoice, fmt.Errorf("unsupported tool choice type: %s", choice.Type)
	}

	return toolChoice, nil
}

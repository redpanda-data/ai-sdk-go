package bedrock

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

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
	messages, system, err := rm.mapMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:  aws.String(rm.config.ModelName),
		Messages: messages,
	}

	if len(system) > 0 {
		input.System = system
	}

	// Build inference configuration
	infConfig := rm.buildInferenceConfig()
	if infConfig != nil {
		input.InferenceConfig = infConfig
	}

	// Map tools
	if len(req.Tools) > 0 {
		toolConfig, err := rm.mapToolConfig(req.Tools, req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		input.ToolConfig = toolConfig
	}

	return input, nil
}

// ToConverseStreamInput converts an llm.Request to a bedrockruntime.ConverseStreamInput.
func (rm *RequestMapper) ToConverseStreamInput(req *llm.Request) (*bedrockruntime.ConverseStreamInput, error) {
	messages, system, err := rm.mapMessages(req.Messages)
	if err != nil {
		return nil, fmt.Errorf("%w: message mapping failed: %w", llm.ErrRequestMapping, err)
	}

	input := &bedrockruntime.ConverseStreamInput{
		ModelId:  aws.String(rm.config.ModelName),
		Messages: messages,
	}

	if len(system) > 0 {
		input.System = system
	}

	infConfig := rm.buildInferenceConfig()
	if infConfig != nil {
		input.InferenceConfig = infConfig
	}

	if len(req.Tools) > 0 {
		toolConfig, err := rm.mapToolConfig(req.Tools, req.ToolChoice)
		if err != nil {
			return nil, fmt.Errorf("%w: tool mapping failed: %w", llm.ErrRequestMapping, err)
		}

		input.ToolConfig = toolConfig
	}

	return input, nil
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

// mapMessages converts llm.Messages to Bedrock Converse types, separating system messages.
func (rm *RequestMapper) mapMessages(messages []llm.Message) ([]types.Message, []types.SystemContentBlock, error) {
	var apiMessages []types.Message
	var system []types.SystemContentBlock

	for _, msg := range messages {
		switch msg.Role {
		case llm.RoleSystem:
			for _, part := range msg.Content {
				if part.IsText() {
					system = append(system, &types.SystemContentBlockMemberText{
						Value: part.Text,
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
		switch {
		case part.IsText():
			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberText{
				Value: part.Text,
			})

		case part.IsToolResponse():
			if part.ToolResponse == nil {
				return apiMsg, errors.New("tool response part has nil ToolResponse")
			}

			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberToolResult{
				Value: rm.mapToolResultBlock(part.ToolResponse),
			})

		default:
			return apiMsg, fmt.Errorf("unsupported part type in user message: %s", part.Kind)
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
		switch {
		case part.IsText():
			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberText{
				Value: part.Text,
			})

		case part.IsToolRequest():
			if part.ToolRequest == nil {
				return apiMsg, errors.New("tool request part has nil ToolRequest")
			}

			// Parse arguments to a generic map for document.Interface
			var input map[string]any
			if err := json.Unmarshal(part.ToolRequest.Arguments, &input); err != nil {
				return apiMsg, fmt.Errorf("failed to parse tool arguments: %w", err)
			}

			apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberToolUse{
				Value: types.ToolUseBlock{
					ToolUseId: aws.String(part.ToolRequest.ID),
					Name:      aws.String(part.ToolRequest.Name),
					Input:     document.NewLazyDocument(input),
				},
			})

		case part.IsReasoning():
			// Pass reasoning traces back as reasoning content blocks
			if part.ReasoningTrace != nil && part.ReasoningTrace.Text != "" {
				apiMsg.Content = append(apiMsg.Content, &types.ContentBlockMemberReasoningContent{
					Value: &types.ReasoningContentBlockMemberReasoningText{
						Value: types.ReasoningTextBlock{
							Text:      aws.String(part.ReasoningTrace.Text),
							Signature: aws.String(part.ReasoningTrace.ID),
						},
					},
				})
			}

		default:
			return apiMsg, fmt.Errorf("unsupported part type in assistant message: %s", part.Kind)
		}
	}

	return apiMsg, nil
}

// mapToolResultBlock converts a tool response to a Bedrock ToolResultBlock.
func (rm *RequestMapper) mapToolResultBlock(resp *llm.ToolResponse) types.ToolResultBlock {
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
		return nil, errors.New("ToolChoiceNone should be handled by not passing tools")

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

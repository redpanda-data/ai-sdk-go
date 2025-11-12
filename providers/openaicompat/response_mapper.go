package openaicompat

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper handles conversion from OpenAI Chat Completion API responses to unified format.
type ResponseMapper struct{}

// NewResponseMapper creates a new ResponseMapper. The mapper is stateless and can be reused.
func NewResponseMapper() *ResponseMapper {
	return &ResponseMapper{}
}

// FromProvider converts an OpenAI ChatCompletion response to our unified Response format.
func (rm *ResponseMapper) FromProvider(apiResp *openai.ChatCompletion) (*llm.Response, error) {
	if apiResp == nil {
		return nil, fmt.Errorf("%w: response is nil", llm.ErrResponseMapping)
	}

	// Extract the first choice (Chat API always returns at least one choice)
	if len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("%w: no choices in response", llm.ErrResponseMapping)
	}

	choice := apiResp.Choices[0]
	message := choice.Message

	// Build response content from message
	// Pre-allocate for reasoning + text + tool calls
	content := make([]*llm.Part, 0, 2+len(message.ToolCalls))

	// Check for reasoning_content in extra fields (for reasoning models like DeepSeek-R1, o1)
	// This field is not yet in the official Go SDK but is in the API response
	if reasoningField, ok := message.JSON.ExtraFields["reasoning_content"]; ok && reasoningField.Raw() != "" {
		var reasoningContent string
		if err := json.Unmarshal([]byte(reasoningField.Raw()), &reasoningContent); err == nil && reasoningContent != "" {
			content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
				Text: reasoningContent,
			}))
		}
	}

	// Add text content if present
	if message.Content != "" {
		content = append(content, &llm.Part{
			Kind: llm.PartText,
			Text: message.Content,
		})
	}

	// Add tool calls if present
	for _, toolCall := range message.ToolCalls {
		// Skip non-function tool calls
		if toolCall.Type != "function" {
			continue
		}

		content = append(content, &llm.Part{
			Kind: llm.PartToolRequest,
			ToolRequest: &llm.ToolRequest{
				ID:        toolCall.ID,
				Name:      toolCall.Function.Name,
				Arguments: json.RawMessage(toolCall.Function.Arguments),
			},
		})
	}

	// Map finish reason
	finishReason, err := rm.mapFinishReason(choice.FinishReason, len(message.ToolCalls) > 0)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrResponseMapping, err)
	}

	// Extract usage statistics
	var usage *llm.TokenUsage
	if apiResp.Usage.TotalTokens > 0 {
		usage = &llm.TokenUsage{
			InputTokens:     int(apiResp.Usage.PromptTokens),
			OutputTokens:    int(apiResp.Usage.CompletionTokens),
			TotalTokens:     int(apiResp.Usage.TotalTokens),
			ReasoningTokens: int(apiResp.Usage.CompletionTokensDetails.ReasoningTokens),
		}
	} else {
		// If no usage provided, return empty usage structure
		// This can happen with some OpenAI-compatible APIs
		usage = &llm.TokenUsage{
			InputTokens:     0,
			OutputTokens:    0,
			TotalTokens:     0,
			ReasoningTokens: 0,
		}
	}

	return &llm.Response{
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        usage,
		ID:           apiResp.ID,
	}, nil
}

// FromProviderError maps OpenAI API errors to SDK error categories.
func (rm *ResponseMapper) FromProviderError(err error) error {
	// For now, just pass through the error
	// TODO: Map specific OpenAI error types to SDK error categories
	return err
}

// mapFinishReason converts OpenAI finish reason to unified format.
func (rm *ResponseMapper) mapFinishReason(reason string, hasToolCalls bool) (llm.FinishReason, error) {
	// Tool calls take precedence
	if hasToolCalls {
		return llm.FinishReasonToolCalls, nil
	}

	switch reason {
	case "stop":
		return llm.FinishReasonStop, nil

	case "length":
		return llm.FinishReasonLength, nil

	case "content_filter":
		return llm.FinishReasonContentFilter, nil

	case "tool_calls":
		return llm.FinishReasonToolCalls, nil

	default:
		return llm.FinishReasonUnknown, fmt.Errorf("unknown finish reason: %s", reason)
	}
}

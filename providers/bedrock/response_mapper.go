package bedrock

import (
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper converts Bedrock Converse API responses to llm.Response.
type ResponseMapper struct {
	modelDefinition ModelDefinition
}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper(definition ModelDefinition) *ResponseMapper {
	return &ResponseMapper{modelDefinition: definition}
}

// FromConverseOutput converts a Bedrock ConverseOutput to llm.Response.
func (m *ResponseMapper) FromConverseOutput(stopReason types.StopReason, output types.ConverseOutput, usage *types.TokenUsage) (*llm.Response, error) {
	if output == nil {
		return nil, fmt.Errorf("%w: nil provider output", llm.ErrResponseMapping)
	}

	// Extract message from the output union
	msgOutput, ok := output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return nil, fmt.Errorf("%w: unexpected output type", llm.ErrResponseMapping)
	}

	content, hasToolCalls := m.mapContentBlocks(msgOutput.Value.Content)

	var tokenUsage *llm.TokenUsage
	if usage != nil {
		tokenUsage = m.mapTokenUsage(usage)
	}

	var finishReason llm.FinishReason
	if hasToolCalls {
		finishReason = llm.FinishReasonToolCalls
	} else {
		finishReason = m.mapStopReason(stopReason)
	}

	return &llm.Response{
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        tokenUsage,
	}, nil
}

// mapContentBlocks converts Bedrock content blocks to llm.Parts.
func (m *ResponseMapper) mapContentBlocks(blocks []types.ContentBlock) ([]*llm.Part, bool) {
	parts := make([]*llm.Part, 0, len(blocks))
	hasToolCalls := false

	for _, block := range blocks {
		switch v := block.(type) {
		case *types.ContentBlockMemberText:
			if v.Value != "" {
				parts = append(parts, llm.NewTextPart(v.Value))
			}

		case *types.ContentBlockMemberToolUse:
			hasToolCalls = true

			parts = append(parts, m.mapToolUseBlock(&v.Value))

		case *types.ContentBlockMemberReasoningContent:
			if part := m.mapReasoningBlock(v.Value); part != nil {
				parts = append(parts, part)
			}
		}
	}

	return parts, hasToolCalls
}

// mapStopReason converts Bedrock's StopReason to llm.FinishReason.
func (m *ResponseMapper) mapStopReason(reason types.StopReason) llm.FinishReason {
	switch reason {
	case types.StopReasonEndTurn, types.StopReasonStopSequence:
		return llm.FinishReasonStop

	case types.StopReasonToolUse:
		return llm.FinishReasonToolCalls

	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return llm.FinishReasonLength

	case types.StopReasonContentFiltered, types.StopReasonGuardrailIntervened:
		return llm.FinishReasonContentFilter

	case types.StopReasonMalformedModelOutput, types.StopReasonMalformedToolUse:
		return llm.FinishReasonStop

	default:
		return llm.FinishReasonStop
	}
}

// mapToolUseBlock converts a Bedrock ToolUseBlock to an llm.Part.
func (m *ResponseMapper) mapToolUseBlock(block *types.ToolUseBlock) *llm.Part {
	argsJSON := m.marshalToolInput(block.Input)

	var id, name string
	if block.ToolUseId != nil {
		id = *block.ToolUseId
	}

	if block.Name != nil {
		name = *block.Name
	}

	return llm.NewToolRequestPart(&llm.ToolRequest{
		ID:        id,
		Name:      name,
		Arguments: argsJSON,
	})
}

// marshalToolInput marshals a document.Interface to JSON, defaulting to "{}".
func (m *ResponseMapper) marshalToolInput(input interface{ UnmarshalSmithyDocument(any) error }) json.RawMessage {
	if input != nil {
		var raw any
		if err := input.UnmarshalSmithyDocument(&raw); err == nil {
			if b, err := json.Marshal(raw); err == nil {
				return b
			}
		}
	}

	return json.RawMessage("{}")
}

// mapReasoningBlock converts a Bedrock ReasoningContentBlock to an llm.Part, or nil if empty.
func (m *ResponseMapper) mapReasoningBlock(value types.ReasoningContentBlock) *llm.Part {
	if value == nil {
		return nil
	}

	rt, ok := value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok {
		return nil
	}

	var text, sig string
	if rt.Value.Text != nil {
		text = *rt.Value.Text
	}

	if rt.Value.Signature != nil {
		sig = *rt.Value.Signature
	}

	if text == "" {
		return nil
	}

	return llm.NewReasoningPart(&llm.ReasoningTrace{
		ID:   sig,
		Text: text,
	})
}

// mapTokenUsage converts Bedrock TokenUsage to llm.TokenUsage.
func (m *ResponseMapper) mapTokenUsage(usage *types.TokenUsage) *llm.TokenUsage {
	result := &llm.TokenUsage{
		MaxInputTokens: m.modelDefinition.Constraints.MaxInputTokens,
	}

	if usage.InputTokens != nil {
		result.InputTokens = int(*usage.InputTokens)
	}

	if usage.OutputTokens != nil {
		result.OutputTokens = int(*usage.OutputTokens)
	}

	if usage.TotalTokens != nil {
		result.TotalTokens = int(*usage.TotalTokens)
	}

	if usage.CacheReadInputTokens != nil {
		result.CachedTokens = int(*usage.CacheReadInputTokens)
	}

	return result
}

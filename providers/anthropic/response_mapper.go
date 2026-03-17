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
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	blockTypeText     = "text"
	blockTypeToolUse  = "tool_use"
	blockTypeThinking = "thinking"
)

// ResponseMapper converts Anthropic API payloads to llm.Response.
type ResponseMapper struct {
	modelDefinition ModelDefinition
}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper(definition ModelDefinition) *ResponseMapper {
	return &ResponseMapper{modelDefinition: definition}
}

// FromProvider converts an Anthropic Beta Messages API payload into llm.Response.
func (m *ResponseMapper) FromProvider(r *anthropic.BetaMessage) (*llm.Response, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil provider response", llm.ErrResponseMapping)
	}

	// Collect content from response blocks
	content := make([]*llm.Part, 0, len(r.Content))
	hasToolCalls := false

	for _, block := range r.Content {
		switch block.Type {
		case blockTypeText:
			// Text content block
			if block.Text != "" {
				content = append(content, llm.NewTextPart(block.Text))
			}

		case blockTypeToolUse:
			// Tool use block
			hasToolCalls = true

			content = append(content, llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			}))

		case blockTypeThinking:
			// Thinking block (extended thinking / reasoning)
			if block.Thinking != "" {
				content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
					ID:   block.Signature,
					Text: block.Thinking,
				}))
			}

		case "redacted_thinking":
			// Redacted thinking block - include metadata but no text
			content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
				ID:       block.Signature,
				Text:     "[redacted thinking]",
				Metadata: map[string]any{"redacted": true},
			}))

		default:
			// Unknown block type - skip it
			continue
		}
	}

	// Extract usage information
	var usage *llm.TokenUsage
	if r.Usage.InputTokens > 0 || r.Usage.OutputTokens > 0 {
		usage = &llm.TokenUsage{
			InputTokens:     int(r.Usage.InputTokens),
			OutputTokens:    int(r.Usage.OutputTokens),
			TotalTokens:     int(r.Usage.InputTokens + r.Usage.OutputTokens),
			CachedTokens:    int(r.Usage.CacheReadInputTokens),
			ReasoningTokens: 0, // Anthropic doesn't separate reasoning tokens in usage
			MaxInputTokens:  m.modelDefinition.Constraints.MaxInputTokens,
		}
	}

	// Map finish reason
	var finishReason llm.FinishReason
	if hasToolCalls {
		finishReason = llm.FinishReasonToolCalls
	} else {
		finishReason = m.mapStopReason(r.StopReason)
	}

	return &llm.Response{
		ID: r.ID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
}

// mapStopReason converts Anthropic's stop reason to our unified finish reason.
func (m *ResponseMapper) mapStopReason(reason anthropic.BetaStopReason) llm.FinishReason {
	switch reason {
	case anthropic.BetaStopReasonEndTurn:
		return llm.FinishReasonStop

	case anthropic.BetaStopReasonMaxTokens:
		return llm.FinishReasonLength

	case anthropic.BetaStopReasonStopSequence:
		return llm.FinishReasonStop

	case anthropic.BetaStopReasonToolUse:
		return llm.FinishReasonToolCalls

	case anthropic.BetaStopReasonRefusal:
		return llm.FinishReasonContentFilter

	case anthropic.BetaStopReasonModelContextWindowExceeded:
		return llm.FinishReasonLength

	case anthropic.BetaStopReasonPauseTurn:
		// Paused turns are a special case - treat as incomplete
		return llm.FinishReasonLength

	default:
		// Unknown reason - default to stop
		return llm.FinishReasonStop
	}
}

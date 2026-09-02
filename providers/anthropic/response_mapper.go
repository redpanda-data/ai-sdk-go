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
type ResponseMapper struct{}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper() *ResponseMapper {
	return &ResponseMapper{}
}

// FromProvider converts an Anthropic Beta Messages API payload into llm.Response.
func (m *ResponseMapper) FromProvider(r *anthropic.BetaMessage) (*llm.Response, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil provider response", llm.ErrResponseMapping)
	}

	// Collect content from response blocks
	content := make([]llm.Part, 0, len(r.Content))
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

			content = append(content, llm.NewToolRequestPart(block.ID, block.Name, block.Input))

		case blockTypeThinking:
			// Thinking block (extended thinking / reasoning)
			if block.Thinking != "" {
				content = append(content, &llm.ReasoningPart{
					Text:      block.Thinking,
					Signature: block.Signature,
				})
			}

		case "redacted_thinking":
			// Redacted thinking block - include metadata but no text
			content = append(content, &llm.ReasoningPart{
				Text:      "[redacted thinking]",
				Signature: block.Signature,
				Metadata:  map[string]any{"redacted": true},
			})

		default:
			// Unknown block type - skip it
			continue
		}
	}

	// Extract usage information. Anthropic's input_tokens, cache_read_input_tokens,
	// and cache_creation_input_tokens are already disjoint in the API response,
	// so they map straight onto llm.TokenUsage's disjoint buckets. The per-TTL
	// breakdown lives in usage.cache_creation.ephemeral_{5m,1h}_input_tokens; if
	// that breakdown is absent or covers fewer tokens than the aggregate
	// cache_creation_input_tokens, the remainder lands in
	// CacheCreationUnknownTTLTokens so BilledInputTokens() stays accurate.
	//
	// Anthropic's thinking tokens are billed at the output rate and are not
	// reported separately in usage, so ReasoningTokens stays zero.
	var usage *llm.TokenUsage

	if r.Usage.InputTokens > 0 || r.Usage.OutputTokens > 0 ||
		r.Usage.CacheReadInputTokens > 0 || r.Usage.CacheCreationInputTokens > 0 {
		ephemeral5m := int(r.Usage.CacheCreation.Ephemeral5mInputTokens)
		ephemeral1h := int(r.Usage.CacheCreation.Ephemeral1hInputTokens)

		var unknownTTL int
		if aggregate := int(r.Usage.CacheCreationInputTokens); aggregate > ephemeral5m+ephemeral1h {
			unknownTTL = aggregate - ephemeral5m - ephemeral1h
		}

		usage = &llm.TokenUsage{
			InputTokens:                   int(r.Usage.InputTokens),
			CachedInputTokens:             int(r.Usage.CacheReadInputTokens),
			CacheCreation5mTokens:         ephemeral5m,
			CacheCreation1hTokens:         ephemeral1h,
			CacheCreationUnknownTTLTokens: unknownTTL,
			OutputTokens:                  int(r.Usage.OutputTokens),
		}
	}

	// Map finish reason. The provider's own stop reason wins when it signals
	// anything other than a clean "end_turn": max_tokens, context_window,
	// pause_turn, refusal all need to propagate so the agent loop can react.
	// Only upgrade a plain Stop to ToolCalls when tool_use blocks are
	// present — truncation signals must never be masked by tool calls that
	// happened to complete before the stream was cut short.
	finishReason := m.mapStopReason(r.StopReason)
	if hasToolCalls && finishReason == llm.FinishReasonStop {
		finishReason = llm.FinishReasonToolCalls
	}

	return &llm.Response{
		ID: r.ID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason:    finishReason,
		Usage:           usage,
		ServiceTier:     llm.NormalizeServiceTier(string(r.Usage.ServiceTier)),
		Speed:           llm.NormalizeSpeed(string(r.Usage.Speed)),
		InferenceRegion: r.Usage.InferenceGeo,
		InvokedModelID:  resolveInvokedModelID(r.Model),
	}, nil
}

// resolveInvokedModelID collapses a provider-reported model ID (possibly a
// timestamped snapshot) to its catalog offering ID; IDs the catalog does
// not know pass through unchanged.
func resolveInvokedModelID(model string) string {
	if id, ok := Catalog().ResolveID(model); ok {
		return id
	}

	return model
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
		// Input overflowed the context window — a genuinely different condition
		// from an output-token cut (max_tokens). Keep them distinct so downstream
		// can deliver-and-continue on truncation but fail truthfully on overflow.
		return llm.FinishReasonContextOverflow

	case anthropic.BetaStopReasonPauseTurn:
		// Paused turns are a special case - treat as incomplete
		return llm.FinishReasonLength

	default:
		// Unknown reason - default to stop
		return llm.FinishReasonStop
	}
}

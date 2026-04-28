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
func (m *ResponseMapper) FromConverseOutput(
	stopReason types.StopReason,
	output types.ConverseOutput,
	usage *types.TokenUsage,
	performanceConfig *types.PerformanceConfiguration,
	serviceTier *types.ServiceTier,
	trace *types.ConverseTrace,
) (*llm.Response, error) {
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

	// Map finish reason. Truncation signals (max_tokens, etc.) must propagate
	// through to the caller; only upgrade a plain Stop to ToolCalls when tool
	// use blocks are present. See providers/anthropic/response_mapper.go for
	// the full rationale.
	finishReason := m.mapStopReason(stopReason)
	if hasToolCalls && finishReason == llm.FinishReasonStop {
		finishReason = llm.FinishReasonToolCalls
	}

	// Empty content surfaces as error regardless of stop reason — see
	// providers/anthropic/response_mapper.go for full rationale.
	if len(content) == 0 {
		return nil, fmt.Errorf("%w: provider returned no content blocks (stop_reason=%s)",
			llm.ErrResponseMapping, stopReason)
	}

	resp := &llm.Response{
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason:   finishReason,
		Usage:          tokenUsage,
		InvokedModelID: m.modelDefinition.Name,
	}

	m.applyResponseMetadata(resp, performanceConfig, serviceTier, promptRouterFromTrace(trace))

	return resp, nil
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

// mapTokenUsage converts Bedrock Converse TokenUsage to the normalized
// llm.TokenUsage shape. Bedrock's InputTokens, CacheReadInputTokens, and
// CacheWriteInputTokens are already disjoint, matching the shape directly.
// Per-TTL write breakdown from CacheDetails is split into 5m / 1h buckets;
// if the breakdown is missing or covers fewer tokens than the aggregate
// CacheWriteInputTokens, the remainder routes to CacheCreationUnknownTTLTokens
// so BilledInputTokens() stays accurate. Unknown TTL string values (if
// Bedrock introduces more) are still recorded in the unknown-TTL bucket.
func (m *ResponseMapper) mapTokenUsage(usage *types.TokenUsage) *llm.TokenUsage {
	result := &llm.TokenUsage{}

	if usage.InputTokens != nil {
		result.InputTokens = int(*usage.InputTokens)
	}

	if usage.OutputTokens != nil {
		result.OutputTokens = int(*usage.OutputTokens)
	}

	if usage.CacheReadInputTokens != nil {
		result.CachedInputTokens = int(*usage.CacheReadInputTokens)
	}

	var knownBreakdownTokens int

	for _, detail := range usage.CacheDetails {
		if detail.InputTokens == nil {
			continue
		}

		tokens := int(*detail.InputTokens)
		switch detail.Ttl {
		case types.CacheTTLFiveMinutes:
			result.CacheCreation5mTokens += tokens
			knownBreakdownTokens += tokens
		case types.CacheTTLOneHour:
			result.CacheCreation1hTokens += tokens
			knownBreakdownTokens += tokens
		default:
			result.CacheCreationUnknownTTLTokens += tokens
			knownBreakdownTokens += tokens
		}
	}

	if usage.CacheWriteInputTokens != nil {
		if aggregate := int(*usage.CacheWriteInputTokens); aggregate > knownBreakdownTokens {
			result.CacheCreationUnknownTTLTokens += aggregate - knownBreakdownTokens
		}
	}

	return result
}

func (m *ResponseMapper) applyResponseMetadata(
	resp *llm.Response,
	performanceConfig *types.PerformanceConfiguration,
	serviceTier *types.ServiceTier,
	promptRouter *types.PromptRouterTrace,
) {
	if resp == nil {
		return
	}

	if performanceConfig != nil && performanceConfig.Latency != "" {
		resp.Speed = llm.NormalizeSpeed(string(performanceConfig.Latency))
	}

	if serviceTier != nil {
		resp.ServiceTier = llm.NormalizeServiceTier(string(serviceTier.Type))
	}

	if promptRouter != nil && promptRouter.InvokedModelId != nil && *promptRouter.InvokedModelId != "" {
		if def, ok := lookupModel(*promptRouter.InvokedModelId); ok {
			resp.InvokedModelID = def.Name
			return
		}

		resp.InvokedModelID = *promptRouter.InvokedModelId
	}
}

func promptRouterFromTrace(trace *types.ConverseTrace) *types.PromptRouterTrace {
	if trace == nil {
		return nil
	}

	return trace.PromptRouter
}

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
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper converts Google API payloads to llm.Response.
type ResponseMapper struct {
	modelDefinition ModelDefinition
}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper(definition ModelDefinition) *ResponseMapper {
	return &ResponseMapper{modelDefinition: definition}
}

// FromProvider converts a Google GenerateContentResponse into llm.Response.
func (m *ResponseMapper) FromProvider(r *genai.GenerateContentResponse) (*llm.Response, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil provider response", llm.ErrResponseMapping)
	}

	// Gemini can return multiple candidates, but we typically use the first one
	if len(r.Candidates) == 0 {
		return nil, fmt.Errorf("%w: no candidates in response", llm.ErrResponseMapping)
	}

	candidate := r.Candidates[0]
	if candidate.Content == nil {
		return nil, fmt.Errorf("%w: candidate has no content", llm.ErrResponseMapping)
	}

	// Convert content parts
	content, hasToolCalls, err := m.mapParts(candidate.Content.Parts)
	if err != nil {
		return nil, err
	}

	// Extract usage information. Google reports CachedContentTokenCount as a
	// subset of PromptTokenCount; the normalized llm.TokenUsage shape is
	// disjoint, so we un-subset it here. ThoughtsTokenCount is already
	// additive in Google's model (the SDK doc states
	// TotalTokenCount = prompt + candidates + tool_use + thoughts), so it
	// maps directly onto ReasoningTokens. ToolUsePromptTokenCount is a
	// separate billable input dimension and populates ToolUseInputTokens.
	var usage *llm.TokenUsage

	if r.UsageMetadata != nil {
		cachedIn := int(r.UsageMetadata.CachedContentTokenCount)
		usage = &llm.TokenUsage{
			InputTokens:        int(r.UsageMetadata.PromptTokenCount) - cachedIn,
			CachedInputTokens:  cachedIn,
			ToolUseInputTokens: int(r.UsageMetadata.ToolUsePromptTokenCount),
			OutputTokens:       int(r.UsageMetadata.CandidatesTokenCount),
			ReasoningTokens:    int(r.UsageMetadata.ThoughtsTokenCount),
		}
	}

	// Map finish reason
	finishReason := m.mapFinishReason(candidate.FinishReason, hasToolCalls)

	// Empty content surfaces as error regardless of finish reason — see
	// providers/anthropic/response_mapper.go for full rationale. Gemini
	// can produce empty content when max_tokens hits before any Part is
	// emitted, when safety filters strip all text, or when the candidate
	// only carries unsupported part types (file_data, inline_data).
	if len(content) == 0 {
		return nil, fmt.Errorf("%w: provider returned no content blocks (finish_reason=%s)",
			llm.ErrResponseMapping, candidate.FinishReason)
	}

	return &llm.Response{
		ID: r.ResponseID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason:   finishReason,
		Usage:          usage,
		InvokedModelID: resolvedModelID(r.ModelVersion, m.modelDefinition.Name),
	}, nil
}

func resolvedModelID(providerModel, fallback string) string {
	if providerModel == "" {
		return fallback
	}

	resolved := resolveModelFamily(providerModel)
	if resolved == "" {
		return fallback
	}

	return resolved
}

// mapParts converts Gemini Parts to unified Parts.
func (m *ResponseMapper) mapParts(parts []*genai.Part) ([]*llm.Part, bool, error) {
	content := make([]*llm.Part, 0, len(parts))
	hasToolCalls := false

	for _, part := range parts {
		switch {
		case part.Text != "":
			// Check if this is a thinking/reasoning part
			if part.Thought {
				// This is a thinking part
				var signature string
				if len(part.ThoughtSignature) > 0 {
					signature = string(part.ThoughtSignature)
				}

				content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
					ID:   signature,
					Text: part.Text,
				}))
			} else {
				// Regular text part
				content = append(content, llm.NewTextPart(part.Text))
			}

		case part.FunctionCall != nil:
			// Tool use (function call)
			hasToolCalls = true

			// Convert arguments to JSON
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, false, fmt.Errorf("failed to marshal function call args: %w", err)
			}

			// Generate ID if not provided by Gemini (using cmp.Or pattern from fantasy)
			id := cmp.Or(part.FunctionCall.ID, uuid.New().String())

			toolPart := llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        id,
				Name:      part.FunctionCall.Name,
				Arguments: argsJSON,
			})

			// Preserve thought signature for Gemini 3 Pro multi-turn conversations
			// Gemini 3 Pro requires thought signatures to be passed back during function calling
			if len(part.ThoughtSignature) > 0 {
				if toolPart.Metadata == nil {
					toolPart.Metadata = make(map[string]any)
				}

				toolPart.Metadata[metadataKeyThoughtSignature] = part.ThoughtSignature
			}

			content = append(content, toolPart)

		default:
			// Skip unsupported part types (file data, inline data, etc.)
			continue
		}
	}

	return content, hasToolCalls, nil
}

// mapFinishReason converts Gemini's finish reason to our unified finish reason.
func (m *ResponseMapper) mapFinishReason(reason genai.FinishReason, hasToolCalls bool) llm.FinishReason {
	mapped := mapGenAIFinishReason(reason)

	// Only upgrade a plain Stop to ToolCalls when tool use blocks are
	// present. Truncation and safety signals must propagate through so the
	// agent loop can react. See providers/anthropic/response_mapper.go for
	// the full rationale.
	if hasToolCalls && mapped == llm.FinishReasonStop {
		return llm.FinishReasonToolCalls
	}

	return mapped
}

func mapGenAIFinishReason(reason genai.FinishReason) llm.FinishReason {
	switch reason {
	case genai.FinishReasonStop:
		return llm.FinishReasonStop

	case genai.FinishReasonMaxTokens:
		return llm.FinishReasonLength

	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent:
		return llm.FinishReasonContentFilter

	case genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall:
		return llm.FinishReasonUnknown

	case genai.FinishReasonLanguage,
		genai.FinishReasonNoImage:
		return llm.FinishReasonContentFilter

	case genai.FinishReasonUnspecified,
		genai.FinishReasonOther:
		return llm.FinishReasonStop

	default:
		// Unknown reason - default to stop
		return llm.FinishReasonStop
	}
}

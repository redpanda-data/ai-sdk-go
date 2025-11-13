package gemini

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper converts Gemini API payloads to llm.Response.
// It's stateless; the zero value is ready to use.
type ResponseMapper struct{}

// NewResponseMapper returns a ready-to-use mapper.
// Zero value is also valid; this exists for callers that prefer constructors.
func NewResponseMapper() *ResponseMapper { return &ResponseMapper{} }

// stripMarkdownCodeFence removes markdown code fences from JSON responses.
// Gemini sometimes wraps JSON in ```json ... ``` even in JSON mode.
//
//nolint:nestif // Fence parsing requires nested structure
func stripMarkdownCodeFence(text string) string {
	// Remove ```json\n at start and \n``` at end
	if len(text) > 7 && text[0:3] == "```" {
		// Find first newline after opening fence
		start := 3
		for start < len(text) && text[start] != '\n' {
			start++
		}

		if start < len(text) {
			start++ // skip the newline
		}

		// Find closing fence
		end := len(text)
		if end > 3 && text[end-3:] == "```" {
			end -= 3
			// Also trim trailing newline before fence
			if end > 0 && text[end-1] == '\n' {
				end--
			}
		}

		if start < end {
			return text[start:end]
		}
	}

	return text
}

// FromProvider converts a Gemini GenerateContentResponse into llm.Response.
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

	// Extract usage information
	var usage *llm.TokenUsage
	if r.UsageMetadata != nil {
		usage = &llm.TokenUsage{
			InputTokens:  int(r.UsageMetadata.PromptTokenCount),
			OutputTokens: int(r.UsageMetadata.CandidatesTokenCount),
			TotalTokens:  int(r.UsageMetadata.TotalTokenCount),
			// Gemini doesn't separate reasoning tokens in usage
			ReasoningTokens: 0,
		}
	}

	// Map finish reason
	finishReason := m.mapFinishReason(candidate.FinishReason, hasToolCalls)

	return &llm.Response{
		ID: r.ResponseID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}, nil
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
				// Regular text part - strip markdown code fences if present
				// (Gemini sometimes wraps JSON in ```json...``` even in JSON mode)
				text := stripMarkdownCodeFence(part.Text)
				content = append(content, llm.NewTextPart(text))
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

			content = append(content, llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        id,
				Name:      part.FunctionCall.Name,
				Arguments: argsJSON,
			}))

		case part.FunctionResponse != nil:
			// Tool response (function response)
			// Convert response to JSON
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, false, fmt.Errorf("failed to marshal function response: %w", err)
			}

			content = append(content, llm.NewToolResponsePart(&llm.ToolResponse{
				ID:     part.FunctionResponse.ID,
				Result: responseJSON,
			}))

		case part.ExecutableCode != nil:
			// Code execution - treat as text
			content = append(content, llm.NewTextPart(part.ExecutableCode.Code))

		case part.CodeExecutionResult != nil:
			// Code execution result - treat as text
			content = append(content, llm.NewTextPart(part.CodeExecutionResult.Output))

		default:
			// Skip unsupported part types (file data, inline data, etc.)
			continue
		}
	}

	return content, hasToolCalls, nil
}

// mapFinishReason converts Gemini's finish reason to our unified finish reason.
//

func (m *ResponseMapper) mapFinishReason(reason genai.FinishReason, hasToolCalls bool) llm.FinishReason {
	// If there are tool calls, the finish reason should be ToolCalls
	if hasToolCalls {
		return llm.FinishReasonToolCalls
	}

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

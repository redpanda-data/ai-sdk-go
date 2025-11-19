package openai

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3/responses"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper converts OpenAI Responses API payloads to llm.Response.
type ResponseMapper struct {
	modelDefinition ModelDefinition
}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper(definition ModelDefinition) *ResponseMapper {
	return &ResponseMapper{modelDefinition: definition}
}

// FromProvider converts an OpenAI Responses API payload into llm.Response.
func (m *ResponseMapper) FromProvider(r *responses.Response) (*llm.Response, error) {
	if r == nil {
		return nil, fmt.Errorf("%w: nil provider response", llm.ErrResponseMapping)
	}

	// 1. Authoritative and non-final statuses.
	switch r.Status {
	case responses.ResponseStatusFailed:
		// Surface a deterministic error even if the provider omitted details.
		if r.Error.Message == "" && r.Error.Code == "" {
			return nil, &llm.ProviderError{
				Base:    llm.ErrAPICall,
				Code:    "failed",
				Message: "provider status=failed without error payload",
			}
		}

		return nil, m.providerErrorFrom(r.Error)

	case responses.ResponseStatusInProgress, responses.ResponseStatusQueued:
		return nil, &llm.ProviderError{
			Base:    llm.ErrAPICall,
			Code:    string(r.Status),
			Message: "non-final provider status",
		}

	case responses.ResponseStatusCompleted, responses.ResponseStatusCancelled, responses.ResponseStatusIncomplete:
		// Final statuses - continue processing
	}

	// 2. Defensive: API response may stuff Error even if Status != failed.
	// Not yet observed, purely defensive.
	if r.Error.Message != "" {
		return nil, m.providerErrorFrom(r.Error)
	}

	// 3. No output with non-failed status is a provider-side issue (invalid response)
	if len(r.Output) == 0 {
		return nil, &llm.ProviderError{
			Base:    llm.ErrServerError,
			Message: "provider returned empty output on non-failed response",
		}
	}

	// 4. Collect content and detect tool calls.
	content := make([]*llm.Part, 0, len(r.Output))
	hasToolCalls := false

	for _, out := range r.Output {
		switch out.Type {
		case outputTypeMessage:
			for _, p := range out.Content {
				if p.Type == contentTypeOutputText && p.Text != "" {
					content = append(content, llm.NewTextPart(p.Text))
				}
			}

		case outputTypeFunctionCall:
			fc, ok := out.AsAny().(responses.ResponseFunctionToolCall)
			if !ok {
				return nil, fmt.Errorf("%w: function_call with unexpected shape (output id: %s)",
					llm.ErrResponseMapping, out.ID)
			}

			hasToolCalls = true

			content = append(content, llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        fc.CallID,
				Name:      fc.Name,
				Arguments: json.RawMessage(fc.Arguments),
			}))

		case outputTypeReasoning:
			for i, s := range out.Summary {
				if s.Text == "" {
					continue
				}

				content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
					ID:       fmt.Sprintf("%s-%d", out.ID, i),
					Text:     s.Text,
					Metadata: map[string]any{"summary_index": i},
				}))
			}

		default:
			return nil, fmt.Errorf("%w: unsupported output type %q (output id: %s)",
				llm.ErrResponseMapping, out.Type, out.ID)
		}
	}

	// 5. Usage extraction if available.
	var usage *llm.TokenUsage
	if r.Usage.TotalTokens > 0 {
		usage = &llm.TokenUsage{
			InputTokens:     int(r.Usage.InputTokens),
			OutputTokens:    int(r.Usage.OutputTokens),
			TotalTokens:     int(r.Usage.TotalTokens),
			CachedTokens:    int(r.Usage.InputTokensDetails.CachedTokens),
			ReasoningTokens: int(r.Usage.OutputTokensDetails.ReasoningTokens),
			MaxInputTokens:  m.modelDefinition.Constraints.MaxTokensLimit,
		}
	} else {
		// Even if TotalTokens is 0, provide usage structure with MaxInputTokens
		usage = &llm.TokenUsage{
			InputTokens:     int(r.Usage.InputTokens),
			OutputTokens:    int(r.Usage.OutputTokens),
			TotalTokens:     int(r.Usage.TotalTokens),
			CachedTokens:    int(r.Usage.InputTokensDetails.CachedTokens),
			ReasoningTokens: int(r.Usage.OutputTokensDetails.ReasoningTokens),
			MaxInputTokens:  m.modelDefinition.Constraints.MaxTokensLimit,
		}
	}

	// 6. Finish reason: tool calls take precedence.
	var finish llm.FinishReason
	if hasToolCalls {
		finish = llm.FinishReasonToolCalls
	} else {
		finish = m.mapFinishReasonFromStatus(string(r.Status), r.IncompleteDetails)
	}

	return &llm.Response{
		ID: r.ID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason: finish,
		Usage:        usage,
	}, nil
}

func (*ResponseMapper) providerErrorFrom(e responses.ResponseError) *llm.ProviderError {
	if base, ok := codeToBaseErr[e.Code]; ok {
		return &llm.ProviderError{
			Base:    base,
			Code:    string(e.Code),
			Message: e.Message,
		}
	}

	return &llm.ProviderError{
		Base:    llm.ErrAPICall,
		Code:    string(e.Code),
		Message: e.Message,
	}
}

// mapFinishReasonFromStatus maps provider status/incomplete details to our FinishReason.
func (*ResponseMapper) mapFinishReasonFromStatus(status string, inc responses.ResponseIncompleteDetails) llm.FinishReason {
	switch status {
	case "completed":
		return llm.FinishReasonStop
	case "incomplete":
		switch inc.Reason {
		case "max_output_tokens":
			return llm.FinishReasonLength
		case "content_filter":
			return llm.FinishReasonContentFilter
		default:
			return llm.FinishReasonUnknown
		}
	default:
		return llm.FinishReasonUnknown
	}
}

var codeToBaseErr = map[responses.ResponseErrorCode]error{
	responses.ResponseErrorCodeRateLimitExceeded:           llm.ErrRateLimitExceeded,
	responses.ResponseErrorCodeImageContentPolicyViolation: llm.ErrContentPolicyViolation,

	// Invalid input family
	responses.ResponseErrorCodeInvalidPrompt:             llm.ErrInvalidInput,
	responses.ResponseErrorCodeInvalidImage:              llm.ErrInvalidInput,
	responses.ResponseErrorCodeInvalidImageFormat:        llm.ErrInvalidInput,
	responses.ResponseErrorCodeInvalidBase64Image:        llm.ErrInvalidInput,
	responses.ResponseErrorCodeInvalidImageURL:           llm.ErrInvalidInput,
	responses.ResponseErrorCodeImageTooLarge:             llm.ErrInvalidInput,
	responses.ResponseErrorCodeImageTooSmall:             llm.ErrInvalidInput,
	responses.ResponseErrorCodeInvalidImageMode:          llm.ErrInvalidInput,
	responses.ResponseErrorCodeImageFileTooLarge:         llm.ErrInvalidInput,
	responses.ResponseErrorCodeUnsupportedImageMediaType: llm.ErrInvalidInput,
	responses.ResponseErrorCodeEmptyImageFile:            llm.ErrInvalidInput,

	// Server-ish family
	responses.ResponseErrorCodeServerError:           llm.ErrServerError,
	responses.ResponseErrorCodeVectorStoreTimeout:    llm.ErrServerError,
	responses.ResponseErrorCodeImageParseError:       llm.ErrServerError,
	responses.ResponseErrorCodeFailedToDownloadImage: llm.ErrServerError,
	responses.ResponseErrorCodeImageFileNotFound:     llm.ErrServerError,
}

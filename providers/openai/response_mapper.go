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

package openai

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3/responses"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ResponseMapper converts OpenAI Responses API payloads to llm.Response.
type ResponseMapper struct{}

// NewResponseMapper returns a ready-to-use mapper.
func NewResponseMapper() *ResponseMapper {
	return &ResponseMapper{}
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
	content := make([]llm.Part, 0, len(r.Output))
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

			content = append(content, llm.NewToolRequestPart(fc.CallID, fc.Name, normalizeToolArguments(fc.Arguments)))

		case outputTypeReasoning:
			for i, s := range out.Summary {
				if s.Text == "" {
					continue
				}

				content = append(content, &llm.ReasoningPart{
					ID:       fmt.Sprintf("%s-%d", out.ID, i),
					Text:     s.Text,
					Metadata: map[string]any{"summary_index": i},
				})
			}

		default:
			return nil, fmt.Errorf("%w: unsupported output type %q (output id: %s)",
				llm.ErrResponseMapping, out.Type, out.ID)
		}
	}

	// 5. Usage extraction. OpenAI reports cached_tokens and
	// cache_write_tokens as subsets of input_tokens, and reasoning_tokens as
	// a subset of output_tokens. The normalized llm.TokenUsage shape is
	// disjoint, so we un-subset them here.
	inputTotal := r.Usage.InputTokens
	cachedIn := r.Usage.InputTokensDetails.CachedTokens

	cacheWrite := r.Usage.InputTokensDetails.CacheWriteTokens
	if inputTotal < 0 || cachedIn < 0 || cacheWrite < 0 || cachedIn > inputTotal || cacheWrite > inputTotal-cachedIn {
		return nil, fmt.Errorf("%w: invalid input token usage: total=%d cached=%d cache_write=%d",
			llm.ErrResponseMapping, inputTotal, cachedIn, cacheWrite)
	}

	outputTotal := r.Usage.OutputTokens

	reasoning := r.Usage.OutputTokensDetails.ReasoningTokens
	if outputTotal < 0 || reasoning < 0 || reasoning > outputTotal {
		return nil, fmt.Errorf("%w: invalid output token usage: total=%d reasoning=%d",
			llm.ErrResponseMapping, outputTotal, reasoning)
	}

	usage := &llm.TokenUsage{
		InputTokens:                   int(inputTotal - cachedIn - cacheWrite),
		CachedInputTokens:             int(cachedIn),
		CacheCreationUnknownTTLTokens: int(cacheWrite),
		OutputTokens:                  int(outputTotal - reasoning),
		ReasoningTokens:               int(reasoning),
	}

	// 6. Finish reason. Truncation and filter signals must propagate through
	// to the caller; only upgrade a plain Stop to ToolCalls when tool use
	// blocks are present. See providers/anthropic/response_mapper.go for
	// the full rationale.
	finish := m.mapFinishReasonFromStatus(string(r.Status), r.IncompleteDetails)
	if hasToolCalls && finish == llm.FinishReasonStop {
		finish = llm.FinishReasonToolCalls
	}

	return &llm.Response{
		ID: r.ID,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: content,
		},
		FinishReason:   finish,
		Usage:          usage,
		ServiceTier:    llm.NormalizeServiceTier(string(r.ServiceTier)),
		InvokedModelID: resolveInvokedModelID(r.Model),
	}, nil
}

// resolveInvokedModelID collapses a provider-reported model ID (possibly a
// timestamped snapshot) to its catalog offering ID; IDs the catalog does
// not know pass through unchanged. Every provider applies this rule, so
// InvokedModelID has one meaning SDK-wide.
func resolveInvokedModelID(model string) string {
	if offering, ok := Catalog().Resolve(model); ok {
		return offering.ID
	}

	return model
}

func (*ResponseMapper) providerErrorFrom(e responses.ResponseError) *llm.ProviderError {
	// response.failed can carry context_length_exceeded, which the typed
	// ResponseErrorCode enum lacks.
	if isContextOverflow(string(e.Code), e.Message) {
		return &llm.ProviderError{
			Base:    llm.ErrContextOverflow,
			Code:    string(e.Code),
			Message: e.Message,
		}
	}

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

// normalizeToolArguments converts a raw function-call arguments string into
// a JSON object payload. OpenAI emits an empty arguments string for
// zero-parameter tool calls; passed through verbatim, downstream tool
// executors serialize the call as `"arguments": null`, which strict MCP
// servers reject. Mirrors the Bedrock mapper, which defaults absent tool
// input to {}.
func normalizeToolArguments(args string) json.RawMessage {
	if args == "" {
		return json.RawMessage(`{}`)
	}

	return json.RawMessage(args)
}

package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/openai/openai-go/v2"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

var _ llm.Model = (*Model)(nil)

// Model implements the llm.Model interface for OpenAI-compatible models using Chat Completion API.
type Model struct {
	provider       *Provider
	config         *Config
	capabilities   llm.ModelCapabilities
	client         *openai.Client
	requestMapper  *RequestMapper
	responseMapper *ResponseMapper
}

// Name returns the model identifier.
func (m *Model) Name() string {
	return m.config.ModelName
}

// Capabilities returns what features this model supports.
func (m *Model) Capabilities() llm.ModelCapabilities {
	return m.capabilities
}

// Generate performs a single, non-streaming request to the OpenAI Chat Completion API.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	// Convert our unified request to Chat Completion API format
	apiReq, err := m.requestMapper.ToProvider(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	// Make the API call using Chat Completion API
	response, err := m.client.Chat.Completions.New(ctx, apiReq)
	if err != nil {
		mappedErr := m.responseMapper.FromProviderError(err)
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, mappedErr)
	}

	// Convert Chat Completion API response back to our format
	return m.responseMapper.FromProvider(response)
}

// GenerateEvents performs a streaming request to the OpenAI Chat Completion API.
// It returns a Go 1.23+ iterator for streaming LLM responses.
func (m *Model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		// Convert our unified request to Chat Completion API format
		apiReq, err := m.requestMapper.ToProvider(req)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err))
			return
		}

		// Enable usage reporting in streaming mode
		apiReq.StreamOptions = openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.Bool(true),
		}

		// Create streaming request using Chat Completion API
		stream := m.client.Chat.Completions.NewStreaming(ctx, apiReq)
		defer stream.Close() // Automatic cleanup, even on early break

		var (
			accumulatedContent   strings.Builder
			accumulatedReasoning strings.Builder
			toolCalls            = make(map[int]*openai.ChatCompletionChunkChoiceDeltaToolCall)
			finalResponse        *openai.ChatCompletion
			usage                *openai.CompletionUsage
		)

		// Process streaming events
		for stream.Next() {
			chunk := stream.Current()

			// Accumulate usage from any chunk that has it
			// OpenAI sends usage in the last chunk (which has empty choices) when stream_options.include_usage is true
			if chunk.Usage.JSON.TotalTokens.Valid() && chunk.Usage.TotalTokens > 0 {
				usage = &chunk.Usage
			}

			// Skip further processing if no choices
			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta

			// Handle reasoning content deltas (for reasoning models like DeepSeek-R1, o1)
			if reasoningField, ok := delta.JSON.ExtraFields["reasoning_content"]; ok && reasoningField.Raw() != "" {
				var reasoningDelta string
				if err := json.Unmarshal([]byte(reasoningField.Raw()), &reasoningDelta); err == nil && reasoningDelta != "" {
					accumulatedReasoning.WriteString(reasoningDelta)

					if !yield(llm.ContentPartEvent{
						Index: int(choice.Index),
						Part:  llm.NewReasoningPart(&llm.ReasoningTrace{Text: reasoningDelta}),
					}, nil) {
						return // Early break, defer runs
					}
				}
			}

			// Handle text content deltas
			if delta.Content != "" {
				accumulatedContent.WriteString(delta.Content)

				if !yield(llm.ContentPartEvent{
					Index: int(choice.Index),
					Part:  llm.NewTextPart(delta.Content),
				}, nil) {
					return // Early break, defer runs
				}
			}

			// Handle tool call deltas
			for _, toolCallDelta := range delta.ToolCalls {
				idx := int(toolCallDelta.Index)

				// Initialize or get existing tool call
				if _, exists := toolCalls[idx]; !exists {
					toolCalls[idx] = &openai.ChatCompletionChunkChoiceDeltaToolCall{
						Index: toolCallDelta.Index,
					}
				}

				tc := toolCalls[idx]

				// Accumulate ID
				if toolCallDelta.ID != "" {
					tc.ID = toolCallDelta.ID
				}

				// Accumulate type
				if toolCallDelta.Type != "" {
					tc.Type = toolCallDelta.Type
				}

				// Accumulate function name and arguments
				if toolCallDelta.Function.Name != "" {
					tc.Function.Name = toolCallDelta.Function.Name
				}

				if toolCallDelta.Function.Arguments != "" {
					tc.Function.Arguments += toolCallDelta.Function.Arguments
				}
			}

			// Check for finish reason
			if choice.FinishReason != "" {
				finalResponse = m.buildFinalResponse(&chunk, choice.FinishReason, accumulatedReasoning.String(), accumulatedContent.String(), toolCalls)

				// Emit tool calls if present
				if !m.emitToolCalls(toolCalls, yield) {
					return
				}

				// Don't break yet - continue reading to get the usage chunk
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			mappedErr := m.responseMapper.FromProviderError(err)
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, mappedErr))

			return
		}

		// Emit final StreamEndEvent
		if finalResponse != nil {
			// Add final accumulated usage to response
			if usage != nil {
				finalResponse.Usage = *usage
			}

			// Build the final unified response with accumulated reasoning
			resp, err := m.buildStreamEndResponse(finalResponse, accumulatedReasoning.String())
			if err != nil {
				yield(nil, fmt.Errorf("%w: %w", llm.ErrResponseMapping, err))
				return
			}

			yield(llm.StreamEndEvent{Response: resp}, nil)
		} else {
			// No final response - stream ended without finish reason
			yield(nil, fmt.Errorf("%w: stream ended without finish reason", llm.ErrResponseMapping))
		}
	}
}

// buildFinalResponse constructs the final ChatCompletion response from accumulated data.
func (*Model) buildFinalResponse(chunk *openai.ChatCompletionChunk, finishReason, reasoningContent, content string, toolCalls map[int]*openai.ChatCompletionChunkChoiceDeltaToolCall) *openai.ChatCompletion {
	resp := &openai.ChatCompletion{
		ID:      chunk.ID,
		Object:  "openai.completion",
		Created: chunk.Created,
		Model:   chunk.Model,
		Choices: []openai.ChatCompletionChoice{
			{
				Index: 0,
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
	}

	// Add accumulated tool calls to final message
	if len(toolCalls) > 0 {
		var tcArray []openai.ChatCompletionMessageToolCallUnion

		for i := range len(toolCalls) {
			if tc, ok := toolCalls[i]; ok {
				tcArray = append(tcArray, openai.ChatCompletionMessageToolCallUnion{
					ID:   tc.ID,
					Type: tc.Type,
					Function: openai.ChatCompletionMessageFunctionToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				})
			}
		}

		resp.Choices[0].Message.ToolCalls = tcArray
	}

	return resp
}

// buildStreamEndResponse constructs the final unified response for StreamEndEvent,
// including accumulated reasoning content that was streamed separately.
func (m *Model) buildStreamEndResponse(apiResp *openai.ChatCompletion, accumulatedReasoning string) (*llm.Response, error) {
	if apiResp == nil || len(apiResp.Choices) == 0 {
		return nil, fmt.Errorf("%w: invalid final response", llm.ErrResponseMapping)
	}

	choice := apiResp.Choices[0]
	message := choice.Message

	// Build content parts: reasoning + text + tool calls
	content := make([]*llm.Part, 0, 2+len(message.ToolCalls))

	// Add accumulated reasoning if present
	if accumulatedReasoning != "" {
		content = append(content, llm.NewReasoningPart(&llm.ReasoningTrace{
			Text: accumulatedReasoning,
		}))
	}

	// Add text content if present
	if message.Content != "" {
		content = append(content, llm.NewTextPart(message.Content))
	}

	// Add tool calls if present
	for _, toolCall := range message.ToolCalls {
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
	finishReason, err := m.responseMapper.mapFinishReason(choice.FinishReason, len(message.ToolCalls) > 0)
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
		usage = &llm.TokenUsage{}
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

// emitToolCalls emits ContentPartEvent for each accumulated tool call.
func (*Model) emitToolCalls(toolCalls map[int]*openai.ChatCompletionChunkChoiceDeltaToolCall, yield func(llm.Event, error) bool) bool {
	for i := range len(toolCalls) {
		if tc, ok := toolCalls[i]; ok {
			toolPart := llm.Part{
				Kind: llm.PartToolRequest,
				ToolRequest: &llm.ToolRequest{
					ID:        tc.ID,
					Name:      tc.Function.Name,
					Arguments: json.RawMessage(tc.Function.Arguments),
				},
			}
			if !yield(llm.ContentPartEvent{
				Index: i,
				Part:  &toolPart,
			}, nil) {
				return false
			}
		}
	}

	return true
}

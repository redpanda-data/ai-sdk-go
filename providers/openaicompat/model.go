package openaicompat

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"

	"github.com/openai/openai-go/v3"

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

// Provider returns the provider name.
func (m *Model) Provider() string {
	return m.provider.Name()
}

// Capabilities returns what features this model supports.
func (m *Model) Capabilities() llm.ModelCapabilities {
	return m.capabilities
}

// Constraints returns the model's validation rules and limitations.
func (m *Model) Constraints() llm.ModelConstraints {
	return m.config.Constraints
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
		// Double-wrap: see anthropic/model.go Generate for rationale.
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err))
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
			toolCalls            = make(map[int]*llm.ToolRequest)
			finishReason         llm.FinishReason
			responseID           string
			usage                *llm.TokenUsage
		)

		// Process streaming events
		for stream.Next() {
			chunk := stream.Current()

			// Accumulate usage from any chunk that has it
			// OpenAI sends usage in the last chunk (which has empty choices) when stream_options.include_usage is true
			if chunk.Usage.JSON.TotalTokens.Valid() && chunk.Usage.TotalTokens > 0 {
				usage = &llm.TokenUsage{
					InputTokens:     int(chunk.Usage.PromptTokens),
					OutputTokens:    int(chunk.Usage.CompletionTokens),
					TotalTokens:     int(chunk.Usage.TotalTokens),
					ReasoningTokens: int(chunk.Usage.CompletionTokensDetails.ReasoningTokens),
				}
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
					toolCalls[idx] = &llm.ToolRequest{}
				}

				tc := toolCalls[idx]

				// Accumulate ID
				if toolCallDelta.ID != "" {
					tc.ID = toolCallDelta.ID
				}

				// Accumulate function name
				if toolCallDelta.Function.Name != "" {
					tc.Name = toolCallDelta.Function.Name
				}

				// Accumulate arguments
				if toolCallDelta.Function.Arguments != "" {
					tc.Arguments = append(tc.Arguments, []byte(toolCallDelta.Function.Arguments)...)
				}
			}

			// Check for finish reason
			if choice.FinishReason != "" {
				// Store response ID
				responseID = chunk.ID

				// Map finish reason to llm type
				mappedReason, err := m.responseMapper.mapFinishReason(choice.FinishReason, len(toolCalls) > 0)
				if err != nil {
					yield(nil, fmt.Errorf("%w: %w", llm.ErrResponseMapping, err))
					return
				}

				finishReason = mappedReason

				// Emit tool calls if present
				if !m.emitToolCalls(toolCalls, yield) {
					return
				}

				// Don't break yet - continue reading to get the usage chunk
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			// Double-wrap: see anthropic/model.go Generate for rationale.
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))

			return
		}

		// Emit final StreamEndEvent
		if finishReason != "" {
			// Build the final unified response from accumulated data
			resp := m.buildStreamEndResponse(
				responseID,
				accumulatedReasoning.String(),
				accumulatedContent.String(),
				toolCalls,
				finishReason,
				usage,
			)

			yield(llm.StreamEndEvent{Response: resp}, nil)
		} else {
			// No finish reason received - stream ended unexpectedly
			yield(nil, fmt.Errorf("%w: stream ended without finish reason", llm.ErrResponseMapping))
		}
	}
}

// buildStreamEndResponse constructs the final unified response for StreamEndEvent
// from accumulated streaming data.
func (*Model) buildStreamEndResponse(
	id string,
	reasoning, content string,
	toolCalls map[int]*llm.ToolRequest,
	finishReason llm.FinishReason,
	usage *llm.TokenUsage,
) *llm.Response {
	// Build content parts: reasoning + text + tool calls
	parts := make([]*llm.Part, 0, 2+len(toolCalls))

	// Add accumulated reasoning if present
	if reasoning != "" {
		parts = append(parts, llm.NewReasoningPart(&llm.ReasoningTrace{
			Text: reasoning,
		}))
	}

	// Add text content if present
	if content != "" {
		parts = append(parts, llm.NewTextPart(content))
	}

	// Add tool calls in index order
	for i := range len(toolCalls) {
		if tc, ok := toolCalls[i]; ok {
			parts = append(parts, llm.NewToolRequestPart(tc))
		}
	}

	// Ensure usage is not nil
	if usage == nil {
		usage = &llm.TokenUsage{}
	}

	return &llm.Response{
		ID: id,
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: parts,
		},
		FinishReason: finishReason,
		Usage:        usage,
	}
}

// emitToolCalls emits ContentPartEvent for each accumulated tool call.
func (*Model) emitToolCalls(toolCalls map[int]*llm.ToolRequest, yield func(llm.Event, error) bool) bool {
	for i := range len(toolCalls) {
		if tc, ok := toolCalls[i]; ok {
			if !yield(llm.ContentPartEvent{
				Index: i,
				Part:  llm.NewToolRequestPart(tc),
			}, nil) {
				return false
			}
		}
	}

	return true
}

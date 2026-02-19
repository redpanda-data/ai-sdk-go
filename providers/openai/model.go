package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

var _ llm.Model = (*Model)(nil)

// Model implements the llm.Model interface for OpenAI models.
type Model struct {
	provider       *Provider
	config         *Config
	definition     ModelDefinition
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
	return m.definition.Capabilities
}

// Constraints returns the model's validation rules and limitations.
func (m *Model) Constraints() llm.ModelConstraints {
	return m.definition.Constraints
}

// SupportedReasoningEfforts returns the reasoning efforts this model supports, in ascending order (safest/lowest first).
// Returns empty slice for non-reasoning models.
func (m *Model) SupportedReasoningEfforts() []ReasoningEffort {
	return m.definition.SupportedReasoningEfforts
}

// Generate performs a single, non-streaming request to the OpenAI Responses API.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	// Convert our unified request to Responses API format
	apiReq, err := m.requestMapper.ToProvider(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	// Make the API call using Responses API
	response, err := m.client.Responses.New(ctx, apiReq)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err))
	}

	// Convert Responses API response back to our format
	return m.responseMapper.FromProvider(response)
}

// GenerateEvents performs a streaming request to the OpenAI Responses API.
// It returns a Go 1.23+ iterator for streaming LLM responses.
func (m *Model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		// Convert our unified request to Responses API format
		apiReq, err := m.requestMapper.ToProvider(req)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err))
			return
		}

		// Create streaming request using Responses API
		stream := m.client.Responses.NewStreaming(ctx, apiReq)
		defer stream.Close() // Automatic cleanup, even on early break

		var finalResponse *responses.Response

		// Process streaming events
		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			// Text content chunks
			case streamEventOutputTextDelta:
				e := event.AsResponseOutputTextDelta()
				if !yield(llm.ContentPartEvent{
					Index: int(e.OutputIndex),
					Part:  llm.NewTextPart(e.Delta),
				}, nil) {
					return // Early break, defer runs
				}

			// Reasoning summary text deltas (incremental reasoning content)
			case streamEventReasoningSummaryTextDelta:
				e := event.AsResponseReasoningSummaryTextDelta()
				if e.Delta != "" {
					if !yield(llm.ContentPartEvent{
						Index: int(e.OutputIndex),
						Part: llm.NewReasoningPart(&llm.ReasoningTrace{
							ID:   e.ItemID,
							Text: e.Delta,
							Metadata: map[string]any{
								"sequence_number": e.SequenceNumber,
								"summary_index":   e.SummaryIndex,
							},
						}),
					}, nil) {
						return
					}
				}

			// Output item completion (tool calls)
			case streamEventOutputItemDone:
				e := event.AsResponseOutputItemDone()
				if toolCall, ok := e.Item.AsAny().(responses.ResponseFunctionToolCall); ok {
					if !yield(llm.ContentPartEvent{
						Index: int(e.OutputIndex),
						Part: llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        toolCall.CallID,
							Name:      toolCall.Name,
							Arguments: json.RawMessage(toolCall.Arguments),
						}),
					}, nil) {
						return
					}
				}

			// Recoverable errors (in-band)
			case streamEventError:
				e := event.AsError()
				if !yield(llm.ErrorEvent{
					Message: e.Message,
					Code:    e.Code,
				}, nil) {
					return
				}

			// Final response with usage and finish reason
			case streamEventResponseCompleted:
				e := event.AsResponseCompleted()
				finalResponse = &e.Response

			// Ignore other event types
			default:
				continue
			}
		}

		// Check for transport/cancellation errors
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))
			return
		}

		// Send final event with response
		if finalResponse != nil {
			completeResponse, err := m.responseMapper.FromProvider(finalResponse)
			if err != nil {
				yield(llm.StreamEndEvent{
					Error: fmt.Errorf("%w: %w", llm.ErrResponseMapping, err),
				}, nil)
			} else {
				yield(llm.StreamEndEvent{
					Response: completeResponse,
				}, nil)
			}
		}
	}
}

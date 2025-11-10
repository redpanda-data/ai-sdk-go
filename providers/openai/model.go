package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"

	"github.com/openai/openai-go/v2"
	"github.com/openai/openai-go/v2/packages/ssestream"
	"github.com/openai/openai-go/v2/responses"

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

// Capabilities returns what features this model supports.
func (m *Model) Capabilities() llm.ModelCapabilities {
	return m.definition.Capabilities
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
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, err)
	}

	// Convert Responses API response back to our format
	return m.responseMapper.FromProvider(response)
}

// GenerateStream performs a streaming request to the OpenAI Responses API.
func (m *Model) GenerateStream(ctx context.Context, req *llm.Request) (llm.EventStream, error) {
	// Convert our unified request to Responses API format
	apiReq, err := m.requestMapper.ToProvider(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	// Create streaming request using Responses API
	stream := m.client.Responses.NewStreaming(ctx, apiReq)

	return &responsesEventStream{
		stream: stream,
		ctx:    ctx,
	}, nil
}

// responsesEventStream implements llm.EventStream for Responses API streaming.
type responsesEventStream struct {
	stream         *ssestream.Stream[responses.ResponseStreamEventUnion]
	responseMapper ResponseMapper
	ctx            context.Context
	closed         bool

	// We store the final response from the 'response.completed' event to access usage and finish reason.
	finalResponse *responses.Response
	endEventSent  bool
}

// Recv returns the next streaming event from the Responses API.
func (s *responsesEventStream) Recv() (llm.StreamEvent, error) {
	if s.closed {
		return nil, llm.ErrStreamClosed
	}

	// Loop through stream events. We might receive several internal OpenAI events
	// before we get one that maps to our llm.StreamEvent abstraction.
	for s.stream.Next() {
		event := s.stream.Current()
		switch event.Type {
		// This event provides text chunks.
		case streamEventOutputTextDelta:
			e := event.AsResponseOutputTextDelta()

			return llm.ContentPartEvent{
				Index: int(e.OutputIndex),
				Part:  llm.NewTextPart(e.Delta),
			}, nil

		// Reasoning summary text deltas (incremental reasoning content)
		case streamEventReasoningSummaryTextDelta:
			// The SDK provides a typed struct for reasoning deltas
			e := event.AsResponseReasoningSummaryTextDelta()
			if e.Delta != "" {
				return llm.ContentPartEvent{
					Index: int(e.OutputIndex),
					Part: llm.NewReasoningPart(&llm.ReasoningTrace{
						ID:   e.ItemID,
						Text: e.Delta,
						Metadata: map[string]any{
							"sequence_number": e.SequenceNumber,
							"summary_index":   e.SummaryIndex,
						},
					}),
				}, nil
			}

			continue // Continue if delta is empty

		// This event signals that an output item (like a tool call) is complete
		case streamEventOutputItemDone:
			e := event.AsResponseOutputItemDone()
			// Check if this is a function call (tool request)
			if toolCall, ok := e.Item.AsAny().(responses.ResponseFunctionToolCall); ok {
				return llm.ContentPartEvent{
					Index: int(e.OutputIndex),
					Part: llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        toolCall.CallID,
						Name:      toolCall.Name,
						Arguments: json.RawMessage(toolCall.Arguments),
					}),
				}, nil
			}

			continue // Ignore non-tool-call output items

		// An error occurred during the stream.
		case streamEventError:
			e := event.AsError()

			return llm.ErrorEvent{
				Message: e.Message,
				Code:    e.Code,
			}, nil

		// This event contains the final response details, including usage and finish reason.
		// We capture it but don't end the stream yet. The stream ends when s.stream.Next() is false.
		case streamEventResponseCompleted:
			e := event.AsResponseCompleted()
			s.finalResponse = &e.Response

			continue // Continue to let the stream naturally end.

		// Ignore other event types that don't map to our abstraction (e.g., in_progress, searching).
		default:
			continue
		}
	}

	// After the loop, the stream has ended. Check for errors.
	err := s.stream.Err()
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, err)
	}

	// The stream is complete. Emit StreamEndEvent once, then EOF on the next call.
	if s.finalResponse != nil && !s.endEventSent {
		completeResponse, err := s.responseMapper.FromProvider(s.finalResponse)
		if err != nil {
			return llm.StreamEndEvent{
				Error: fmt.Errorf("failed mapping the final response to llm.Response: %w", err),
			}, nil
		}

		endEvent := llm.StreamEndEvent{
			Response: completeResponse,
		}
		s.endEventSent = true
		s.finalResponse = nil // Clear to prevent re-emission

		return endEvent, nil
	}

	// No more events
	return nil, io.EOF
}

// Close aborts the underlying stream.
func (s *responsesEventStream) Close() error {
	if s.closed {
		return nil
	}

	s.closed = true

	return s.stream.Close()
}

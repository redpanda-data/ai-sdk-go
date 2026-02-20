package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

var _ llm.Model = (*Model)(nil)

// Model implements the llm.Model interface for Anthropic models.
type Model struct {
	provider       *Provider
	config         *Config
	definition     ModelDefinition
	client         *anthropic.Client
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

// Generate performs a single, non-streaming request to the Anthropic Beta Messages API.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	// Convert our unified request to Anthropic Beta Messages API format
	apiReq, err := m.requestMapper.ToProvider(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	// Make the API call using Beta Messages API
	response, err := m.client.Beta.Messages.New(ctx, apiReq)
	if err != nil {
		// Double-wrap: ErrAPICall (for backward compat) + classified error (e.g.
		// ErrRateLimitExceeded) so both errors.Is checks work on the same error.
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err))
	}

	// Convert Beta Messages API response back to our format
	return m.responseMapper.FromProvider(response)
}

// GenerateEvents performs a streaming request to the Anthropic Beta Messages API.
// It returns a Go 1.23+ iterator for streaming LLM responses.
//
//nolint:gocyclo // Streaming event handling requires switching on multiple event types
func (m *Model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		// Convert our unified request to Anthropic Beta Messages API format
		apiReq, err := m.requestMapper.ToProvider(req)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err))
			return
		}

		// Create streaming request using Beta Messages API
		stream := m.client.Beta.Messages.NewStreaming(ctx, apiReq)
		defer stream.Close() // Automatic cleanup, even on early break

		// Track content blocks for aggregation
		contentBlocks := make(map[int]*contentBlockAccumulator)

		var finalMessage *anthropic.BetaMessage

		// Process streaming events
		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "message_start":
				// Message metadata - store for final response
				e := event.AsMessageStart()
				finalMessage = &e.Message

			case "content_block_start":
				// New content block starting
				e := event.AsContentBlockStart()
				acc := &contentBlockAccumulator{
					index: int(e.Index),
				}

				// Determine block type from Type field
				acc.blockType = e.ContentBlock.Type

				// For tool_use, save the initial data
				switch e.ContentBlock.Type {
				case blockTypeToolUse:
					acc.toolUse = &toolUseData{
						ID:   e.ContentBlock.ID,
						Name: e.ContentBlock.Name,
					}
				case blockTypeThinking:
					acc.thinkingSignature = e.ContentBlock.Signature
					acc.textContent = e.ContentBlock.Thinking
				case blockTypeText:
					acc.textContent = e.ContentBlock.Text
				}

				contentBlocks[int(e.Index)] = acc

			case "content_block_delta":
				// Content delta for a specific block
				e := event.AsContentBlockDelta()

				acc, ok := contentBlocks[int(e.Index)]
				if !ok {
					continue
				}

				// Handle different delta types based on Type field
				switch e.Delta.Type {
				case "text_delta":
					if e.Delta.Text != "" {
						// Text delta
						acc.textContent += e.Delta.Text
						if !yield(llm.ContentPartEvent{
							Index: int(e.Index),
							Part:  llm.NewTextPart(e.Delta.Text),
						}, nil) {
							return
						}
					}
				case "thinking_delta":
					if e.Delta.Thinking != "" {
						// Thinking delta (reasoning)
						acc.textContent += e.Delta.Thinking
						if !yield(llm.ContentPartEvent{
							Index: int(e.Index),
							Part: llm.NewReasoningPart(&llm.ReasoningTrace{
								ID:   acc.thinkingSignature,
								Text: e.Delta.Thinking,
							}),
						}, nil) {
							return
						}
					}
				case "signature_delta":
					if e.Delta.Signature != "" {
						// Signature for thinking block
						acc.thinkingSignature = e.Delta.Signature
					}
				case "input_json_delta":
					// Tool use arguments delta
					acc.toolArgs += e.Delta.PartialJSON
				}

			case "content_block_stop":
				// Content block completed
				e := event.AsContentBlockStop()

				acc, ok := contentBlocks[int(e.Index)]
				if !ok {
					continue
				}

				// For tool use blocks, emit the complete tool request
				if acc.blockType == blockTypeToolUse && acc.toolUse != nil {
					// Use accumulated args from input_json_delta events
					argsJSON := json.RawMessage(acc.toolArgs)
					if acc.toolArgs == "" {
						argsJSON = json.RawMessage("{}")
					}

					if !yield(llm.ContentPartEvent{
						Index: int(e.Index),
						Part: llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        acc.toolUse.ID,
							Name:      acc.toolUse.Name,
							Arguments: argsJSON,
						}),
					}, nil) {
						return
					}
				}

			case "message_delta":
				// Message-level metadata updates (usage, stop reason)
				e := event.AsMessageDelta()
				if finalMessage != nil {
					finalMessage.StopReason = e.Delta.StopReason
					if e.Usage.OutputTokens > 0 {
						finalMessage.Usage.OutputTokens = e.Usage.OutputTokens
					}
				}

			case "message_stop":
				// Stream completed successfully
				// Will be handled after loop

			default:
				// Unknown event type - ignore.
				// Note: prior to Anthropic SDK v1.22.1, SSE "error" events were delivered
				// as discrete stream events and handled here. Since v1.22.1 the SDK
				// converts them into stream.Err(), which we handle via classifyError below.
				continue
			}
		}

		// Check for transport/cancellation errors
		if err := stream.Err(); err != nil {
			// See Generate for ErrAPICall double-wrap rationale.
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))
			return
		}

		// Send final event with complete response
		if finalMessage != nil {
			// Build final content from accumulators
			finalContent := make([]anthropic.BetaContentBlockUnion, 0, len(contentBlocks))

			for i := range len(contentBlocks) {
				acc, ok := contentBlocks[i]
				if !ok {
					continue
				}

				switch acc.blockType {
				case blockTypeText:
					finalContent = append(finalContent, anthropic.BetaContentBlockUnion{
						Type: blockTypeText,
						Text: acc.textContent,
					})

				case blockTypeThinking:
					finalContent = append(finalContent, anthropic.BetaContentBlockUnion{
						Type:      blockTypeThinking,
						Thinking:  acc.textContent,
						Signature: acc.thinkingSignature,
					})

				case blockTypeToolUse:
					if acc.toolUse != nil {
						// Use accumulated toolArgs from input_json_delta events
						finalContent = append(finalContent, anthropic.BetaContentBlockUnion{
							Type:  blockTypeToolUse,
							ID:    acc.toolUse.ID,
							Name:  acc.toolUse.Name,
							Input: json.RawMessage(acc.toolArgs),
						})
					}
				}
			}

			finalMessage.Content = finalContent

			completeResponse, err := m.responseMapper.FromProvider(finalMessage)
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

// contentBlockAccumulator tracks state for a single content block during streaming.
type contentBlockAccumulator struct {
	index             int
	blockType         string // "text", "tool_use", "thinking"
	textContent       string
	toolArgs          string
	toolUse           *toolUseData // Store tool use data
	thinkingSignature string
}

// toolUseData stores tool use information during streaming.
type toolUseData struct {
	ID   string
	Name string
}

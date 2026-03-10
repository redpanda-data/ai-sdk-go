package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

var _ llm.Model = (*Model)(nil)

// Model implements the llm.Model interface for Bedrock models via the Converse API.
type Model struct {
	provider       *Provider
	config         *Config
	definition     ModelDefinition
	client         *bedrockruntime.Client
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

// Generate performs a single, non-streaming request using the Bedrock Converse API.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	input, err := m.requestMapper.ToConverseInput(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	output, err := m.client.Converse(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err))
	}

	return m.responseMapper.FromConverseOutput(output.StopReason, output.Output, output.Usage)
}

// GenerateEvents performs a streaming request using the Bedrock ConverseStream API.
// It returns a Go 1.23+ iterator for streaming LLM responses.
//
//nolint:gocyclo // Streaming event handling requires switching on multiple event types
func (m *Model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		input, err := m.requestMapper.ToConverseStreamInput(req)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err))
			return
		}

		output, err := m.client.ConverseStream(ctx, input)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))
			return
		}

		stream := output.GetStream()
		defer stream.Close()

		// Track content blocks for aggregation
		contentBlocks := make(map[int]*contentBlockAccumulator)

		var stopReason types.StopReason
		var tokenUsage *types.TokenUsage

		for event := range stream.Events() {
			switch e := event.(type) {
			case *types.ConverseStreamOutputMemberMessageStart:
				// Message started — role info only, nothing to emit

			case *types.ConverseStreamOutputMemberContentBlockStart:
				idx := 0
				if e.Value.ContentBlockIndex != nil {
					idx = int(*e.Value.ContentBlockIndex)
				}

				acc := &contentBlockAccumulator{index: idx}

				if e.Value.Start != nil {
					if toolStart, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
						acc.blockType = blockTypeToolUse
						acc.toolUse = &toolUseData{}
						if toolStart.Value.ToolUseId != nil {
							acc.toolUse.ID = *toolStart.Value.ToolUseId
						}
						if toolStart.Value.Name != nil {
							acc.toolUse.Name = *toolStart.Value.Name
						}
					}
				}

				contentBlocks[idx] = acc

			case *types.ConverseStreamOutputMemberContentBlockDelta:
				idx := 0
				if e.Value.ContentBlockIndex != nil {
					idx = int(*e.Value.ContentBlockIndex)
				}

				acc, ok := contentBlocks[idx]
				if !ok {
					acc = &contentBlockAccumulator{index: idx}
					contentBlocks[idx] = acc
				}

				if e.Value.Delta != nil {
					switch delta := e.Value.Delta.(type) {
					case *types.ContentBlockDeltaMemberText:
						if acc.blockType == "" {
							acc.blockType = blockTypeText
						}

						acc.textContent += delta.Value

						if !yield(llm.ContentPartEvent{
							Index: idx,
							Part:  llm.NewTextPart(delta.Value),
						}, nil) {
							return
						}

					case *types.ContentBlockDeltaMemberToolUse:
						if delta.Value.Input != nil {
							acc.toolArgs += *delta.Value.Input
						}

					case *types.ContentBlockDeltaMemberReasoningContent:
						if delta.Value != nil {
							switch rd := delta.Value.(type) {
							case *types.ReasoningContentBlockDeltaMemberText:
								if acc.blockType == "" {
									acc.blockType = blockTypeReasoning
								}

								acc.textContent += rd.Value

								// Signature arrives after all text deltas, so streaming
								// reasoning events carry an empty ID. The final assembled
								// part in buildFinalParts includes the signature.
								if !yield(llm.ContentPartEvent{
									Index: idx,
									Part: llm.NewReasoningPart(&llm.ReasoningTrace{
										Text: rd.Value,
									}),
								}, nil) {
									return
								}

							case *types.ReasoningContentBlockDeltaMemberSignature:
								acc.reasoningSignature = rd.Value
							}
						}
					}
				}

			case *types.ConverseStreamOutputMemberContentBlockStop:
				idx := 0
				if e.Value.ContentBlockIndex != nil {
					idx = int(*e.Value.ContentBlockIndex)
				}

				acc, ok := contentBlocks[idx]
				if !ok {
					continue
				}

				// For tool use blocks, emit the complete tool request
				if acc.blockType == blockTypeToolUse && acc.toolUse != nil {
					argsJSON := json.RawMessage(acc.toolArgs)
					if acc.toolArgs == "" {
						argsJSON = json.RawMessage("{}")
					}

					if !yield(llm.ContentPartEvent{
						Index: idx,
						Part: llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        acc.toolUse.ID,
							Name:      acc.toolUse.Name,
							Arguments: argsJSON,
						}),
					}, nil) {
						return
					}
				}

			case *types.ConverseStreamOutputMemberMessageStop:
				stopReason = e.Value.StopReason

			case *types.ConverseStreamOutputMemberMetadata:
				tokenUsage = e.Value.Usage
			}
		}

		// Check for stream errors
		if err := stream.Err(); err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))
			return
		}

		// Build final response from accumulated content
		finalParts := m.buildFinalParts(contentBlocks)

		var usage *llm.TokenUsage
		if tokenUsage != nil {
			usage = m.responseMapper.mapTokenUsage(tokenUsage)
		}

		var finishReason llm.FinishReason
		hasToolCalls := false
		for _, part := range finalParts {
			if part.IsToolRequest() {
				hasToolCalls = true

				break
			}
		}

		if hasToolCalls {
			finishReason = llm.FinishReasonToolCalls
		} else {
			finishReason = m.responseMapper.mapStopReason(stopReason)
		}

		yield(llm.StreamEndEvent{
			Response: &llm.Response{
				Message: llm.Message{
					Role:    llm.RoleAssistant,
					Content: finalParts,
				},
				FinishReason: finishReason,
				Usage:        usage,
			},
		}, nil)
	}
}

// buildFinalParts constructs the complete content from stream accumulators.
func (m *Model) buildFinalParts(blocks map[int]*contentBlockAccumulator) []*llm.Part {
	parts := make([]*llm.Part, 0, len(blocks))

	for i := range len(blocks) {
		acc, ok := blocks[i]
		if !ok {
			continue
		}

		switch acc.blockType {
		case blockTypeText:
			if acc.textContent != "" {
				parts = append(parts, llm.NewTextPart(acc.textContent))
			}

		case blockTypeToolUse:
			if acc.toolUse != nil {
				argsJSON := json.RawMessage(acc.toolArgs)
				if acc.toolArgs == "" {
					argsJSON = json.RawMessage("{}")
				}

				parts = append(parts, llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        acc.toolUse.ID,
					Name:      acc.toolUse.Name,
					Arguments: argsJSON,
				}))
			}

		case blockTypeReasoning:
			if acc.textContent != "" {
				parts = append(parts, llm.NewReasoningPart(&llm.ReasoningTrace{
					ID:   acc.reasoningSignature,
					Text: acc.textContent,
				}))
			}
		}
	}

	return parts
}

const (
	blockTypeText      = "text"
	blockTypeToolUse   = "tool_use"
	blockTypeReasoning = "reasoning"
)

// contentBlockAccumulator tracks state for a single content block during streaming.
type contentBlockAccumulator struct {
	index              int
	blockType          string
	textContent        string
	toolArgs           string
	toolUse            *toolUseData
	reasoningSignature string
}

// toolUseData stores tool use information during streaming.
type toolUseData struct {
	ID   string
	Name string
}

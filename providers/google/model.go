package google

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"iter"

	"github.com/google/uuid"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

var _ llm.Model = (*Model)(nil)

// Model implements the llm.Model interface for Google Gemini models.
type Model struct {
	provider       *Provider
	config         *Config
	definition     ModelDefinition
	client         *genai.Client
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

// Generate performs a single, non-streaming request to the Google API.
func (m *Model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	// Convert our unified request to Google API format
	contents, config, err := m.requestMapper.ToProvider(req)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err)
	}

	// Determine which model name to use (custom override if set, otherwise the configured model)
	modelName := m.config.ModelName
	if m.config.CustomModelName != "" {
		modelName = m.config.CustomModelName
	}

	// Make the API call
	response, err := m.client.Models.GenerateContent(ctx, modelName, contents, config)
	if err != nil {
		// Double-wrap: see anthropic/model.go Generate for rationale.
		return nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err))
	}

	// Convert Gemini response back to our format
	return m.responseMapper.FromProvider(response)
}

// GenerateEvents performs a streaming request to the Google API.
// It returns a Go 1.23+ iterator for streaming LLM responses.
func (m *Model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		// Convert our unified request to Google API format
		contents, config, err := m.requestMapper.ToProvider(req)
		if err != nil {
			yield(nil, fmt.Errorf("%w: %w", llm.ErrRequestMapping, err))
			return
		}

		// Determine which model name to use (custom override if set, otherwise the configured model)
		modelName := m.config.ModelName
		if m.config.CustomModelName != "" {
			modelName = m.config.CustomModelName
		}

		// Create streaming request
		stream := m.client.Models.GenerateContentStream(ctx, modelName, contents, config)

		// Track accumulated content for final response
		var allParts []*genai.Part
		var finalResponse *genai.GenerateContentResponse

		// Process streaming events
		for response, err := range stream {
			if err != nil {
				// Double-wrap: see anthropic/model.go Generate for rationale.
				yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, classifyError(err)))
				return
			}

			// Store the last response for final event
			finalResponse = response

			// Process each candidate (typically only one)
			if len(response.Candidates) == 0 {
				continue
			}

			candidate := response.Candidates[0]
			if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
				continue
			}

			// Accumulate parts for final response
			allParts = append(allParts, candidate.Content.Parts...)

			// Process each part and emit events
			for idx, part := range candidate.Content.Parts {
				var event llm.Event

				switch {
				case part.Text != "":
					if part.Thought {
						// Thinking/reasoning part
						var signature string
						if len(part.ThoughtSignature) > 0 {
							signature = string(part.ThoughtSignature)
						}

						event = llm.ContentPartEvent{
							Index: idx,
							Part: llm.NewReasoningPart(&llm.ReasoningTrace{
								ID:   signature,
								Text: part.Text,
							}),
						}
					} else {
						// Regular text delta
						event = llm.ContentPartEvent{
							Index: idx,
							Part:  llm.NewTextPart(part.Text),
						}
					}

				case part.FunctionCall != nil:
					// Tool call event
					// Convert arguments to JSON
					argsJSON, err := m.marshalFunctionArgs(part.FunctionCall.Args)
					if err != nil {
						yield(llm.ErrorEvent{
							Message: fmt.Sprintf("failed to marshal function call args: %v", err),
						}, nil)

						return
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

					event = llm.ContentPartEvent{
						Index: idx,
						Part:  toolPart,
					}

				default:
					continue
				}

				if !yield(event, nil) {
					return
				}
			}
		}

		// Send final event with complete response
		if finalResponse != nil {
			// Reconstruct the final response with all accumulated parts
			reconstructed := &genai.GenerateContentResponse{
				ResponseID:    finalResponse.ResponseID,
				UsageMetadata: finalResponse.UsageMetadata,
				Candidates:    finalResponse.Candidates,
			}

			// Replace the candidate's content with accumulated parts
			if len(reconstructed.Candidates) > 0 && reconstructed.Candidates[0].Content != nil {
				reconstructed.Candidates[0].Content.Parts = allParts
			}

			completeResponse, err := m.responseMapper.FromProvider(reconstructed)
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

// marshalFunctionArgs is a helper to marshal function arguments/responses.
func (m *Model) marshalFunctionArgs(args map[string]any) ([]byte, error) {
	if args == nil {
		return []byte("{}"), nil
	}

	return json.Marshal(args)
}

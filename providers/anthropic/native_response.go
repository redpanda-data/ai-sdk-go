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

package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// outboundMessage is the Anthropic message response format for serialization.
type outboundMessage struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Role       string                 `json:"role"`
	Model      string                 `json:"model"`
	Content    []outboundContentBlock `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      outboundUsage          `json:"usage"`
}

// outboundContentBlock is a content block in an outbound Anthropic response.
type outboundContentBlock struct {
	Type string `json:"type"`

	// text block
	Text string `json:"text,omitempty"`

	// tool_use block
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// thinking block
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// outboundUsage is the Anthropic usage format for serialization.
type outboundUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// outboundSSEEvent wraps a streaming SSE event payload.
type outboundSSEEvent struct {
	Type string `json:"type"`

	// message_start
	Message *outboundMessage `json:"message,omitempty"`

	// content_block_start
	Index        *int                  `json:"index,omitempty"`
	ContentBlock *outboundContentBlock `json:"content_block,omitempty"`

	// content_block_delta
	Delta *outboundDelta `json:"delta,omitempty"`

	// message_delta
	Usage *outboundUsage `json:"usage,omitempty"`
}

// outboundDelta is a delta payload in a content_block_delta or message_delta event.
type outboundDelta struct {
	Type       string `json:"type,omitempty"`
	Text       string `json:"text,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
}

// nativeResponseBody is the Anthropic message response format for deserialization.
// Separate from outboundMessage to support fields only present in incoming responses
// (e.g. cache token counts).
type nativeResponseBody struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	Role       string                 `json:"role"`
	Model      string                 `json:"model"`
	Content    []outboundContentBlock `json:"content"`
	StopReason string                 `json:"stop_reason"`
	Usage      nativeResponseUsage    `json:"usage"`
}

type nativeResponseUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

// ResponseFromNative parses an Anthropic /v1/messages response JSON body
// into a unified llm.Response and extracts the model name.
func ResponseFromNative(body []byte) (*llm.Response, string, error) {
	var nr nativeResponseBody
	if err := json.Unmarshal(body, &nr); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal native response: %w", err)
	}

	resp := &llm.Response{
		ID:           nr.ID,
		FinishReason: nativeStopReasonToFinish(nr.StopReason),
		Usage: &llm.TokenUsage{
			InputTokens:  nr.Usage.InputTokens,
			OutputTokens: nr.Usage.OutputTokens,
			TotalTokens:  nr.Usage.InputTokens + nr.Usage.OutputTokens,
			CachedTokens: nr.Usage.CacheReadInputTokens,
		},
	}

	var parts []*llm.Part
	for _, block := range nr.Content {
		switch block.Type {
		case blockTypeText:
			parts = append(parts, llm.NewTextPart(block.Text))
		case blockTypeToolUse:
			parts = append(parts, llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: block.Input,
			}))
		case blockTypeThinking:
			parts = append(parts, llm.NewReasoningPart(&llm.ReasoningTrace{
				ID:   block.Signature,
				Text: block.Thinking,
			}))
		}
	}

	resp.Message = llm.NewMessage(llm.RoleAssistant, parts...)

	return resp, nr.Model, nil
}

// nativeStopReasonToFinish maps Anthropic's stop_reason to a unified FinishReason.
func nativeStopReasonToFinish(reason string) llm.FinishReason {
	switch reason {
	case "end_turn":
		return llm.FinishReasonStop
	case "max_tokens":
		return llm.FinishReasonLength
	case "tool_use":
		return llm.FinishReasonToolCalls
	case "refusal":
		return llm.FinishReasonContentFilter
	default:
		return llm.FinishReasonStop
	}
}

// ResponseToNative serializes a unified llm.Response to Anthropic's native JSON format.
func ResponseToNative(resp *llm.Response, model string) ([]byte, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil response")
	}

	msg := outboundMessage{
		ID:         resp.ID,
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		StopReason: finishReasonToNative(resp.FinishReason),
	}

	if resp.Usage != nil {
		msg.Usage = outboundUsage{
			InputTokens:  resp.Usage.InputTokens,
			OutputTokens: resp.Usage.OutputTokens,
		}
	}

	content := make([]outboundContentBlock, 0, len(resp.Message.Content))
	for _, part := range resp.Message.Content {
		switch {
		case part.IsText():
			content = append(content, outboundContentBlock{
				Type: blockTypeText,
				Text: part.Text,
			})

		case part.IsToolRequest() && part.ToolRequest != nil:
			content = append(content, outboundContentBlock{
				Type:  blockTypeToolUse,
				ID:    part.ToolRequest.ID,
				Name:  part.ToolRequest.Name,
				Input: part.ToolRequest.Arguments,
			})

		case part.IsReasoning() && part.ReasoningTrace != nil:
			content = append(content, outboundContentBlock{
				Type:      blockTypeThinking,
				Thinking:  part.ReasoningTrace.Text,
				Signature: part.ReasoningTrace.ID,
			})
		}
	}

	msg.Content = content

	return json.Marshal(msg)
}

// EventToNative serializes a single llm.Event to one or more Anthropic SSE data payloads.
// Some llm events map to multiple SSE events (e.g., StreamEndEvent produces
// message_delta + message_stop).
func EventToNative(event llm.Event, model string) ([][]byte, error) {
	if event == nil {
		return nil, fmt.Errorf("nil event")
	}

	switch e := event.(type) {
	case llm.ContentPartEvent:
		return contentPartEventToNative(e)

	case llm.StreamEndEvent:
		return streamEndEventToNative(e, model)

	default:
		return nil, fmt.Errorf("unsupported event type: %T", event)
	}
}

func contentPartEventToNative(e llm.ContentPartEvent) ([][]byte, error) {
	if e.Part == nil {
		return nil, fmt.Errorf("nil part in content event")
	}

	switch {
	case e.Part.IsText():
		evt := outboundSSEEvent{
			Type:  "content_block_delta",
			Index: intPtr(e.Index),
			Delta: &outboundDelta{
				Type: "text_delta",
				Text: e.Part.Text,
			},
		}

		data, err := json.Marshal(evt)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil

	case e.Part.IsToolRequest() && e.Part.ToolRequest != nil:
		// Tool calls arrive complete in llm, emit as content_block_start
		block := &outboundContentBlock{
			Type:  blockTypeToolUse,
			ID:    e.Part.ToolRequest.ID,
			Name:  e.Part.ToolRequest.Name,
			Input: e.Part.ToolRequest.Arguments,
		}

		evt := outboundSSEEvent{
			Type:         "content_block_start",
			Index:        intPtr(e.Index),
			ContentBlock: block,
		}

		data, err := json.Marshal(evt)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil

	case e.Part.IsReasoning() && e.Part.ReasoningTrace != nil:
		evt := outboundSSEEvent{
			Type:  "content_block_delta",
			Index: intPtr(e.Index),
			Delta: &outboundDelta{
				Type: "thinking_delta",
				Text: e.Part.ReasoningTrace.Text,
			},
		}

		data, err := json.Marshal(evt)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil

	default:
		return nil, fmt.Errorf("unsupported part kind: %s", e.Part.Kind)
	}
}

func streamEndEventToNative(e llm.StreamEndEvent, model string) ([][]byte, error) {
	var results [][]byte

	if e.Response != nil {
		delta := outboundSSEEvent{
			Type: "message_delta",
			Delta: &outboundDelta{
				Type:       "message_delta",
				StopReason: finishReasonToNative(e.Response.FinishReason),
			},
		}

		if e.Response.Usage != nil {
			delta.Usage = &outboundUsage{
				InputTokens:  e.Response.Usage.InputTokens,
				OutputTokens: e.Response.Usage.OutputTokens,
			}
		}

		data, err := json.Marshal(delta)
		if err != nil {
			return nil, err
		}

		results = append(results, data)
	}

	// message_stop is always the final event
	stop := outboundSSEEvent{Type: "message_stop"}

	data, err := json.Marshal(stop)
	if err != nil {
		return nil, err
	}

	results = append(results, data)

	return results, nil
}

// finishReasonToNative maps a unified FinishReason to Anthropic's stop_reason string.
func finishReasonToNative(reason llm.FinishReason) string {
	switch reason {
	case llm.FinishReasonStop:
		return "end_turn"
	case llm.FinishReasonLength:
		return "max_tokens"
	case llm.FinishReasonToolCalls:
		return "tool_use"
	case llm.FinishReasonContentFilter:
		return "refusal"
	default:
		return "end_turn"
	}
}

func intPtr(i int) *int {
	return &i
}

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

package google

import (
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Response-side native types for JSON serialization.

type nativeRespBody struct {
	Candidates    []nativeRespCandidate `json:"candidates"`
	UsageMetadata *nativeRespUsage      `json:"usageMetadata,omitempty"`
	ModelVersion  string                `json:"modelVersion,omitempty"`
}

type nativeRespCandidate struct {
	Content      *nativeRespContent `json:"content"`
	FinishReason string             `json:"finishReason,omitempty"`
	Index        int                `json:"index"`
}

type nativeRespContent struct {
	Parts []nativeRespPart `json:"parts"`
	Role  string           `json:"role"`
}

type nativeRespPart struct {
	Text         string                `json:"text,omitempty"`
	FunctionCall *nativeRespFuncCall   `json:"functionCall,omitempty"`
	Thought      bool                  `json:"thought,omitempty"`
}

type nativeRespFuncCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type nativeRespUsage struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// ResponseToNative serializes a unified llm.Response to Gemini's native JSON format.
func ResponseToNative(resp *llm.Response, model string) ([]byte, error) {
	if resp == nil {
		return nil, fmt.Errorf("nil response")
	}

	candidate := nativeRespCandidate{
		Index:        0,
		FinishReason: finishReasonToGemini(resp.FinishReason),
		Content: &nativeRespContent{
			Role: "model",
		},
	}

	parts := make([]nativeRespPart, 0, len(resp.Message.Content))
	for _, part := range resp.Message.Content {
		switch {
		case part.IsText():
			parts = append(parts, nativeRespPart{
				Text: part.Text,
			})

		case part.IsToolRequest() && part.ToolRequest != nil:
			var args map[string]any
			if len(part.ToolRequest.Arguments) > 0 {
				if err := json.Unmarshal(part.ToolRequest.Arguments, &args); err != nil {
					return nil, fmt.Errorf("failed to unmarshal tool request args: %w", err)
				}
			}
			parts = append(parts, nativeRespPart{
				FunctionCall: &nativeRespFuncCall{
					Name: part.ToolRequest.Name,
					Args: args,
				},
			})

		case part.IsReasoning() && part.ReasoningTrace != nil:
			parts = append(parts, nativeRespPart{
				Thought: true,
				Text:    part.ReasoningTrace.Text,
			})
		}
	}

	candidate.Content.Parts = parts

	body := nativeRespBody{
		Candidates:   []nativeRespCandidate{candidate},
		ModelVersion: model,
	}

	if resp.Usage != nil {
		body.UsageMetadata = &nativeRespUsage{
			PromptTokenCount:     resp.Usage.InputTokens,
			CandidatesTokenCount: resp.Usage.OutputTokens,
			TotalTokenCount:      resp.Usage.TotalTokens,
		}
	}

	return json.Marshal(body)
}

// EventToNative serializes a single llm.Event to one or more Gemini SSE data payloads.
// Gemini streaming returns individual response objects (same structure as non-streaming).
func EventToNative(event llm.Event, model string) ([][]byte, error) {
	if event == nil {
		return nil, fmt.Errorf("nil event")
	}

	switch e := event.(type) {
	case llm.ContentPartEvent:
		return contentPartEventToGemini(e)

	case llm.StreamEndEvent:
		return streamEndEventToGemini(e, model)

	default:
		return nil, fmt.Errorf("unsupported event type: %T", event)
	}
}

func contentPartEventToGemini(e llm.ContentPartEvent) ([][]byte, error) {
	if e.Part == nil {
		return nil, fmt.Errorf("nil part in content event")
	}

	var part nativeRespPart

	switch {
	case e.Part.IsText():
		part = nativeRespPart{Text: e.Part.Text}

	case e.Part.IsToolRequest() && e.Part.ToolRequest != nil:
		var args map[string]any
		if len(e.Part.ToolRequest.Arguments) > 0 {
			if err := json.Unmarshal(e.Part.ToolRequest.Arguments, &args); err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool request args: %w", err)
			}
		}
		part = nativeRespPart{
			FunctionCall: &nativeRespFuncCall{
				Name: e.Part.ToolRequest.Name,
				Args: args,
			},
		}

	case e.Part.IsReasoning() && e.Part.ReasoningTrace != nil:
		part = nativeRespPart{
			Thought: true,
			Text:    e.Part.ReasoningTrace.Text,
		}

	default:
		return nil, fmt.Errorf("unsupported part kind: %s", e.Part.Kind)
	}

	body := nativeRespBody{
		Candidates: []nativeRespCandidate{
			{
				Content: &nativeRespContent{
					Parts: []nativeRespPart{part},
					Role:  "model",
				},
				Index: 0,
			},
		},
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return [][]byte{data}, nil
}

func streamEndEventToGemini(e llm.StreamEndEvent, model string) ([][]byte, error) {
	if e.Response == nil {
		// No response info available, emit empty candidate with STOP
		body := nativeRespBody{
			Candidates: []nativeRespCandidate{
				{
					Content: &nativeRespContent{
						Parts: []nativeRespPart{},
						Role:  "model",
					},
					FinishReason: "STOP",
					Index:        0,
				},
			},
			ModelVersion: model,
		}

		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}

		return [][]byte{data}, nil
	}

	candidate := nativeRespCandidate{
		Content: &nativeRespContent{
			Parts: []nativeRespPart{},
			Role:  "model",
		},
		FinishReason: finishReasonToGemini(e.Response.FinishReason),
		Index:        0,
	}

	body := nativeRespBody{
		Candidates:   []nativeRespCandidate{candidate},
		ModelVersion: model,
	}

	if e.Response.Usage != nil {
		body.UsageMetadata = &nativeRespUsage{
			PromptTokenCount:     e.Response.Usage.InputTokens,
			CandidatesTokenCount: e.Response.Usage.OutputTokens,
			TotalTokenCount:      e.Response.Usage.TotalTokens,
		}
	}

	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	return [][]byte{data}, nil
}

// finishReasonToGemini maps a unified FinishReason to Gemini's finishReason string.
func finishReasonToGemini(reason llm.FinishReason) string {
	switch reason {
	case llm.FinishReasonStop:
		return "STOP"
	case llm.FinishReasonLength:
		return "MAX_TOKENS"
	case llm.FinishReasonToolCalls:
		return "STOP" // Gemini doesn't have a specific tool_calls reason
	case llm.FinishReasonContentFilter:
		return "SAFETY"
	default:
		return "STOP"
	}
}

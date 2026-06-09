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

package llm

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"
)

// Part represents a single piece of content inside a Message or streamed
// ContentPartEvent. It is a sealed discriminated union: only the concrete
// pointer types declared in this package satisfy the interface.
//
// Pointer receivers are intentional. Only the *Part forms (e.g. *TextPart)
// satisfy Part — this prevents the footgun where both &TextPart{} and
// TextPart{} satisfy the interface but type switches only match one form,
// silently dropping the other.
//
// Consumers should type switch on the concrete types:
//
//	for _, p := range msg.Content {
//	    switch p := p.(type) {
//	    case *llm.TextPart:
//	        // p.Text
//	    case *llm.ToolRequestPart:
//	        // p.ID, p.Name, p.Arguments
//	    case *llm.ToolResponsePart:
//	        // p.ID, p.Result, p.IsError
//	    case *llm.ReasoningPart:
//	        // p.Text, p.Signature
//	    }
//	}
type Part interface {
	isPart()
}

// TextPart contains plain text content.
type TextPart struct {
	Text string `json:"text,omitempty"`
}

// ToolRequestPart contains a request from the model to execute a tool.
type ToolRequestPart struct {
	// ID is a unique identifier for this tool request within the conversation.
	ID string `json:"id"`
	// Name is the name of the tool to execute.
	Name string `json:"name"`
	// Arguments contains the tool input as JSON. The structure depends on the
	// tool's input schema.
	Arguments json.RawMessage `json:"arguments,omitempty"`
	// Metadata carries provider-specific context that must round-trip back
	// to the provider on the next request (e.g. Gemini 3 thought signatures).
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ToolResponsePart contains the result of executing a tool.
type ToolResponsePart struct {
	// ID matches the ID from the corresponding ToolRequestPart.
	ID string `json:"id"`
	// Name is the name of the tool that was executed.
	Name string `json:"name"`
	// Result contains the tool output as JSON. The structure depends on the
	// tool's output schema. When IsError is true, Result may be empty.
	Result json.RawMessage `json:"result,omitempty"`
	// IsError signals tool execution failure. When true, Result is typically
	// an error payload (e.g. {"error":"..."}).
	IsError bool `json:"is_error,omitempty"`
}

// ReasoningPart contains reasoning thoughts/traces from the model. This is the
// model's internal reasoning process, which may be exposed or summarized
// depending on the provider and configuration.
type ReasoningPart struct {
	// Text contains the reasoning content as text. For streaming responses,
	// this may be built up incrementally.
	Text string `json:"text,omitempty"`
	// Signature is provider-specific data attached to the trace (e.g.
	// Anthropic thinking signatures, OpenAI item IDs). Empty for providers
	// that do not emit one.
	Signature string `json:"signature,omitempty"`
	// Metadata carries provider-specific context (obfuscation keys, summary
	// indexes, redaction flags, etc.).
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewTextPart returns a *TextPart containing the given text.
func NewTextPart(text string) *TextPart {
	return &TextPart{Text: text}
}

// NewToolRequestPart returns a *ToolRequestPart with the given fields.
func NewToolRequestPart(id, name string, args json.RawMessage) *ToolRequestPart {
	return &ToolRequestPart{ID: id, Name: name, Arguments: args}
}

// NewToolResponsePart returns a *ToolResponsePart with a successful result.
func NewToolResponsePart(id, name string, result json.RawMessage) *ToolResponsePart {
	return &ToolResponsePart{ID: id, Name: name, Result: result}
}

// NewToolErrorPart returns a *ToolResponsePart representing a failed tool
// execution. The error message is encoded as a JSON object {"error": ...}
// for compatibility with provider tool_result block formats.
func NewToolErrorPart(id, name, message string) *ToolResponsePart {
	payload, err := json.Marshal(map[string]string{"error": message})
	if err != nil {
		// json.Marshal of map[string]string never fails in practice; fall
		// back to a literal to keep the API total.
		payload = json.RawMessage(`{"error":""}`)
	}

	return &ToolResponsePart{ID: id, Name: name, Result: payload, IsError: true}
}

// NewReasoningPart returns a *ReasoningPart with the given text.
func NewReasoningPart(text string) *ReasoningPart {
	return &ReasoningPart{Text: text}
}

// isPart marker methods. Pointer receivers seal the interface to *Part forms.

func (*TextPart) isPart()         {}
func (*ToolRequestPart) isPart()  {}
func (*ToolResponsePart) isPart() {}
func (*ReasoningPart) isPart()    {}

// JoinTextParts concatenates the Text field of every *TextPart in the slice.
// Non-text parts are ignored.
func JoinTextParts(parts []Part) string {
	var b strings.Builder

	for _, part := range parts {
		if tp, ok := part.(*TextPart); ok && tp != nil {
			b.WriteString(tp.Text)
		}
	}

	return b.String()
}

// partKind discriminator used in MarshalPart / UnmarshalPart envelopes.
const (
	partKindText         = "text"
	partKindToolRequest  = "tool_request"
	partKindToolResponse = "tool_response"
	partKindReasoning    = "reasoning"
)

// MarshalPart encodes a Part as a flat JSON envelope with a "type" field.
// Typed-nil receivers (e.g. var p *TextPart; var part Part = p) marshal as
// JSON null rather than panicking.
//

func MarshalPart(p Part) ([]byte, error) {
	switch v := p.(type) {
	case nil:
		return []byte("null"), nil

	case *TextPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		}{Type: partKindText, Text: v.Text})

	case *ToolRequestPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type      string          `json:"type"`
			ID        string          `json:"id"`
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments,omitempty"`
			Metadata  map[string]any  `json:"metadata,omitempty"`
		}{Type: partKindToolRequest, ID: v.ID, Name: v.Name, Arguments: v.Arguments, Metadata: v.Metadata})

	case *ToolResponsePart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type    string          `json:"type"`
			ID      string          `json:"id"`
			Name    string          `json:"name"`
			Result  json.RawMessage `json:"result,omitempty"`
			IsError bool            `json:"is_error,omitempty"`
		}{Type: partKindToolResponse, ID: v.ID, Name: v.Name, Result: v.Result, IsError: v.IsError})

	case *ReasoningPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type      string         `json:"type"`
			Text      string         `json:"text,omitempty"`
			Signature string         `json:"signature,omitempty"`
			Metadata  map[string]any `json:"metadata,omitempty"`
		}{Type: partKindReasoning, Text: v.Text, Signature: v.Signature, Metadata: v.Metadata})

	default:
		return nil, fmt.Errorf("llm: cannot marshal unknown Part type %T", p)
	}
}

// UnmarshalPart decodes a Part from the flat envelope produced by MarshalPart.
// A JSON null returns (nil, nil) — the caller may keep or filter the nil slot
// based on context.
func UnmarshalPart(data []byte) (Part, error) {
	trim := strings.TrimSpace(string(data))
	if trim == "" || trim == "null" {
		return nil, nil //nolint:nilnil // null Parts are valid wire values
	}

	var head struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(data, &head); err != nil {
		return nil, fmt.Errorf("llm: decode Part envelope: %w", err)
	}

	switch head.Type {
	case partKindText:
		var v TextPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode TextPart: %w", err)
		}

		return &v, nil

	case partKindToolRequest:
		var v ToolRequestPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ToolRequestPart: %w", err)
		}

		return &v, nil

	case partKindToolResponse:
		var v ToolResponsePart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ToolResponsePart: %w", err)
		}

		return &v, nil

	case partKindReasoning:
		var v ReasoningPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ReasoningPart: %w", err)
		}

		return &v, nil

	default:
		return nil, fmt.Errorf("llm: unknown Part type %q", head.Type)
	}
}

// ClonePart returns a deep copy of p. Typed-nil and untyped nil receivers
// return nil. Use this at session-persistence boundaries to ensure callers
// cannot mutate persisted Part fields through shared pointers.
func ClonePart(p Part) Part {
	switch v := p.(type) {
	case nil:
		return nil

	case *TextPart:
		if v == nil {
			return nil
		}

		clone := *v

		return &clone

	case *ToolRequestPart:
		if v == nil {
			return nil
		}

		clone := *v
		if v.Arguments != nil {
			clone.Arguments = append(json.RawMessage(nil), v.Arguments...)
		}

		if v.Metadata != nil {
			clone.Metadata = maps.Clone(v.Metadata)
		}

		return &clone

	case *ToolResponsePart:
		if v == nil {
			return nil
		}

		clone := *v
		if v.Result != nil {
			clone.Result = append(json.RawMessage(nil), v.Result...)
		}

		return &clone

	case *ReasoningPart:
		if v == nil {
			return nil
		}

		clone := *v
		if v.Metadata != nil {
			clone.Metadata = maps.Clone(v.Metadata)
		}

		return &clone

	default:
		return p
	}
}

// CloneMessage returns a deep copy of m. The returned message's Content slice
// is independent of the original and each Part is deep-copied via ClonePart.
func CloneMessage(m Message) Message {
	out := Message{Role: m.Role}
	if m.Content == nil {
		return out
	}

	out.Content = make([]Part, len(m.Content))
	for i, p := range m.Content {
		out.Content[i] = ClonePart(p)
	}

	return out
}

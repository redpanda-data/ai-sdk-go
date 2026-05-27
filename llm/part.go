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
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
)

// Part represents a single unit of content within a Message or Response.
//
// Part is a sealed interface: only the concrete pointer types declared in
// this package satisfy it. Discriminate with a type switch:
//
//	switch p := part.(type) {
//	case *llm.TextPart:
//	    ...
//	case *llm.ToolRequestPart:
//	    ...
//	}
//
// The marker method has a pointer receiver so value-form literals such as
// `TextPart{}` deliberately do not satisfy the interface. Construct parts
// with the New*Part helpers or `&TextPart{...}` literals.
type Part interface {
	isPart()
}

// TextPart contains plain text content.
type TextPart struct {
	Text string `json:"text,omitempty"`
}

// ToolRequestPart represents a request from the model to execute a tool.
type ToolRequestPart struct {
	// ID is a unique identifier for this tool request within the conversation.
	ID string `json:"id"`

	// Name is the name of the tool to execute.
	Name string `json:"name"`

	// Arguments contains the tool input as JSON.
	Arguments json.RawMessage `json:"arguments,omitempty"`

	// Metadata carries provider-specific data that must round-trip with this
	// tool call (e.g. Gemini 3 Pro thought signatures). Untouched by the SDK
	// core; populated and consumed by individual providers.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// ToolResponsePart represents the result of executing a tool, sent back to
// the model to continue the conversation.
type ToolResponsePart struct {
	// ID matches the ID from the corresponding ToolRequestPart.
	ID string `json:"id"`

	// Name is the name of the tool that was executed.
	Name string `json:"name"`

	// Result contains the tool output as JSON.
	Result json.RawMessage `json:"result,omitempty"`

	// IsError indicates the tool reported a failure.
	// When true, Result typically contains an error payload.
	IsError bool `json:"is_error,omitempty"`
}

// ReasoningPart represents reasoning thoughts/traces from the model.
type ReasoningPart struct {
	// ID is a unique identifier for this reasoning trace.
	ID string `json:"id,omitempty"`

	// Text contains the reasoning content.
	Text string `json:"text,omitempty"`

	// Signature is a provider-supplied opaque token that authenticates
	// the reasoning block on subsequent turns (e.g. Anthropic extended
	// thinking signatures, OpenAI reasoning IDs).
	Signature string `json:"signature,omitempty"`

	// Metadata provides additional provider-specific context about the
	// reasoning trace.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewTextPart creates a TextPart with the given text.
func NewTextPart(text string) *TextPart {
	return &TextPart{Text: text}
}

// NewToolRequestPart creates a ToolRequestPart with the given fields.
func NewToolRequestPart(id, name string, arguments json.RawMessage) *ToolRequestPart {
	return &ToolRequestPart{ID: id, Name: name, Arguments: arguments}
}

// NewToolResponsePart creates a ToolResponsePart with the given fields.
// Set isError true when result is an error payload.
func NewToolResponsePart(id, name string, result json.RawMessage, isError bool) *ToolResponsePart {
	return &ToolResponsePart{ID: id, Name: name, Result: result, IsError: isError}
}

// NewReasoningPart creates a ReasoningPart with the given text.
func NewReasoningPart(text string) *ReasoningPart {
	return &ReasoningPart{Text: text}
}

func (*TextPart) isPart()         {}
func (*ToolRequestPart) isPart()  {}
func (*ToolResponsePart) isPart() {}
func (*ReasoningPart) isPart()    {}

// JoinTextParts concatenates the text from every *TextPart in parts.
// Non-text parts are ignored.
func JoinTextParts(parts []Part) string {
	var b strings.Builder

	for _, p := range parts {
		if tp, ok := p.(*TextPart); ok && tp != nil {
			b.WriteString(tp.Text)
		}
	}

	return b.String()
}

// Part type discriminators used on the JSON wire envelope.
const (
	partTypeText         = "text"
	partTypeToolRequest  = "tool_request"
	partTypeToolResponse = "tool_response"
	partTypeReasoning    = "reasoning"
)

// MarshalPart encodes a Part as a flat JSON object with a discriminator:
//
//	{"type":"text","text":"..."}
//	{"type":"tool_request","id":"...","name":"...","arguments":...}
//
// A typed-nil pointer (e.g. `var p *TextPart; MarshalPart(p)`) marshals to
// JSON null rather than panicking.
func MarshalPart(p Part) ([]byte, error) {
	switch v := p.(type) {
	case *TextPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type string `json:"type"`
			Text string `json:"text,omitempty"`
		}{partTypeText, v.Text})

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
		}{partTypeToolRequest, v.ID, v.Name, v.Arguments, v.Metadata})

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
		}{partTypeToolResponse, v.ID, v.Name, v.Result, v.IsError})

	case *ReasoningPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type      string         `json:"type"`
			ID        string         `json:"id,omitempty"`
			Text      string         `json:"text,omitempty"`
			Signature string         `json:"signature,omitempty"`
			Metadata  map[string]any `json:"metadata,omitempty"`
		}{partTypeReasoning, v.ID, v.Text, v.Signature, v.Metadata})

	case nil:
		return []byte("null"), nil

	default:
		return nil, fmt.Errorf("llm: unknown Part type %T", p)
	}
}

// UnmarshalPart decodes a Part previously encoded by MarshalPart. The input
// must be a JSON object containing a "type" discriminator (or the literal
// null, which yields a nil Part).
func UnmarshalPart(data []byte) (Part, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return nil, nil //nolint:nilnil // null input yields nil Part by design
	}

	var head struct {
		Type string `json:"type"`
	}

	err := json.Unmarshal(data, &head)
	if err != nil {
		return nil, fmt.Errorf("llm: decode part envelope: %w", err)
	}

	switch head.Type {
	case partTypeText:
		var v TextPart

		err := json.Unmarshal(data, &v)
		if err != nil {
			return nil, fmt.Errorf("llm: decode text part: %w", err)
		}

		return &v, nil

	case partTypeToolRequest:
		var v ToolRequestPart

		err := json.Unmarshal(data, &v)
		if err != nil {
			return nil, fmt.Errorf("llm: decode tool request part: %w", err)
		}

		return &v, nil

	case partTypeToolResponse:
		var v ToolResponsePart

		err := json.Unmarshal(data, &v)
		if err != nil {
			return nil, fmt.Errorf("llm: decode tool response part: %w", err)
		}

		return &v, nil

	case partTypeReasoning:
		var v ReasoningPart

		err := json.Unmarshal(data, &v)
		if err != nil {
			return nil, fmt.Errorf("llm: decode reasoning part: %w", err)
		}

		return &v, nil

	case "":
		return nil, fmt.Errorf("llm: part envelope missing %q discriminator", "type")

	default:
		return nil, fmt.Errorf("llm: unknown part type %q", head.Type)
	}
}

// ClonePart returns a deep copy of p. The returned value is independent of
// the input: mutating one will not affect the other.
func ClonePart(p Part) Part {
	switch v := p.(type) {
	case *TextPart:
		if v == nil {
			return nil
		}

		out := *v

		return &out

	case *ToolRequestPart:
		if v == nil {
			return nil
		}

		out := *v

		out.Arguments = cloneRawMessage(v.Arguments)
		if v.Metadata != nil {
			out.Metadata = maps.Clone(v.Metadata)
		}

		return &out

	case *ToolResponsePart:
		if v == nil {
			return nil
		}

		out := *v
		out.Result = cloneRawMessage(v.Result)

		return &out

	case *ReasoningPart:
		if v == nil {
			return nil
		}

		out := *v
		if v.Metadata != nil {
			out.Metadata = maps.Clone(v.Metadata)
		}

		return &out

	case nil:
		return nil

	default:
		// Unknown Part implementations: best effort, return as-is. We do
		// not allow external types to satisfy Part, so this is unreachable
		// in practice.
		return p
	}
}

func cloneRawMessage(in json.RawMessage) json.RawMessage {
	if in == nil {
		return nil
	}

	out := make(json.RawMessage, len(in))
	copy(out, in)

	return out
}

// CloneMessage returns a deep copy of m. The Content slice and every Part
// in it are duplicated so the result is safe to mutate independently of m.
func CloneMessage(m Message) Message {
	out := Message{Role: m.Role}
	if m.Content != nil {
		out.Content = make([]Part, len(m.Content))
		for i, p := range m.Content {
			out.Content[i] = ClonePart(p)
		}
	}

	return out
}

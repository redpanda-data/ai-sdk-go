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

// Part is a sealed-interface discriminated union representing a single
// piece of content within a Message or streaming event.
//
// The unexported isPart() marker prevents external packages from
// implementing Part. The marker uses pointer receivers, so only the
// pointer form (e.g. *TextPart) satisfies Part; value-form usage is a
// compile-time error and prevents accidental shallow copies that would
// alias mutable internal state.
//
// Concrete implementations:
//
//   - *TextPart for plain text content.
//   - *ToolRequestPart for a model-emitted tool invocation.
//   - *ToolResponsePart for the result of executing a tool.
//   - *ReasoningPart for reasoning thoughts surfaced by the model.
//
// Consumers type-switch on the concrete pointer rather than reading a
// "kind" discriminator and dereferencing optional fields:
//
//	for _, p := range msg.Content {
//	    switch p := p.(type) {
//	    case *llm.TextPart:
//	        fmt.Println(p.Text)
//	    case *llm.ToolRequestPart:
//	        invokeTool(p.ID, p.Name, p.Arguments)
//	    }
//	}
type Part interface {
	isPart()
}

// TextPart contains plain textual content.
type TextPart struct {
	// Text is the textual content. For streaming deltas, this is the
	// incremental fragment; for assembled messages, the full text.
	Text string `json:"text"`

	// Metadata carries provider-specific or call-site information that
	// should travel with the part but does not change its semantics.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewTextPart creates a TextPart with the given text. The returned
// pointer satisfies Part and lets callers read fields without a type
// assertion when they hold the concrete type.
func NewTextPart(text string) *TextPart {
	return &TextPart{Text: text}
}

func (*TextPart) isPart() {}

// ToolRequestPart represents a model request to execute a tool.
// This corresponds to function calling in various AI models.
type ToolRequestPart struct {
	// ID uniquely identifies this tool request within the conversation
	// so a matching ToolResponsePart can reference it.
	ID string `json:"id"`

	// Name is the name of the tool to execute.
	Name string `json:"name"`

	// Arguments is the JSON-encoded input to the tool. The structure
	// depends on the tool's input schema.
	Arguments json.RawMessage `json:"arguments"`

	// Metadata carries provider-specific or call-site information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewToolRequestPart creates a ToolRequestPart with the given fields.
func NewToolRequestPart(id, name string, arguments json.RawMessage) *ToolRequestPart {
	return &ToolRequestPart{ID: id, Name: name, Arguments: arguments}
}

func (*ToolRequestPart) isPart() {}

// ToolResponsePart contains the result of a tool execution.
// This is sent back to the model to continue the conversation.
type ToolResponsePart struct {
	// ID matches the ID from the corresponding ToolRequestPart.
	ID string `json:"id"`

	// Name is the name of the tool that was executed.
	Name string `json:"name"`

	// Result is the JSON-encoded tool output. The structure depends on
	// the tool's output schema. When Error is non-empty, Result should
	// be ignored.
	Result json.RawMessage `json:"result"`

	// Error contains a human-readable error message when execution
	// failed. When non-empty, Result should be ignored.
	Error string `json:"error,omitempty"`

	// Metadata carries provider-specific or call-site information.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewToolResponsePart creates a ToolResponsePart with the given result.
func NewToolResponsePart(id, name string, result json.RawMessage) *ToolResponsePart {
	return &ToolResponsePart{ID: id, Name: name, Result: result}
}

// NewToolErrorResponsePart creates a ToolResponsePart describing a tool
// execution failure.
func NewToolErrorResponsePart(id, name, errMsg string) *ToolResponsePart {
	return &ToolResponsePart{ID: id, Name: name, Error: errMsg}
}

func (*ToolResponsePart) isPart() {}

// ReasoningPart represents reasoning thoughts/traces from the model.
// This contains the model's internal reasoning process, which may be
// exposed or summarized depending on the provider and configuration.
type ReasoningPart struct {
	// ID uniquely identifies this reasoning trace. For Anthropic, this
	// is the signature; for OpenAI, the response ID.
	ID string `json:"id,omitempty"`

	// Text contains the reasoning content. For streaming responses,
	// this may be built up incrementally.
	Text string `json:"text"`

	// Metadata carries provider-specific information such as
	// obfuscation keys, reasoning effort levels, or redaction flags.
	Metadata map[string]any `json:"metadata,omitempty"`
}

// NewReasoningPart creates a ReasoningPart with the given text.
func NewReasoningPart(text string) *ReasoningPart {
	return &ReasoningPart{Text: text}
}

func (*ReasoningPart) isPart() {}

// JoinTextParts combines all TextPart text from the slice into a single
// string. Non-text parts are ignored.
func JoinTextParts(parts []Part) string {
	var texts []string

	for _, p := range parts {
		if t, ok := p.(*TextPart); ok && t != nil {
			texts = append(texts, t.Text)
		}
	}

	return strings.Join(texts, "")
}

// PartsOfType returns all parts of the given concrete pointer type from
// the slice. This replaces the old FilterParts(PartKind) helper with a
// type-safe alternative that requires no runtime kind comparison.
//
// Typical use: PartsOfType[*llm.ToolRequestPart](msg.Content).
func PartsOfType[T Part](parts []Part) []T {
	var out []T

	for _, p := range parts {
		if v, ok := p.(T); ok {
			out = append(out, v)
		}
	}

	return out
}

// ClonePart returns a deep copy of p. The returned value is independent
// of the input: mutating the clone does not affect the original. Use it
// at persistence and event-emission boundaries where the caller and
// callee must not share mutable Part state.
//
// Returns nil if p is an interface-nil or a typed-nil pointer.
func ClonePart(p Part) Part {
	if p == nil {
		return nil
	}

	switch v := p.(type) {
	case *TextPart:
		if v == nil {
			return nil
		}

		return &TextPart{
			Text:     v.Text,
			Metadata: cloneMetadata(v.Metadata),
		}
	case *ToolRequestPart:
		if v == nil {
			return nil
		}

		return &ToolRequestPart{
			ID:        v.ID,
			Name:      v.Name,
			Arguments: cloneRawMessage(v.Arguments),
			Metadata:  cloneMetadata(v.Metadata),
		}
	case *ToolResponsePart:
		if v == nil {
			return nil
		}

		return &ToolResponsePart{
			ID:       v.ID,
			Name:     v.Name,
			Result:   cloneRawMessage(v.Result),
			Error:    v.Error,
			Metadata: cloneMetadata(v.Metadata),
		}
	case *ReasoningPart:
		if v == nil {
			return nil
		}

		return &ReasoningPart{
			ID:       v.ID,
			Text:     v.Text,
			Metadata: cloneMetadata(v.Metadata),
		}
	default:
		// Unknown Part — return as-is. The sealed marker prevents
		// external implementations, so this branch is unreachable for
		// well-formed inputs.
		return p
	}
}

// CloneMessage returns a deep copy of m. The returned Message shares no
// mutable state with the input; each Part is cloned via ClonePart.
func CloneMessage(m Message) Message {
	if len(m.Content) == 0 {
		return Message{Role: m.Role}
	}

	out := Message{
		Role:    m.Role,
		Content: make([]Part, len(m.Content)),
	}
	for i, p := range m.Content {
		out.Content[i] = ClonePart(p)
	}

	return out
}

func cloneMetadata(in map[string]any) map[string]any {
	if in == nil {
		return nil
	}

	out := make(map[string]any, len(in))
	maps.Copy(out, in)

	return out
}

func cloneRawMessage(in json.RawMessage) json.RawMessage {
	if in == nil {
		return nil
	}

	out := make(json.RawMessage, len(in))
	copy(out, in)

	return out
}

// Part wire-format constants used by MarshalPart and UnmarshalPart.
const (
	partTypeText         = "text"
	partTypeToolRequest  = "tool_request"
	partTypeToolResponse = "tool_response"
	partTypeReasoning    = "reasoning"
)

// MarshalPart encodes a Part as a flat JSON envelope of the form
// {"type":"...", ...concrete fields...}. Interface-nil and typed-nil
// pointers marshal as JSON null.
func MarshalPart(p Part) ([]byte, error) {
	if p == nil {
		return []byte("null"), nil
	}

	switch v := p.(type) {
	case *TextPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type     string         `json:"type"`
			Text     string         `json:"text"`
			Metadata map[string]any `json:"metadata,omitempty"`
		}{partTypeText, v.Text, v.Metadata})
	case *ToolRequestPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type      string          `json:"type"`
			ID        string          `json:"id"`
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
			Metadata  map[string]any  `json:"metadata,omitempty"`
		}{partTypeToolRequest, v.ID, v.Name, v.Arguments, v.Metadata})
	case *ToolResponsePart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type     string          `json:"type"`
			ID       string          `json:"id"`
			Name     string          `json:"name"`
			Result   json.RawMessage `json:"result"`
			Error    string          `json:"error,omitempty"`
			Metadata map[string]any  `json:"metadata,omitempty"`
		}{partTypeToolResponse, v.ID, v.Name, v.Result, v.Error, v.Metadata})
	case *ReasoningPart:
		if v == nil {
			return []byte("null"), nil
		}

		return json.Marshal(struct {
			Type     string         `json:"type"`
			ID       string         `json:"id,omitempty"`
			Text     string         `json:"text"`
			Metadata map[string]any `json:"metadata,omitempty"`
		}{partTypeReasoning, v.ID, v.Text, v.Metadata})
	default:
		return nil, fmt.Errorf("llm: cannot marshal unknown Part type %T", p)
	}
}

// UnmarshalPart decodes a JSON envelope produced by MarshalPart back
// into the concrete Part pointer.
func UnmarshalPart(data []byte) (Part, error) {
	if len(data) == 0 || string(data) == "null" {
		return nil, nil //nolint:nilnil // nil sentinel for absent/null part
	}

	var probe struct {
		Type string `json:"type"`
	}

	if err := json.Unmarshal(data, &probe); err != nil {
		return nil, fmt.Errorf("llm: decode Part envelope: %w", err)
	}

	switch probe.Type {
	case partTypeText:
		var v TextPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode TextPart: %w", err)
		}

		return &v, nil
	case partTypeToolRequest:
		var v ToolRequestPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ToolRequestPart: %w", err)
		}

		return &v, nil
	case partTypeToolResponse:
		var v ToolResponsePart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ToolResponsePart: %w", err)
		}

		return &v, nil
	case partTypeReasoning:
		var v ReasoningPart
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, fmt.Errorf("llm: decode ReasoningPart: %w", err)
		}

		return &v, nil
	default:
		return nil, fmt.Errorf("llm: unknown Part type %q", probe.Type)
	}
}

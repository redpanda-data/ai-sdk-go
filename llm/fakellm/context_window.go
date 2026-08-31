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

package fakellm

import (
	"fmt"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// CountRequestTokens estimates the input tokens req consumes, counting tool
// arguments, results, reasoning and schemas as well as text — tool payloads are
// most of an agentic prompt. Accuracy is bounded by the configured Tokenizer
// (4 chars per token by default); wire framing is not counted.
func (m *FakeModel) CountRequestTokens(req *llm.Request) int {
	if req == nil {
		return 0
	}

	total := 0

	for _, msg := range req.Messages {
		for _, part := range msg.Content {
			total += m.countPartTokens(part)
		}
	}

	for _, def := range req.Tools {
		total += m.tokenizer.Count(def.Name)
		total += m.tokenizer.Count(def.Description)
		total += m.tokenizer.Count(string(def.Parameters))
	}

	if rf := req.ResponseFormat; rf != nil && rf.JSONSchema != nil {
		total += m.tokenizer.Count(rf.JSONSchema.Name)
		total += m.tokenizer.Count(rf.JSONSchema.Description)
		total += m.tokenizer.Count(string(rf.JSONSchema.Schema))
	}

	return total
}

// countPartTokens sizes one part. Typed-nil pointers are valid Parts (see
// llm.MarshalPart), so each case guards before dereferencing.
func (m *FakeModel) countPartTokens(part llm.Part) int {
	switch p := part.(type) {
	case *llm.TextPart:
		if p == nil {
			return 0
		}

		return m.tokenizer.Count(p.Text)

	case *llm.ReasoningPart:
		if p == nil {
			return 0
		}

		return m.tokenizer.Count(p.Text) + m.tokenizer.Count(p.Signature)

	case *llm.ToolRequestPart:
		if p == nil {
			return 0
		}

		return m.tokenizer.Count(p.Name) + m.tokenizer.Count(string(p.Arguments))

	case *llm.ToolResponsePart:
		if p == nil {
			return 0
		}

		return m.tokenizer.Count(p.Name) + m.tokenizer.Count(string(p.Result))

	default:
		return 0
	}
}

// checkContextWindow returns the rejection for an over-window request, or nil.
//
// Only active once WithContextWindow has been used: enforcing the inherited
// 128K default would change behaviour for every existing test.
func (m *FakeModel) checkContextWindow(req *llm.Request, kind CallKind) error {
	if !m.enforceWindow || m.constraints.MaxInputTokens <= 0 {
		return nil
	}

	used := m.CountRequestTokens(req)
	if used <= m.constraints.MaxInputTokens {
		return nil
	}

	// Anthropic's wording, so a test asserting on the message reads like a real
	// log.
	err := &llm.ProviderError{
		Base:    llm.ErrContextOverflow,
		Code:    "400",
		Message: fmt.Sprintf("prompt is too long: %d tokens > %d maximum", used, m.constraints.MaxInputTokens),
	}

	m.logCall(Call{
		When:     time.Now(),
		Kind:     kind,
		Request:  req,
		Err:      err,
		RuleName: "context-window-overflow",
	})

	return err
}

// rejectStream yields a rejection as an iterator error wrapped in
// llm.ErrAPICall, the shape every real provider produces for a request-time
// failure on the streaming path (verified live against all four, 2026-08-21).
func rejectStream(m *FakeModel, cc *CallContext, err error) func(func(llm.Event, error) bool) {
	return func(yield func(llm.Event, error) bool) {
		defer m.endCall(cc)

		yield(nil, fmt.Errorf("%w: %w", llm.ErrAPICall, err))
	}
}

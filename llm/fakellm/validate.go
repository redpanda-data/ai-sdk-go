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
	"encoding/json"
	"fmt"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ValidateConversation checks a request's message shape the way a provider
// would, so a history rewrite that would 400 in production fails a fake-based
// test instead:
//
//   - no message with zero parts
//   - no ToolResponsePart answering a tool call that is not pending
//     (orphaned tool_result - its tool_use was dropped or never existed)
//   - no tool call left unanswered once the conversation moves on
//     (tool_use without a following result), except calls from the final
//     assistant message, which are legitimately awaiting execution
//   - every ToolResponsePart.Result is valid JSON
//
// Role alternation is deliberately not validated: providers disagree on it.
func ValidateConversation(messages []llm.Message) error {
	// pending maps outstanding tool-call IDs to the index of the assistant
	// message that issued them.
	pending := make(map[string]int)

	for i, msg := range messages {
		if len(msg.Content) == 0 {
			return fmt.Errorf("message %d (%s) has no content", i, msg.Role)
		}

		if msg.Role == llm.RoleAssistant && len(pending) > 0 {
			for id, at := range pending {
				return fmt.Errorf("tool call %q from message %d has no tool result before the next assistant message %d", id, at, i)
			}
		}

		for _, part := range msg.Content {
			switch p := part.(type) {
			case *llm.ToolRequestPart:
				if p == nil {
					continue
				}

				pending[p.ID] = i

			case *llm.ToolResponsePart:
				if p == nil {
					continue
				}

				if _, ok := pending[p.ID]; !ok {
					return fmt.Errorf("message %d carries an orphaned tool result %q: no pending tool call with that id", i, p.ID)
				}

				delete(pending, p.ID)

				if len(p.Result) == 0 || !json.Valid(p.Result) {
					return fmt.Errorf("message %d tool result %q payload is not valid JSON", i, p.ID)
				}
			}
		}
	}

	// Calls still pending are fine only when issued by the final message:
	// their results are legitimately still being executed.
	last := len(messages) - 1
	for id, at := range pending {
		if at != last {
			return fmt.Errorf("tool call %q from message %d has no tool result", id, at)
		}
	}

	return nil
}

// checkConversation rejects a malformed conversation the way a provider
// would, so an invalid history rewrite fails fast in tests. Always on: a
// request that would 400 in production must not pass against the fake.
func (m *FakeModel) checkConversation(req *llm.Request, kind CallKind) error {
	verr := ValidateConversation(req.Messages)
	if verr == nil {
		return nil
	}

	err := &llm.ProviderError{
		Base:    llm.ErrInvalidInput,
		Code:    "invalid_conversation",
		Message: verr.Error(),
	}

	m.logCall(Call{
		When:     time.Now(),
		Kind:     kind,
		Request:  req,
		Err:      err,
		RuleName: "conversation-validator",
	})

	return err
}

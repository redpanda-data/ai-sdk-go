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

package a2a

import (
	"fmt"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

// TestExecutor_ContextWindowExceededIsTerminalAndTruthful verifies that when the
// provider rejects the request before generation because it does not fit the
// context window (surfaced as llm.ErrContextOverflow), the executor fails
// the task with a truthful, actionable message — not the raw provider error. The
// rejection also covers input + max_tokens overflowing while the input alone
// fits, so the message must name lowering the response limit too.
func TestExecutor_ContextWindowExceededIsTerminalAndTruthful(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return nil, fmt.Errorf("anthropic generate: %w", llm.ErrContextOverflow)
		})

	events := runExecutorOnce(t, model, "Summarize this enormous document.")

	final := finalStatusEvent(t, events)
	assert.Equal(t, a2a.TaskStateFailed, final.Status.State,
		"a context-window overflow is terminal")
	assert.Contains(t, statusText(final),
		"Shorten the input or start a new conversation, or lower the response token limit.",
		"the failure must name both remedies the provider names")
}

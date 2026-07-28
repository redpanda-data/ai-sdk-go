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
// provider rejects the request before generation because the conversation does
// not fit the context window (surfaced as llm.ErrContextWindowExceeded), the
// executor fails the task with the same truthful, actionable message it uses for
// the 200-response FinishReasonContextOverflow — not the raw provider error.
func TestExecutor_ContextWindowExceededIsTerminalAndTruthful(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).
		ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
			return nil, fmt.Errorf("anthropic generate: %w", llm.ErrContextWindowExceeded)
		})

	events := runExecutorOnce(t, model, "Summarize this enormous document.")

	final := finalStatusEvent(t, events)
	assert.Equal(t, a2a.TaskStateFailed, final.Status.State,
		"a context-window overflow is terminal")
	assert.Contains(t, statusText(final), "Start a new conversation or shorten the input.",
		"the failure must carry the truthful, actionable overflow message")
}

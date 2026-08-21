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

package fakellm_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

// bigText returns a string long enough to exceed a small token window.
func bigText(tokens int) string {
	return strings.Repeat("x", tokens*4)
}

func TestWithContextWindow(t *testing.T) {
	t.Parallel()

	assert.Equal(t, 128000, fakellm.NewFakeModel().Constraints().MaxInputTokens,
		"the default window is unchanged, so existing tests are unaffected")

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(2000))
	assert.Equal(t, 2000, model.Constraints().MaxInputTokens)
	assert.Equal(t, 4096, model.Constraints().MaxOutputTokens, "other constraints keep their defaults")
}

// TestCountRequestTokens_CountsToolPayloads pins the behaviour overflow testing
// depends on: tool arguments, results, reasoning and schemas all consume
// context. A text-only counter would report zero here and no window would trip.
func TestCountRequestTokens_CountsToolPayloads(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()

	tests := []struct {
		name string
		req  *llm.Request
	}{
		{
			name: "tool request arguments",
			req: &llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("1", "fetch",
					json.RawMessage(`{"q":"`+bigText(200)+`"}`))),
			}},
		},
		{
			name: "tool response result",
			req: &llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("1", "fetch", json.RawMessage(`{}`))),
				llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("1", "fetch",
					json.RawMessage(`{"d":"`+bigText(200)+`"}`), false)),
			}},
		},
		{
			name: "reasoning trace",
			req: &llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant, &llm.ReasoningPart{Text: bigText(200)}),
			}},
		},
		{
			name: "tool schema",
			req: &llm.Request{Tools: []llm.ToolDefinition{{
				Name:       "fetch",
				Parameters: json.RawMessage(`{"x":"` + bigText(200) + `"}`),
			}}},
		},
		{
			name: "response format schema",
			req: &llm.Request{ResponseFormat: &llm.ResponseFormat{
				Type: llm.ResponseFormatJSONSchema,
				JSONSchema: &llm.JSONSchema{
					Name:   "report",
					Schema: json.RawMessage(`{"x":"` + bigText(200) + `"}`),
				},
			}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.GreaterOrEqual(t, model.CountRequestTokens(tt.req), 200)
		})
	}

	assert.Equal(t, 0, model.CountRequestTokens(nil), "nil request counts as empty")
}

// TestCountRequestTokens_MatchesReportedUsage checks the count driving
// enforcement is the same one reported as usage, so a test cannot see a request
// "fit" while being billed for more.
func TestCountRequestTokens_MatchesReportedUsage(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("ok")

	req := &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("1", "fetch", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("1", "fetch",
			json.RawMessage(`{"d":"`+bigText(300)+`"}`), false)),
	}}

	resp, err := model.Generate(t.Context(), req)
	require.NoError(t, err)
	require.NotNil(t, resp.Usage)

	assert.Equal(t, model.CountRequestTokens(req), resp.Usage.InputTokens)
}

// TestOverflow_RejectsBeforeRules verifies enforcement happens ahead of rule
// matching, the way a provider validates a prompt before the model sees it: no
// configured rule can rescue an oversized request.
func TestOverflow_RejectsBeforeRules(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(100))
	model.When(fakellm.Any()).ThenRespondText("this rule must not win")

	req := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(bigText(500)))}}

	resp, err := model.Generate(t.Context(), req)

	require.Error(t, err)
	assert.Nil(t, resp)
	require.ErrorIs(t, err, llm.ErrInvalidInput)
	assert.False(t, llm.IsRetryable(err), "an oversized prompt will not fix itself on retry")
	assert.Contains(t, err.Error(), "prompt is too long")

	var provErr *llm.ProviderError
	require.ErrorAs(t, err, &provErr)
	assert.Equal(t, "400", provErr.Code)

	// The rejected call is recorded, so a test can inspect what broke the window.
	calls := model.Calls()
	require.Len(t, calls, 1)
	assert.Same(t, req, calls[0].Request)
	require.Error(t, calls[0].Err)
}

// TestOverflow_Streaming checks the streaming path reports the rejection as an
// iterator error wrapped in ErrAPICall — the shape every real provider
// produces, and the only one plugins/retry inspects.
func TestOverflow_Streaming(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(100))
	model.When(fakellm.Any()).ThenRespondText("this rule must not win")

	req := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(bigText(500)))}}

	var streamErr error

	for evt, err := range model.GenerateEvents(t.Context(), req) {
		require.Nil(t, evt, "the rejection belongs on the iterator, not an event, got %T", evt)

		streamErr = err
	}

	require.ErrorIs(t, streamErr, llm.ErrAPICall)
	require.ErrorIs(t, streamErr, llm.ErrContextOverflow)
	assert.False(t, llm.IsRetryable(streamErr))
}

func TestOverflow_UnderLimitPassesThrough(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(1000))
	model.When(fakellm.Any()).ThenRespondText("ok")

	resp, err := model.Generate(t.Context(), &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello")),
	}})

	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Message.TextContent())
}

// TestOverflow_NotEnforcedByDefault pins that the default 128K window is
// metadata only. Enforcing it would change behaviour for every existing test.
func TestOverflow_NotEnforcedByDefault(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("ok")

	_, err := model.Generate(t.Context(), &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(bigText(200_000))),
	}})

	require.NoError(t, err)
}

// TestOverflow_ZeroWindowDisablesEnforcement documents that a non-positive
// window means "unknown", not "reject everything".
func TestOverflow_ZeroWindowDisablesEnforcement(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(0))
	model.When(fakellm.Any()).ThenRespondText("ok")

	resp, err := model.Generate(t.Context(), &llm.Request{Messages: []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(bigText(5000))),
	}})

	require.NoError(t, err)
	assert.Equal(t, "ok", resp.Message.TextContent())
}

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
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	llmtesting "github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

// TestFakeModel_StreamEndMatchesGenerate pins the contract documented on
// llm.StreamEndEvent.Response: the end event carries the complete response,
// aggregating everything the deltas reported. An agent loop stores that
// message in session history, so a fake that omits it hands every consumer a
// roleless, contentless assistant turn.
func TestFakeModel_StreamEndMatchesGenerate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		// newModel is called per request so both paths see the same call
		// count, keeping generated IDs comparable.
		newModel func() *llmtesting.FakeModel
	}{
		{
			name: "text",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondText("the real answer")
			},
		},
		{
			name: "text longer than one chunk",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondText(strings.Repeat("long answer. ", 12))
			},
		},
		{
			name: "text with finish reason option",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondText("truncated", llmtesting.WithFinishReason(llm.FinishReasonLength))
			},
		},
		{
			name: "tool call",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondWithToolCall("lookup", map[string]any{"query": "redpanda"})
			},
		},
		{
			name: "mixed text and tool call",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondWith(func(_ *llm.Request, _ *llmtesting.CallContext) (*llm.Response, error) {
						return &llm.Response{
							Message: llm.Message{
								Role: llm.RoleAssistant,
								Content: []llm.Part{
									llm.NewTextPart("let me look that up"),
									llm.NewToolRequestPart("call_mixed", "lookup", json.RawMessage(`{"query":"redpanda"}`)),
								},
							},
							FinishReason: llm.FinishReasonToolCalls,
						}, nil
					})
			},
		},
		{
			name: "json",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenRespondJSON(map[string]string{"city": "SF"})
			},
		},
		{
			name: "streamed text",
			newModel: func() *llmtesting.FakeModel {
				return llmtesting.NewFakeModel().
					When(llmtesting.Any()).
					ThenStreamText("chunked answer", llmtesting.StreamConfig{ChunkSize: 4})
			},
		},
		{
			name:     "fallback with no rules",
			newModel: func() *llmtesting.FakeModel { return llmtesting.NewFakeModel() },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			want, err := tt.newModel().Generate(context.Background(), parityRequest())
			require.NoError(t, err)
			require.NotEmpty(t, want.Message.Content, "fixture must produce content on the Generate path")

			deltas, end := collectStream(t, tt.newModel(), parityRequest())

			assert.Equal(t, want.Message, end.Message, "stream end must aggregate the same message Generate returns")
			assert.Equal(t, want.FinishReason, end.FinishReason)
			assert.Equal(t, want.ID, end.ID)

			require.NotNil(t, end.Usage)
			assert.Equal(t, want.Usage.InputTokens, end.Usage.InputTokens)
			assert.Equal(t, want.Usage.OutputTokens, end.Usage.OutputTokens)

			// The deltas must describe the message the end event aggregates,
			// so a consumer that renders progressively and a consumer that
			// stores the final message agree on what the model said.
			assert.Equal(t, want.Message.TextContent(), textOfParts(deltas))
			assert.Equal(t, toolRequestsOf(want.Message.Content), toolRequestsOf(deltas))
		})
	}
}

// TestFakeModel_StreamedToolRoundTripKeepsAnswer covers the failure as an agent
// loop meets it: the answer after a tool round trip is a plain text rule, and
// before the fix the stream dropped it, leaving a roleless placeholder in
// session history that every later request carried.
func TestFakeModel_StreamedToolRoundTripKeepsAnswer(t *testing.T) {
	t.Parallel()

	const answer = "The weather in San Francisco is 68°F and sunny."

	model := llmtesting.NewFakeModel().
		Scenario("weather-lookup", func(s *llmtesting.ScenarioBuilder) {
			s.OnTurn(0).
				When(llmtesting.HasTool("get_weather")).
				ThenRespondWithToolCall("get_weather", map[string]any{"location": "San Francisco"})

			s.OnTurn(1).
				When(llmtesting.LastMessageHasToolResponse("get_weather")).
				ThenRespondText(answer)
		})

	// user, tool request, tool response, answer
	history := make([]llm.Message, 0, 4)
	history = append(history, llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather in SF?")))

	newReq := func() *llm.Request {
		return &llm.Request{
			Messages: append([]llm.Message(nil), history...),
			Tools:    []llm.ToolDefinition{{Name: "get_weather"}},
			Metadata: map[string]string{"session_id": "test-session"},
		}
	}

	// Turn 0: the model asks for the tool.
	_, toolTurn := collectStream(t, model, newReq())
	require.True(t, toolTurn.HasToolRequests(), "turn 0 must request the tool")

	history = append(history, toolTurn.Message)

	toolReq := toolTurn.ToolRequests()[0]
	history = append(history, llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(
		toolReq.ID, toolReq.Name, json.RawMessage(`{"temperature":68,"condition":"sunny"}`), false,
	)))

	// Turn 1: the text answer must survive the streaming path.
	_, answerTurn := collectStream(t, model, newReq())

	assert.Equal(t, llm.RoleAssistant, answerTurn.Message.Role)
	require.Len(t, answerTurn.Message.Content, 1)
	assert.Equal(t, answer, answerTurn.Message.TextContent())

	// What an agent loop would append to session history.
	history = append(history, answerTurn.Message)
	for i, msg := range history {
		assert.NotEmpty(t, msg.Role, "history[%d] must carry a role", i)
		assert.NotEmpty(t, msg.Content, "history[%d] must carry content", i)
	}
}

func parityRequest() *llm.Request {
	return &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi")),
		},
	}
}

// collectStream drains GenerateEvents, returning the delta parts and the
// response from the terminating StreamEndEvent.
func collectStream(t *testing.T, model *llmtesting.FakeModel, req *llm.Request) ([]llm.Part, *llm.Response) {
	t.Helper()

	var (
		deltas []llm.Part
		end    *llm.Response
	)

	for event, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ContentPartEvent:
			deltas = append(deltas, e.Part)
		case llm.StreamEndEvent:
			require.NoError(t, e.Error)
			require.NotNil(t, e.Response, "stream end must carry a response")

			end = e.Response
		}
	}

	require.NotNil(t, end, "stream must terminate with a StreamEndEvent")

	return deltas, end
}

func textOfParts(parts []llm.Part) string {
	var sb strings.Builder

	for _, part := range parts {
		if tp, ok := part.(*llm.TextPart); ok {
			sb.WriteString(tp.Text)
		}
	}

	return sb.String()
}

func toolRequestsOf(parts []llm.Part) []*llm.ToolRequestPart {
	var out []*llm.ToolRequestPart

	for _, part := range parts {
		if tr, ok := part.(*llm.ToolRequestPart); ok {
			out = append(out, tr)
		}
	}

	return out
}

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
	"context"
	"encoding/json"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func TestExecutorInputModesMatchExecution(t *testing.T) {
	t.Parallel()

	for _, vision := range []bool{false, true} {
		for _, tc := range []struct {
			mode string
			part a2a.Part
			text string
		}{
			{"text/plain", a2a.TextPart{Text: "hello"}, "hello"},
			{"text/csv", a2a.TextPart{Text: "region,count\neu,2"}, "region,count\neu,2"},
			{"text/markdown", a2a.TextPart{Text: "# Report"}, "# Report"},
			{"application/json", a2a.DataPart{Data: map[string]any{"region": "eu"}}, ""},
		} {
			name := "text-model/"
			if vision {
				name = "vision-model/"
			}

			t.Run(name+tc.mode, func(t *testing.T) {
				t.Parallel()

				model := fakellm.NewFakeModel(fakellm.WithCapabilities(llm.ModelCapabilities{Streaming: true, Vision: vision, Audio: vision}))
				model.When(fakellm.Any()).ThenStreamText("accepted", fakellm.StreamConfig{ChunkSize: 8})
				ag, err := llmagent.New("modes", "Inspect the input.", model)
				require.NoError(t, err)
				run, err := runner.New(ag, session.NewInMemoryStore())
				require.NoError(t, err)

				executor := NewExecutor(ag, run, nil)
				require.Equal(t, []string{"text/plain", "text/csv", "text/markdown", "application/json"}, executor.InputModes())
				require.Contains(t, executor.InputModes(), tc.mode)

				queue := &recordingQueue{}
				require.NoError(t, executor.Execute(context.Background(), &a2asrv.RequestContext{
					TaskID: "modes-task", ContextID: "modes-context",
					Message: a2a.NewMessage(a2a.MessageRoleUser, tc.part),
				}, queue))
				require.Equal(t, a2a.TaskStateCompleted, finalStatusEvent(t, queue.recorded()).Status.State)

				calls := model.Calls()
				require.Len(t, calls, 1)
				var userParts []llm.Part

				for _, msg := range calls[0].Request.Messages {
					if msg.Role == llm.RoleUser {
						userParts = append(userParts, msg.Content...)
					}
				}

				require.Len(t, userParts, 1)
				text, ok := userParts[0].(*llm.TextPart)
				require.True(t, ok, "advertised inputs must reach the model as text")

				if tc.mode == "application/json" {
					var data a2a.DataPart
					require.NoError(t, json.Unmarshal([]byte(text.Text), &data))
					require.Equal(t, map[string]any{"region": "eu"}, data.Data)
				} else {
					require.Equal(t, tc.text, text.Text)
				}
			})
		}
	}
}

func TestExecutorOutputModesMatchConversion(t *testing.T) {
	t.Parallel()

	executor := NewExecutor(nil, nil, nil)
	require.Equal(t, []string{"text/plain", "application/json"}, executor.OutputModes())

	msg := MessageFromLLM(llm.NewMessage(llm.RoleAssistant,
		llm.NewTextPart("answer"),
		llm.NewToolResponsePart("call-1", "report", json.RawMessage(`{"count":2}`), false),
	))
	require.Len(t, msg.Parts, 2)
	require.Equal(t, a2a.TextPart{Text: "answer"}, msg.Parts[0])
	data, ok := msg.Parts[1].(a2a.DataPart)
	require.True(t, ok)
	require.Equal(t, "tool_response", data.Metadata["data_type"])
	require.Equal(t, map[string]any{"count": float64(2)}, data.Data["result"])
}

func TestExecutorModesAreCallerOwned(t *testing.T) {
	t.Parallel()

	executor := NewExecutor(nil, nil, nil)
	inputs, outputs := executor.InputModes(), executor.OutputModes()
	require.NotEmpty(t, inputs)
	require.NotEmpty(t, outputs)
	inputs[0], outputs[0] = "image/png", "video/mp4"

	require.Equal(t, "text/plain", executor.InputModes()[0])
	require.Equal(t, "text/plain", executor.OutputModes()[0])
}

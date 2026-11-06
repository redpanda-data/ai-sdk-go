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

package openaicompat

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestReasoningControlsWire(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		opts     []Option
		effort   string
		thinking string
	}{
		{name: "unset stays omitted"},
		{name: "native effort", opts: []Option{WithReasoningEffort("high")}, effort: "high"},
		{name: "provider vocabulary", opts: []Option{WithReasoningEffort("adaptive")}, effort: "adaptive"},
		{name: "enabled", opts: []Option{WithThinking(true)}, thinking: "enabled"},
		{name: "disabled", opts: []Option{WithThinking(false)}, thinking: "disabled"},
		{name: "adaptive", opts: []Option{WithAdaptiveThinking()}, thinking: "adaptive"},
		{name: "adaptive replaces enabled", opts: []Option{WithThinking(true), WithAdaptiveThinking()}, thinking: "adaptive"},
		{name: "disabled replaces adaptive", opts: []Option{WithAdaptiveThinking(), WithThinking(false)}, thinking: "disabled"},
		{name: "enabled replaces adaptive", opts: []Option{WithAdaptiveThinking(), WithThinking(true)}, thinking: "enabled"},
		{name: "effort and adaptive", opts: []Option{WithReasoningEffort("high"), WithAdaptiveThinking()}, effort: "high", thinking: "adaptive"},
	}
	for _, tt := range tests {
		for _, streaming := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", tt.name, streaming), func(t *testing.T) {
				t.Parallel()

				requests := make(chan map[string]any, 1)
				server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					var body map[string]any
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}

					requests <- body

					if streaming {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = fmt.Fprint(w, "data: "+`{"id":"test","object":"chat.completion.chunk","created":1,"model":"custom-model","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\ndata: [DONE]\n\n")
					} else {
						w.Header().Set("Content-Type", "application/json")
						_, _ = fmt.Fprint(w, `{"id":"test","object":"chat.completion","created":1,"model":"custom-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
					}
				}))
				t.Cleanup(server.Close)

				provider, err := NewProvider("placeholder-key", WithBaseURL(server.URL))
				require.NoError(t, err)

				model, err := provider.NewModel("custom-model", tt.opts...)
				require.NoError(t, err)

				request := &llm.Request{Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello")),
				}}

				if streaming {
					var terminal int

					for event, streamErr := range model.GenerateEvents(t.Context(), request) {
						require.NoError(t, streamErr)

						if end, ok := event.(llm.StreamEndEvent); ok {
							terminal++

							require.NoError(t, end.Error)
						}
					}

					require.Equal(t, 1, terminal)
				} else {
					_, err = model.Generate(t.Context(), request)
					require.NoError(t, err)
				}

				body := <-requests
				if tt.effort == "" {
					assert.NotContains(t, body, "reasoning_effort")
				} else {
					assert.Equal(t, tt.effort, body["reasoning_effort"])
				}

				if tt.thinking == "" {
					assert.NotContains(t, body, "thinking")
				} else {
					assert.Equal(t, map[string]any{"type": tt.thinking}, body["thinking"])
				}
			})
		}
	}
}

func TestReasoningEffortConstraints(t *testing.T) {
	t.Parallel()

	unsupported := getDefaultConstraints()
	unsupported.SupportedParams = []string{"temperature"}
	conflicting := getDefaultConstraints()
	conflicting.MutuallyExclusive = [][]string{{"temperature", "reasoning_effort"}}

	tests := []struct {
		name string
		opts []Option
		want string
	}{
		{name: "empty effort", opts: []Option{WithReasoningEffort("")}, want: "invalid reasoning effort"},
		{name: "whitespace effort", opts: []Option{WithReasoningEffort(" ")}, want: "invalid reasoning effort"},
		{name: "padded effort", opts: []Option{WithReasoningEffort("high ")}, want: "invalid reasoning effort"},
		{name: "unsupported parameter", opts: []Option{WithConstraints(unsupported), WithReasoningEffort("high")}, want: "reasoning_effort is not supported"},
		{name: "late constraints", opts: []Option{WithReasoningEffort("high"), WithConstraints(unsupported)}, want: "reasoning_effort is not supported"},
		{name: "conflicting parameters", opts: []Option{WithConstraints(conflicting), WithTemperature(0), WithReasoningEffort("high")}, want: "cannot use"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("placeholder-key")
			require.NoError(t, err)

			_, err = provider.NewModel("custom-model", tt.opts...)
			require.ErrorContains(t, err, tt.want)
		})
	}
}

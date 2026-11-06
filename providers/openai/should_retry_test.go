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

package openai

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry/shouldretrytest"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestShouldRetryHeader(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		p, err := NewProvider("test-key", WithBaseURL(baseURL))
		require.NoError(t, err)

		m, err := p.NewModel(ModelGPT5_4)
		require.NoError(t, err)

		return m
	},
		shouldretrytest.Failure{
			Name:    "HTTP 503",
			Status:  http.StatusServiceUnavailable,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    []byte(`{"error":{"type":"server_error","code":"server_error","message":"every member failed"}}`),
		},
		shouldretrytest.Failure{
			Name:       "error event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("data: {\"error\":{\"type\":\"server_error\",\"code\":\"server_error\",\"message\":\"every member failed\"}}\n\n"),
			StreamOnly: true,
		},
		shouldretrytest.Failure{
			Name:       "Responses error event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("event: error\ndata: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"every member failed\",\"param\":null,\"sequence_number\":1}\n\n"),
			StreamOnly: true,
		},
		shouldretrytest.Failure{
			Name:       "response.failed on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("event: response.failed\ndata: {\"type\":\"response.failed\",\"sequence_number\":1,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"failed\",\"error\":{\"code\":\"server_error\",\"message\":\"every member failed\"},\"output\":[]}}\n\n"),
			StreamOnly: true,
		},
	)
}

// If a stream sends an error event and then completes, the error event stays a
// warning, and the call ends with the completed response.
func TestGenerateEvents_ErrorEventBeforeCompletionIsAWarning(t *testing.T) {
	t.Parallel()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("event: error\ndata: {\"type\":\"error\",\"code\":\"server_error\",\"message\":\"a hiccup\",\"param\":null,\"sequence_number\":1}\n\n" +
			"event: response.completed\ndata: {\"type\":\"response.completed\",\"sequence_number\":2,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"completed\",\"model\":\"gpt-5.4\",\"output\":[{\"type\":\"message\",\"id\":\"msg_1\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}]}}\n\n"))
	}))
	t.Cleanup(srv.Close)

	p, err := NewProvider("test-key", WithBaseURL(srv.URL))
	require.NoError(t, err)

	m, err := p.NewModel(ModelGPT5_4)
	require.NoError(t, err)

	var warned, ended bool

	for event, err := range m.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}}) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ErrorEvent:
			warned = true
		case llm.StreamEndEvent:
			require.NoError(t, e.Error)

			ended = true
		}
	}

	assert.True(t, warned, "the error event should reach the caller as an ErrorEvent")
	assert.True(t, ended, "the call should end with the completed response")
}

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
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestStreamSSECommentKeepalives asserts that SSE comment lines in the
// response body do not terminate the stream.
//
// OpenRouter emits ": OPENROUTER PROCESSING" keepalives while it queues a
// request, and some vLLM and LiteLLM deployments do the same. openai-go
// v3.42.0 skipped the comment line but then dispatched an event with an empty
// data buffer on the blank line that follows it, so json.Unmarshal saw no
// bytes and the stream died with "unexpected end of JSON input" before
// delivering a single chunk. Fixed in openai-go v3.43.0; this test fails
// against v3.42.0 and guards against a downgrade.
func TestStreamSSECommentKeepalives(t *testing.T) {
	t.Parallel()

	const chunks = ": OPENROUTER PROCESSING\n" +
		"\n" +
		": OPENROUTER PROCESSING\n" +
		"\n" +
		`data: {"id":"gen-1","object":"chat.completion.chunk","model":"m",` +
		`"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}` + "\n\n" +
		": OPENROUTER PROCESSING\n" +
		"\n" +
		`data: {"id":"gen-1","object":"chat.completion.chunk","model":"m",` +
		`"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}` + "\n\n" +
		"data: [DONE]\n\n"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(chunks))
	}))
	t.Cleanup(server.Close)

	provider, err := NewProvider("sk-test-key", WithBaseURL(server.URL))
	require.NoError(t, err)

	model, err := provider.NewModel("m")
	require.NoError(t, err)

	var (
		text  strings.Builder
		final *llm.Response
	)

	for event, err := range model.GenerateEvents(t.Context(), &llm.Request{
		Messages: []llm.Message{{
			Role:    llm.RoleUser,
			Content: []llm.Part{llm.NewTextPart("hi")},
		}},
	}) {
		require.NoError(t, err)

		switch e := event.(type) {
		case llm.ContentPartEvent:
			if p, ok := e.Part.(*llm.TextPart); ok {
				text.WriteString(p.Text)
			}
		case llm.StreamEndEvent:
			final = e.Response
		}
	}

	assert.Equal(t, "hi there", text.String())
	require.NotNil(t, final)
	assert.Equal(t, llm.FinishReasonStop, final.FinishReason)
}

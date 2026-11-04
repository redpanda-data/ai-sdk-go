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

package anthropic_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
)

// TestStreaming_FinalUsageFromMessageDelta checks that the usage reported in
// the final message_delta event (cumulative input and cache counters, which
// grow when server tools such as web search run) replaces the snapshot taken
// at message_start, as non-streaming Generate reports it.
func TestStreaming_FinalUsageFromMessageDelta(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		deltaUsage string
		want       llm.TokenUsage
	}{
		{
			name:       "delta carries cumulative input and cache counters",
			deltaUsage: `{"input_tokens":5120,"cache_read_input_tokens":64,"cache_creation_input_tokens":32,"output_tokens":30}`,
			want: llm.TokenUsage{
				InputTokens:                   5120,
				CachedInputTokens:             64,
				CacheCreationUnknownTTLTokens: 32,
				OutputTokens:                  30,
			},
		},
		{
			name:       "delta without input counters keeps message_start values",
			deltaUsage: `{"output_tokens":30}`,
			want: llm.TokenUsage{
				InputTokens:  120,
				OutputTokens: 30,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			sse := "event: message_start\n" +
				`data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-opus-4-5-20250929","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":120,"output_tokens":0}}}` + "\n\n" +
				"event: content_block_start\n" +
				`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
				"event: content_block_delta\n" +
				`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}` + "\n\n" +
				"event: content_block_stop\n" +
				`data: {"type":"content_block_stop","index":0}` + "\n\n" +
				"event: message_delta\n" +
				`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":` + tt.deltaUsage + `}` + "\n\n" +
				"event: message_stop\n" +
				`data: {"type":"message_stop"}` + "\n\n"

			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(sse))
			}))
			defer srv.Close()

			provider, err := anthropic.NewProvider(
				"test-key",
				anthropic.WithBaseURL(srv.URL),
				anthropic.WithHTTPClient(srv.Client()),
			)
			require.NoError(t, err)

			model, err := provider.NewModel(anthropic.ModelClaudeOpus45)
			require.NoError(t, err)

			req := &llm.Request{
				Messages: []llm.Message{
					{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("hello")}},
				},
			}

			var streamEnd *llm.StreamEndEvent

			for event, err := range model.GenerateEvents(context.Background(), req) {
				require.NoError(t, err)

				if e, ok := event.(llm.StreamEndEvent); ok {
					streamEnd = &e
				}
			}

			require.NotNil(t, streamEnd)
			require.NotNil(t, streamEnd.Response)
			require.NotNil(t, streamEnd.Response.Usage)
			assert.Equal(t, tt.want, *streamEnd.Response.Usage)
		})
	}
}

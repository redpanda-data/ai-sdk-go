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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
)

func messagesStream(start string, deltas []string, terminal bool) string {
	var out strings.Builder
	fmt.Fprintf(&out, "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"served-model\",\"content\":[],\"stop_reason\":null,\"usage\":%s}}\n\n", start)
	out.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"OK\"}}\n\nevent: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")

	for _, delta := range deltas {
		fmt.Fprintf(&out, "event: message_delta\ndata: %s\n\n", delta)
	}

	if terminal {
		out.WriteString("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}

	return out.String()
}

func messagesWireModel(t *testing.T, payload string, foreign bool) llm.Model {
	t.Helper()

	offering, ok := anthropic.Catalog().Lookup(anthropic.ModelClaudeOpus45)
	require.True(t, ok)

	if foreign {
		e := offering.Entry
		e.ID = "accounts/host/models/served-model"
		e.Aliases = nil
		e.Life.ReplacedBy = ""
		c := catalog.MustNew("foreign-host", []catalog.Entry{e})
		offering, ok = c.Lookup(e.ID)
		require.True(t, ok)
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/v1/messages", r.URL.Path)
		var request struct {
			Model string `json:"model"`
		}
		assert.NoError(t, json.NewDecoder(r.Body).Decode(&request))
		assert.Equal(t, offering.ID, request.Model)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(payload))
	}))
	t.Cleanup(server.Close)
	client := sdk.NewClient(option.WithAPIKey("synthetic-test-key"), option.WithBaseURL(server.URL), option.WithHTTPClient(server.Client()), option.WithMaxRetries(0))
	provider, err := anthropic.NewProviderWithClient(&client)
	require.NoError(t, err)

	provider.EnableCaching = false
	model, err := provider.NewModelFromOffering(offering.ID, offering, anthropic.WithMaxTokens(64))
	require.NoError(t, err)

	return model
}

func TestGenerateEventsFinalUsage(t *testing.T) {
	t.Parallel()

	// Fireworks' retained Messages captures have zero start usage and final
	// input/cache counts in message_delta (testdata/contracts/cache-crossover-v5.json).
	for _, tt := range []struct {
		name    string
		start   string
		deltas  []string
		foreign bool
		finish  llm.FinishReason
		want    llm.TokenUsage
	}{
		{
			name:    "foreign final input and cache",
			start:   `{"input_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}`,
			deltas:  []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":1,"cache_read_input_tokens":74,"cache_creation_input_tokens":0,"output_tokens":2}}`},
			foreign: true, want: llm.TokenUsage{InputTokens: 1, CachedInputTokens: 74, OutputTokens: 2},
		},
		{
			name:   "first-party output-only delta preserves input cache and TTL",
			start:  `{"input_tokens":12,"cache_read_input_tokens":8,"cache_creation_input_tokens":9,"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":5},"output_tokens":1}`,
			deltas: []string{`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":6}}`},
			finish: llm.FinishReasonLength,
			want:   llm.TokenUsage{InputTokens: 12, CachedInputTokens: 8, CacheCreation5mTokens: 4, CacheCreation1hTokens: 5, OutputTokens: 6},
		},
		{
			name:  "repeated cumulative counters replace instead of sum",
			start: `{"input_tokens":1,"cache_read_input_tokens":4,"output_tokens":1}`,
			deltas: []string{
				`{"type":"message_delta","delta":{"stop_reason":null},"usage":{"input_tokens":5,"cache_read_input_tokens":7,"output_tokens":3}}`,
				`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":5,"cache_read_input_tokens":7,"output_tokens":3}}`,
			},
			foreign: true, want: llm.TokenUsage{InputTokens: 5, CachedInputTokens: 7, OutputTokens: 3},
		},
		{
			name:   "explicit zero replaces old output and cache",
			start:  `{"input_tokens":12,"cache_read_input_tokens":8,"output_tokens":4}`,
			deltas: []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_read_input_tokens":0,"output_tokens":0}}`},
			want:   llm.TokenUsage{InputTokens: 12},
		},
		{
			name:    "foreign incomplete response keeps final usage",
			start:   `{"input_tokens":0,"output_tokens":0}`,
			deltas:  []string{`{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"input_tokens":24,"output_tokens":1}}`},
			foreign: true, finish: llm.FinishReasonLength, want: llm.TokenUsage{InputTokens: 24, OutputTokens: 1},
		},
		{
			name:   "null counters do not erase known counts",
			start:  `{"input_tokens":12,"cache_read_input_tokens":8,"output_tokens":4}`,
			deltas: []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":null,"cache_read_input_tokens":null,"output_tokens":null}}`},
			want:   llm.TokenUsage{InputTokens: 12, CachedInputTokens: 8, OutputTokens: 4},
		},
		{
			name:   "zero cache writes clear stale TTL breakdown",
			start:  `{"input_tokens":12,"cache_creation_input_tokens":9,"cache_creation":{"ephemeral_5m_input_tokens":4,"ephemeral_1h_input_tokens":5},"output_tokens":1}`,
			deltas: []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"cache_creation_input_tokens":0,"output_tokens":6}}`},
			want:   llm.TokenUsage{InputTokens: 12, OutputTokens: 6},
		},
		{
			name:    "omitted output preserves earlier output",
			start:   `{"input_tokens":1,"cache_read_input_tokens":4,"output_tokens":6}`,
			deltas:  []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":5}}`},
			foreign: true, want: llm.TokenUsage{InputTokens: 5, CachedInputTokens: 4, OutputTokens: 6},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := messagesWireModel(t, messagesStream(tt.start, tt.deltas, true), tt.foreign)
			var response *llm.Response

			for event, err := range model.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
				require.NoError(t, err)

				if end, ok := event.(llm.StreamEndEvent); ok {
					require.NoError(t, end.Error)
					response = end.Response
				}
			}

			require.NotNil(t, response)
			require.NotNil(t, response.Usage)
			assert.Equal(t, tt.want, *response.Usage)

			finish := tt.finish
			if finish == "" {
				finish = llm.FinishReasonStop
			}

			assert.Equal(t, finish, response.FinishReason)

			if tt.foreign {
				assert.Equal(t, llm.ProviderID("foreign-host"), model.Provider())
				assert.Equal(t, "accounts/host/models/served-model", response.InvokedModelID)
			}
		})
	}
}

func TestGenerateEventsRequiresMessageStop(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name    string
		payload string
	}{
		{"empty stream", ""},
		{"EOF before final usage", messagesStream(`{"input_tokens":2,"output_tokens":0}`, nil, false)},
		{"EOF after final usage", messagesStream(`{"input_tokens":2,"output_tokens":0}`, []string{`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`}, false)},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model := messagesWireModel(t, tt.payload, true)
			var gotErr error

			for event, err := range model.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
				if err != nil {
					gotErr = err
				}

				if end, ok := event.(llm.StreamEndEvent); ok {
					assert.Nil(t, end.Response, "premature EOF must not look like a completed response")
				}
			}

			assert.ErrorIs(t, gotErr, llm.ErrStreamClosed)
		})
	}
}

func TestGenerateEventsStreamErrorNotCompleted(t *testing.T) {
	t.Parallel()

	for _, terminal := range []bool{false, true} {
		t.Run(map[bool]string{false: "before stop", true: "after stop"}[terminal], func(t *testing.T) {
			t.Parallel()

			payload := messagesStream(`{"input_tokens":2,"output_tokens":0}`, nil, terminal) +
				"event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"synthetic overload\"}}\n\n"
			model := messagesWireModel(t, payload, true)
			var gotErr error

			for event, err := range model.GenerateEvents(t.Context(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
				if err != nil {
					gotErr = err
				}

				if end, ok := event.(llm.StreamEndEvent); ok {
					assert.Nil(t, end.Response)
				}
			}

			require.ErrorIs(t, gotErr, llm.ErrAPICall)
			assert.ErrorContains(t, gotErr, "synthetic overload")
		})
	}
}

func TestGenerateEventsCancellationRemainsClassified(t *testing.T) {
	t.Parallel()

	model := messagesWireModel(t, "", true)
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	var gotErr error

	for _, err := range model.GenerateEvents(ctx, &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("OK"))}}) {
		if err != nil {
			gotErr = err
		}
	}

	assert.ErrorIs(t, gotErr, context.Canceled)
}

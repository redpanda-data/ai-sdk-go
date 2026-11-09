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

// Package shouldretrytest tests that a provider stops retrying when the server
// sends x-should-retry: false. Its fake server answers every request with the
// same failure, which is an HTTP error or a 200 stream that carries an error
// event.
// Run sends requests through the retry plugin, once with the header and once
// without, and counts how many reach the server.
package shouldretrytest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
)

// Failure is a response the fake server sends for every request. Write the
// body in the provider's wire format, so its SDK parses it as an error from
// that provider.
type Failure struct {
	// Name labels the failure's subtests.
	Name    string
	Status  int
	Headers map[string]string
	Body    []byte
	// StreamOnly runs the failure for GenerateEvents only. Set it for a 200
	// whose body is a stream that carries an error event.
	StreamOnly bool
}

// NewModel builds the provider's model with baseURL as its endpoint. Without
// the header, the provider's own SDK retries an HTTP error as well, so set its
// backoff to a few milliseconds. Otherwise the test takes seconds.
type NewModel func(t *testing.T, baseURL string) llm.Model

// Run calls Generate and GenerateEvents through the retry plugin, against a
// fake server that sends each failure. When the response has
// x-should-retry: false, the server must get exactly one request, and the
// error must not be retryable. When it doesn't, the server must get more than
// one request. That second case shows that the header stops the retries, not
// the failure itself.
func Run(t *testing.T, newModel NewModel, failures ...Failure) {
	t.Helper()

	cases := []struct {
		name      string
		header    string
		wantRetry bool
	}{
		{name: "no retries with x-should-retry false", header: "false", wantRetry: false},
		{name: "retries without the header", header: "", wantRetry: true},
	}

	calls := []struct {
		name   string
		call   func(context.Context, llm.Model) error
		stream bool
	}{
		{name: "Generate", call: generate},
		{name: "GenerateEvents", call: generateEvents, stream: true},
	}

	for _, f := range failures {
		for _, tc := range cases {
			for _, c := range calls {
				if f.StreamOnly && !c.stream {
					continue
				}

				t.Run(f.Name+"/"+tc.name+"/"+c.name, func(t *testing.T) {
					t.Parallel()

					var hits atomic.Int32

					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
						hits.Add(1)

						for k, v := range f.Headers {
							w.Header().Set(k, v)
						}

						// The Anthropic and OpenAI SDKs wait this many
						// milliseconds before they retry, which keeps the case
						// without the header fast.
						w.Header().Set("Retry-After-Ms", "1")

						if tc.header != "" {
							w.Header().Set(shouldretry.Header, tc.header)
						}

						w.WriteHeader(f.Status)
						_, _ = w.Write(f.Body)
					}))
					t.Cleanup(srv.Close)

					model := retry.WrapModel(newModel(t, srv.URL),
						retry.WithMaxRetries(2),
						retry.WithInitialDelay(time.Millisecond),
						retry.WithMaxDelay(time.Millisecond),
					)

					err := c.call(t.Context(), model)
					require.Error(t, err)
					assert.Equal(t, tc.wantRetry, llm.IsRetryable(err), "IsRetryable(%v)", err)

					if tc.wantRetry {
						assert.Greater(t, hits.Load(), int32(1), "without the header, the failure should be retried")
					} else {
						assert.Equal(t, int32(1), hits.Load(), "with x-should-retry: false, the server should get exactly one request")
					}
				})
			}
		}

		for _, c := range calls {
			if f.StreamOnly && !c.stream {
				continue
			}

			t.Run(f.Name+"/x-should-retry false lasts one call/"+c.name, func(t *testing.T) {
				t.Parallel()
				runOneCallOnly(t, newModel, f, c.call)
			})
		}
	}
}

// runOneCallOnly sends two calls through one model. Only the first response
// has x-should-retry: false, so the first call must not be retried and the
// second must be. This fails if the flag outlives the call that got it.
func runOneCallOnly(t *testing.T, newModel NewModel, f Failure, call func(context.Context, llm.Model) error) {
	t.Helper()

	var hits atomic.Int32

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		n := hits.Add(1)

		for k, v := range f.Headers {
			w.Header().Set(k, v)
		}

		w.Header().Set("Retry-After-Ms", "1")

		if n == 1 {
			w.Header().Set(shouldretry.Header, "false")
		}

		w.WriteHeader(f.Status)
		_, _ = w.Write(f.Body)
	}))
	t.Cleanup(srv.Close)

	model := retry.WrapModel(newModel(t, srv.URL),
		retry.WithMaxRetries(2),
		retry.WithInitialDelay(time.Millisecond),
		retry.WithMaxDelay(time.Millisecond),
	)

	err := call(t.Context(), model)
	require.Error(t, err)
	assert.False(t, llm.IsRetryable(err), "the first call's error should not be retryable")
	require.Equal(t, int32(1), hits.Load(), "the first call should not be retried")

	err = call(t.Context(), model)
	require.Error(t, err)
	assert.True(t, llm.IsRetryable(err), "the second call's response has no header, so its error should be retryable")
	assert.Greater(t, hits.Load(), int32(2), "the second call should be retried")
}

func generate(ctx context.Context, model llm.Model) error {
	_, err := model.Generate(ctx, request())

	return err
}

func generateEvents(ctx context.Context, model llm.Model) error {
	for _, err := range model.GenerateEvents(ctx, request()) {
		if err != nil {
			return err
		}
	}

	return nil
}

func request() *llm.Request {
	return &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}}
}

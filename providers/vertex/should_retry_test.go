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

package vertex_test

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry/shouldretrytest"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func newTestModel(t *testing.T, baseURL, model string) llm.Model {
	t.Helper()

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("us-east5"),
		vertex.WithBaseURL(baseURL+"/llm/v1/providers/my-vertex"),
		vertex.WithHTTPClient(&http.Client{}),
	)
	require.NoError(t, err)

	m, err := p.NewModel(model)
	require.NoError(t, err)

	return m
}

// Vertex Claude builds its own Anthropic client and passes it to
// anthropic.NewProviderWithClient, so it doesn't go through the path the
// anthropic tests cover.
func TestShouldRetryHeader_Claude(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		return newTestModel(t, baseURL, vertex.ModelClaudeHaiku45)
	},
		shouldretrytest.Failure{
			Name:    "HTTP 503",
			Status:  http.StatusServiceUnavailable,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    []byte(`{"type":"error","error":{"type":"overloaded_error","message":"every member failed"}}`),
		},
		shouldretrytest.Failure{
			Name:       "error event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"every member failed\"}}\n\n"),
			StreamOnly: true,
		},
	)
}

func TestShouldRetryHeader(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		return newTestModel(t, baseURL, vertex.ModelGemini38Flash)
	},
		shouldretrytest.Failure{
			Name:    "HTTP 503",
			Status:  http.StatusServiceUnavailable,
			Headers: map[string]string{"Content-Type": "application/json"},
			Body:    []byte(`{"error":{"code":503,"message":"every member failed","status":"UNAVAILABLE"}}`),
		},
		shouldretrytest.Failure{
			Name:       "error event on a 200 stream",
			Status:     http.StatusOK,
			Headers:    map[string]string{"Content-Type": "text/event-stream"},
			Body:       []byte("data: {\"error\":{\"code\":503,\"message\":\"every member failed\",\"status\":\"UNAVAILABLE\"}}\n\n"),
			StreamOnly: true,
		},
	)
}

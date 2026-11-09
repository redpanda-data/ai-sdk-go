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

package google

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry/shouldretrytest"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

var geminiFailures = []shouldretrytest.Failure{
	{
		Name:    "HTTP 503",
		Status:  http.StatusServiceUnavailable,
		Headers: map[string]string{"Content-Type": "application/json"},
		Body:    []byte(`{"error":{"code":503,"message":"every member failed","status":"UNAVAILABLE"}}`),
	},
	{
		Name:       "error event on a 200 stream",
		Status:     http.StatusOK,
		Headers:    map[string]string{"Content-Type": "text/event-stream"},
		Body:       []byte("data: {\"error\":{\"code\":503,\"message\":\"every member failed\",\"status\":\"UNAVAILABLE\"}}\n\n"),
		StreamOnly: true,
	},
	{
		Name:       "CRLF error event followed by another event",
		Status:     http.StatusOK,
		Headers:    map[string]string{"Content-Type": "text/event-stream"},
		Body:       []byte("data: {\"error\":{\"code\":503,\"message\":\"every member failed\",\"status\":\"UNAVAILABLE\"}}\r\n\r\n: keep-alive\r\n\r\n"),
		StreamOnly: true,
	},
}

func TestShouldRetryHeader(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		p, err := NewProvider(context.Background(), "test-key", WithBaseURL(baseURL))
		require.NoError(t, err)

		m, err := p.NewModel(ModelGemini38Flash)
		require.NoError(t, err)

		return m
	}, geminiFailures...)
}

func TestShouldRetryHeader_ProviderWithClient(t *testing.T) {
	t.Parallel()

	shouldretrytest.Run(t, func(t *testing.T, baseURL string) llm.Model {
		t.Helper()

		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:      "test-key",
			Backend:     genai.BackendGeminiAPI,
			HTTPOptions: genai.HTTPOptions{BaseURL: baseURL},
		})
		require.NoError(t, err)

		p, err := NewProviderWithClient(context.Background(), client)
		require.NoError(t, err)

		m, err := p.NewModel(ModelGemini38Flash)
		require.NoError(t, err)

		return m
	}, geminiFailures...)
}

// On a machine with GOOGLE_CLOUD_PROJECT set, genai puts the project into a
// Gemini API client's settings. NewProviderWithClient must still be able to
// build its copy of that client.
func TestProviderWithClient_ProjectFromEnv(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "my-project")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  "test-key",
		Backend: genai.BackendGeminiAPI,
	})
	require.NoError(t, err)

	_, err = NewProviderWithClient(context.Background(), client)
	require.NoError(t, err)
}

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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

// capturedRequest is what the fake Vertex endpoint saw.
type capturedRequest struct {
	Method        string
	Path          string
	RawQuery      string
	Authorization string
	APIKey        string
	Body          string
}

// fakeVertex serves one canned response body for every request and records
// each request it receives.
type fakeVertex struct {
	server *httptest.Server

	mu       sync.Mutex
	requests []capturedRequest
}

func newFakeVertex(t *testing.T, contentType, body string) *fakeVertex {
	t.Helper()

	f := &fakeVertex{}
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)

		f.mu.Lock()
		f.requests = append(f.requests, capturedRequest{
			Method:        r.Method,
			Path:          r.URL.Path,
			RawQuery:      r.URL.RawQuery,
			Authorization: r.Header.Get("Authorization"),
			APIKey:        r.Header.Get("X-Api-Key"),
			Body:          string(b),
		})
		f.mu.Unlock()

		w.Header().Set("Content-Type", contentType)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(f.server.Close)

	return f
}

func (f *fakeVertex) only(t *testing.T) capturedRequest {
	t.Helper()

	f.mu.Lock()
	defer f.mu.Unlock()

	require.Len(t, f.requests, 1)

	return f.requests[0]
}

// tenantToken is the bearer bearerClient sends.
const tenantToken = "tenant-token"

// bearerClient stands in for the AI Gateway client a managed agent holds: it
// stamps the tenant's bearer token on every request.
func bearerClient() *http.Client {
	return &http.Client{Transport: bearerTransport{token: tenantToken}}
}

type bearerTransport struct{ token string }

func (b bearerTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("Authorization", "Bearer "+b.token)

	return http.DefaultTransport.RoundTrip(r)
}

// hello is a one-message user request.
func hello() *llm.Request {
	return &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}}
}

const geminiResponse = `{
  "candidates": [{"content": {"role": "model", "parts": [{"text": "hello from gemini"}]}, "finishReason": "STOP"}],
  "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}
}`

func TestNewModel_GeminiThroughGateway(t *testing.T) {
	t.Parallel()

	fake := newFakeVertex(t, "application/json", geminiResponse)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("us-east5"),
		vertex.WithBaseURL(fake.server.URL+"/llm/v1/providers/my-vertex"),
		vertex.WithHTTPClient(bearerClient()),
	)
	require.NoError(t, err)

	m, err := p.NewModel(vertex.ModelGemini25Flash)
	require.NoError(t, err)
	assert.Equal(t, llm.ProviderID("gcp.vertex"), m.Provider())
	assert.Equal(t, vertex.ModelGemini25Flash, m.Name())

	resp, err := m.Generate(context.Background(), hello())
	require.NoError(t, err)
	assert.Equal(t, "hello from gemini", resp.TextContent())

	got := fake.only(t)
	assert.Equal(t, http.MethodPost, got.Method)
	assert.Equal(t, "/llm/v1/providers/my-vertex/v1/projects/my-project/locations/us-east5/publishers/google/models/gemini-2.5-flash:generateContent", got.Path)
	assert.Equal(t, "Bearer tenant-token", got.Authorization)
}

// TestNewModel_GeminiSendsCatalogID checks a Gemini model's request path
// names the catalog's ID, not the name the caller resolved it by, as the
// Claude path does.
func TestNewModel_GeminiSendsCatalogID(t *testing.T) {
	t.Parallel()

	fake := newFakeVertex(t, "application/json", geminiResponse)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("us-east5"),
		vertex.WithBaseURL(fake.server.URL+"/llm/v1/providers/my-vertex"),
		vertex.WithHTTPClient(bearerClient()),
	)
	require.NoError(t, err)

	m, err := p.NewModel(vertex.ModelGemini25Flash + "-001")
	require.NoError(t, err)
	assert.Equal(t, vertex.ModelGemini25Flash+"-001", m.Name())

	_, err = m.Generate(context.Background(), hello())
	require.NoError(t, err)

	got := fake.only(t)
	assert.Equal(t, "/llm/v1/providers/my-vertex/v1/projects/my-project/locations/us-east5/publishers/google/models/gemini-2.5-flash:generateContent", got.Path)
}

// TestNewModel_NeverDetectsADC proves a negative: building and calling a
// model never runs Application Default Credentials detection. ADC is
// poisoned with a credentials file that does not exist, so detection would
// fail the build; the request must still go out with only the caller's
// bearer.
func TestNewModel_NeverDetectsADC(t *testing.T) {
	// Not parallel: t.Setenv refuses a parallel test.
	t.Setenv("GOOGLE_APPLICATION_CREDENTIALS", t.TempDir()+"/does-not-exist.json")

	fake := newFakeVertex(t, "application/json", geminiResponse)

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("global"),
		vertex.WithBaseURL(fake.server.URL),
		vertex.WithHTTPClient(bearerClient()),
	)
	require.NoError(t, err)

	m, err := p.NewModel(vertex.ModelGemini25Flash)
	require.NoError(t, err)

	_, err = m.Generate(context.Background(), hello())
	require.NoError(t, err)
	assert.Equal(t, "Bearer tenant-token", fake.only(t).Authorization)
}

func TestNewModel_RefusesIncompleteProvider(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []vertex.ProviderOption
		want string
	}{
		{
			name: "no HTTP client, which would let genai detect ADC",
			opts: []vertex.ProviderOption{vertex.WithProject("my-project"), vertex.WithLocation("global")},
			want: "WithHTTPClient",
		},
		{
			name: "no project",
			opts: []vertex.ProviderOption{vertex.WithLocation("global"), vertex.WithHTTPClient(http.DefaultClient)},
			want: "WithProject",
		},
		{
			name: "no location",
			opts: []vertex.ProviderOption{vertex.WithProject("my-project"), vertex.WithHTTPClient(http.DefaultClient)},
			want: "WithLocation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			p, err := vertex.NewProvider(context.Background(), tt.opts...)
			require.NoError(t, err)

			_, err = p.NewModel(vertex.ModelGemini25Flash)
			require.ErrorContains(t, err, tt.want)
		})
	}
}

const claudeResponse = `{
  "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
  "content": [{"type": "text", "text": "hello from claude"}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 3, "output_tokens": 4}
}`

const claudeStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello from claude"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

`

func newGatewayClaude(t *testing.T, fake *fakeVertex) llm.Model {
	t.Helper()

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("us-east5"),
		vertex.WithBaseURL(fake.server.URL+"/llm/v1/providers/my-vertex"),
		vertex.WithHTTPClient(bearerClient()),
	)
	require.NoError(t, err)

	m, err := p.NewModel(vertex.ModelClaudeHaiku45)
	require.NoError(t, err)

	return m
}

func TestNewModel_ClaudeThroughGateway(t *testing.T) {
	t.Parallel()

	fake := newFakeVertex(t, "application/json", claudeResponse)
	m := newGatewayClaude(t, fake)
	assert.Equal(t, llm.ProviderID("gcp.vertex"), m.Provider())
	assert.Equal(t, vertex.ModelClaudeHaiku45, m.Name())

	resp, err := m.Generate(context.Background(), hello())
	require.NoError(t, err)
	assert.Equal(t, "hello from claude", resp.TextContent())

	got := fake.only(t)
	assert.Equal(t, http.MethodPost, got.Method)
	assert.Equal(t, "/llm/v1/providers/my-vertex/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-haiku-4-5:rawPredict", got.Path)
	assert.Equal(t, "Bearer tenant-token", got.Authorization)
	assert.Empty(t, got.RawQuery, "the Anthropic SDK's ?beta=true is not a Vertex parameter")
	assert.False(t, gjson.Get(got.Body, "model").Exists(), "Vertex takes the model from the path and refuses it in the body")
	assert.Equal(t, "vertex-2023-10-16", gjson.Get(got.Body, "anthropic_version").String())
}

func TestNewModel_ClaudeStreamsThroughGateway(t *testing.T) {
	t.Parallel()

	fake := newFakeVertex(t, "text/event-stream", claudeStream)
	m := newGatewayClaude(t, fake)

	var text strings.Builder

	for event, err := range m.GenerateEvents(context.Background(), hello()) {
		require.NoError(t, err)

		if delta, ok := event.(llm.ContentPartEvent); ok {
			text.WriteString(llm.JoinTextParts([]llm.Part{delta.Part}))
		}
	}

	assert.Equal(t, "hello from claude", text.String())
	assert.Equal(t, "/llm/v1/providers/my-vertex/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-haiku-4-5:streamRawPredict", fake.only(t).Path)
}

func TestNewModel_GeminiStreamsThroughGateway(t *testing.T) {
	t.Parallel()

	fake := newFakeVertex(t, "text/event-stream", "data: "+strings.ReplaceAll(geminiResponse, "\n", "")+"\n\n")

	p, err := vertex.NewProvider(context.Background(),
		vertex.WithProject("my-project"),
		vertex.WithLocation("us-east5"),
		vertex.WithBaseURL(fake.server.URL+"/llm/v1/providers/my-vertex"),
		vertex.WithHTTPClient(bearerClient()),
	)
	require.NoError(t, err)

	m, err := p.NewModel(vertex.ModelGemini25Flash)
	require.NoError(t, err)

	var text strings.Builder

	for event, err := range m.GenerateEvents(context.Background(), hello()) {
		require.NoError(t, err)

		if delta, ok := event.(llm.ContentPartEvent); ok {
			text.WriteString(llm.JoinTextParts([]llm.Part{delta.Part}))
		}
	}

	assert.Equal(t, "hello from gemini", text.String())

	got := fake.only(t)
	assert.Equal(t, "/llm/v1/providers/my-vertex/v1/projects/my-project/locations/us-east5/publishers/google/models/gemini-2.5-flash:streamGenerateContent", got.Path)
	assert.Equal(t, "alt=sse", got.RawQuery)
}

// TestNewModel_ClaudeIgnoresAmbientAnthropicKey proves an ANTHROPIC_API_KEY
// in the environment, which the Anthropic SDK reads by default, never
// reaches Vertex or the gateway.
func TestNewModel_ClaudeIgnoresAmbientAnthropicKey(t *testing.T) {
	// Not parallel: t.Setenv refuses a parallel test.
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
	t.Setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")

	fake := newFakeVertex(t, "application/json", claudeResponse)

	_, err := newGatewayClaude(t, fake).Generate(context.Background(), hello())
	require.NoError(t, err)

	got := fake.only(t)
	assert.Empty(t, got.APIKey)
	assert.Equal(t, "Bearer tenant-token", got.Authorization)
}

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

package mcp

import (
	"context"
	"net/http"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

func TestNewStdioTransport(t *testing.T) {
	t.Parallel()

	t.Run("success", func(t *testing.T) {
		t.Parallel()

		factory := NewStdioTransport("echo", []string{"hello"}, nil)
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})

	t.Run("with environment", func(t *testing.T) {
		t.Parallel()

		factory := NewStdioTransport("echo", []string{"test"}, []string{"FOO=bar"})
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})

	t.Run("command not found", func(t *testing.T) {
		t.Parallel()

		factory := NewStdioTransport("nonexistent-command-xyz", nil, nil)
		_, err := factory()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "command not found")
	})
}

func TestNewStreamableTransport(t *testing.T) {
	t.Parallel()

	t.Run("default client", func(t *testing.T) {
		t.Parallel()

		factory := NewStreamableTransport("https://example.com/mcp")
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})

	t.Run("with custom HTTP client", func(t *testing.T) {
		t.Parallel()

		customClient := &http.Client{}
		factory := NewStreamableTransport("https://example.com/mcp",
			WithHTTPClient(customClient))
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})

	t.Run("with OAuth", func(t *testing.T) {
		t.Parallel()

		ctx := context.Background()
		oauthConfig := &oauth2.Config{
			ClientID: "test-client",
			Endpoint: oauth2.Endpoint{
				AuthURL:  "https://example.com/auth",
				TokenURL: "https://example.com/token",
			},
		}
		factory := NewStreamableTransport("https://example.com/mcp",
			WithOAuth(ctx, oauthConfig))
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})
}

func TestNewSSETransport(t *testing.T) {
	t.Parallel()

	t.Run("default client", func(t *testing.T) {
		t.Parallel()

		factory := NewSSETransport("https://example.com/sse")
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})

	t.Run("with custom HTTP client", func(t *testing.T) {
		t.Parallel()

		customClient := &http.Client{}
		factory := NewSSETransport("https://example.com/sse",
			WithHTTPClient(customClient))
		transport, err := factory()
		require.NoError(t, err)
		assert.NotNil(t, transport)
	})
}

func TestWithHTTPHeaders(t *testing.T) {
	t.Parallel()

	t.Run("headers are applied to streamable transport", func(t *testing.T) {
		t.Parallel()

		headers := map[string]string{
			"X-API-Key":      "test-key",
			"X-Custom-Value": "custom",
		}
		factory := NewStreamableTransport("https://example.com", WithHTTPHeaders(headers))
		transport, err := factory()
		require.NoError(t, err)
		require.NotNil(t, transport)

		// Verify transport type and check if headers would be applied
		streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
		require.True(t, ok)
		require.NotNil(t, streamable.HTTPClient)

		// Test that headers are injected via RoundTripper
		rt := streamable.HTTPClient.Transport
		require.NotNil(t, rt)
		hrt, ok := rt.(*headerRoundTripper)
		require.True(t, ok, "transport should be wrapped with headerRoundTripper")
		assert.Equal(t, "test-key", hrt.headers["X-API-Key"])
		assert.Equal(t, "custom", hrt.headers["X-Custom-Value"])
	})

	t.Run("multiple WithHTTPHeaders calls merge headers", func(t *testing.T) {
		t.Parallel()

		factory := NewStreamableTransport("https://example.com",
			WithHTTPHeaders(map[string]string{"X-Key-1": "value1"}),
			WithHTTPHeaders(map[string]string{"X-Key-2": "value2"}),
		)
		transport, err := factory()
		require.NoError(t, err)

		streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
		require.True(t, ok, "expected StreamableClientTransport")
		hrt, ok := streamable.HTTPClient.Transport.(*headerRoundTripper)
		require.True(t, ok, "expected headerRoundTripper")
		assert.Equal(t, "value1", hrt.headers["X-Key-1"])
		assert.Equal(t, "value2", hrt.headers["X-Key-2"])
	})

	t.Run("works with custom HTTP client", func(t *testing.T) {
		t.Parallel()

		customClient := &http.Client{Timeout: 5 * time.Second}
		factory := NewStreamableTransport("https://example.com",
			WithHTTPClient(customClient),
			WithHTTPHeaders(map[string]string{"X-API-Key": "test"}),
		)
		transport, err := factory()
		require.NoError(t, err)

		streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
		require.True(t, ok, "expected StreamableClientTransport")
		assert.Equal(t, 5*time.Second, streamable.HTTPClient.Timeout)

		// Verify headers are wrapped
		hrt, ok := streamable.HTTPClient.Transport.(*headerRoundTripper)
		require.True(t, ok)
		assert.Equal(t, "test", hrt.headers["X-API-Key"])
	})

	t.Run("works with SSE transport", func(t *testing.T) {
		t.Parallel()

		factory := NewSSETransport("https://example.com",
			WithHTTPHeaders(map[string]string{"Authorization": "Bearer token"}),
		)
		transport, err := factory()
		require.NoError(t, err)

		sse, ok := transport.(*sdkmcp.SSEClientTransport)
		require.True(t, ok, "expected SSEClientTransport")
		hrt, ok := sse.HTTPClient.Transport.(*headerRoundTripper)
		require.True(t, ok, "expected headerRoundTripper")
		assert.Equal(t, "Bearer token", hrt.headers["Authorization"])
	})

	t.Run("later headers override earlier ones", func(t *testing.T) {
		t.Parallel()

		factory := NewStreamableTransport("https://example.com",
			WithHTTPHeaders(map[string]string{"X-Key": "first"}),
			WithHTTPHeaders(map[string]string{"X-Key": "second"}),
		)
		transport, err := factory()
		require.NoError(t, err)

		streamable, ok := transport.(*sdkmcp.StreamableClientTransport)
		require.True(t, ok, "expected StreamableClientTransport")
		hrt, ok := streamable.HTTPClient.Transport.(*headerRoundTripper)
		require.True(t, ok, "expected headerRoundTripper")
		assert.Equal(t, "second", hrt.headers["X-Key"])
	})
}

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
	"io"
	"net/http"
	"net/http/cookiejar"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

func TestRegistryTransportRefreshesCredentialsAndClonesRequests(t *testing.T) {
	t.Parallel()

	var token atomic.Int32
	token.Store(1)
	var calls atomic.Int32
	source := oauth2.TokenSource(tokenSourceFunc(func() (*oauth2.Token, error) {
		calls.Add(1)

		return &oauth2.Token{AccessToken: string('0' + token.Load())}, nil
	}))
	headers := http.Header{"X-Api-Key": {"shared"}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		assert.Equal(t, "shared", req.Header.Get("X-Api-Key"))
		assert.Equal(t, "Bearer "+string('0'+token.Load()), req.Header.Get("Authorization"))
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)

	registry, err := NewRegistry("test", server.URL, WithCredentials(func(context.Context) (Credentials, error) {
		return Credentials{PartitionKey: "same", Headers: headers, TokenSource: source}, nil
	}))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL, nil)
	require.NoError(t, err)
	req.Header.Set("X-Original", "untouched")

	for _, value := range []int32{1, 2} {
		token.Store(value)

		resp, err := registry.httpClient("same").Do(req)
		require.NoError(t, err)
		require.NoError(t, resp.Body.Close())
	}

	assert.EqualValues(t, 2, calls.Load())
	assert.Equal(t, http.Header{"X-Api-Key": {"shared"}}, headers)
	assert.Equal(t, http.Header{"X-Original": {"untouched"}}, req.Header)
}

func TestRegistryTransportRejectsCrossOriginRedirects(t *testing.T) {
	t.Parallel()

	var leaked atomic.Bool
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		leaked.Store(true)
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(other.Close)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, other.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(server.Close)

	registry, err := NewRegistry("test", server.URL, WithCredentials(func(context.Context) (Credentials, error) {
		return Credentials{PartitionKey: "public", Headers: http.Header{"Authorization": {"Bearer secret"}}}, nil
	}))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	resp, err := registry.httpClient("public").Do(req)

	if resp != nil {
		require.NoError(t, resp.Body.Close())
	}

	require.ErrorContains(t, err, "cross-origin redirect refused")
	assert.False(t, leaked.Load())
}

func TestRegistryTransportChecksPartitionOnEveryRequest(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)

	registry, err := NewRegistry("test", server.URL, WithCredentials(func(context.Context) (Credentials, error) {
		return Credentials{PartitionKey: "different"}, nil
	}))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })

	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, server.URL, nil)
	require.NoError(t, err)

	resp, err := registry.httpClient("captured").Do(req)

	if resp != nil {
		require.NoError(t, resp.Body.Close())
	}

	require.ErrorIs(t, err, ErrPartitionMismatch)
	assert.Zero(t, requests.Load())
}

func TestRegistryConstructionValidation(t *testing.T) {
	t.Parallel()

	jar, err := cookiejar.New(nil)
	require.NoError(t, err)

	provider := WithCredentials(func(context.Context) (Credentials, error) { return Credentials{PartitionKey: "public"}, nil })

	tests := []struct {
		name     string
		serverID string
		endpoint string
		opts     []RegistryOption
	}{
		{name: "missing server ID", endpoint: "http://example.com", opts: []RegistryOption{provider}},
		{name: "relative endpoint", serverID: "test", endpoint: "/mcp", opts: []RegistryOption{provider}},
		{name: "userinfo", serverID: "test", endpoint: "http://user@example.com", opts: []RegistryOption{provider}},
		{name: "missing provider", serverID: "test", endpoint: "http://example.com"},
		{name: "nil option", serverID: "test", endpoint: "http://example.com", opts: []RegistryOption{provider, nil}},
		{name: "shared cookies", serverID: "test", endpoint: "http://example.com", opts: []RegistryOption{provider, WithRegistryHTTPClient(&http.Client{Jar: jar})}},
		{name: "zero capacity", serverID: "test", endpoint: "http://example.com", opts: []RegistryOption{provider, WithRegistryMaxConnections(0)}},
		{name: "negative idle", serverID: "test", endpoint: "http://example.com", opts: []RegistryOption{provider, WithRegistryIdleTimeout(-time.Second)}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry, err := NewRegistry(tt.serverID, tt.endpoint, tt.opts...)
			require.Error(t, err)
			assert.Nil(t, registry)
		})
	}
}

func TestRegistryTransportUsesCustomHTTPClient(t *testing.T) {
	t.Parallel()

	var requests atomic.Int32
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		requests.Add(1)
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)

	client := server.Client()
	transport := client.Transport
	registry, err := NewRegistry("test", server.URL,
		WithCredentials(func(context.Context) (Credentials, error) { return Credentials{PartitionKey: "public"}, nil }),
		WithRegistryHTTPClient(client))
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, registry.Close()) })

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, nil)
	require.NoError(t, err)
	resp, err := registry.httpClient("public").Do(req)
	require.NoError(t, err)
	_, err = io.Copy(io.Discard, resp.Body)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	assert.EqualValues(t, 1, requests.Load())
	assert.Equal(t, transport, client.Transport)
}

type tokenSourceFunc func() (*oauth2.Token, error)

func (f tokenSourceFunc) Token() (*oauth2.Token, error) { return f() }

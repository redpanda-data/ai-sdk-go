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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

type oauthContractSourceKey struct{}

func TestContextRegistryOAuthRefreshOverRealMCP(t *testing.T) {
	t.Parallel()

	peer := newRegistryPeer(t, false)
	var refreshes atomic.Int32
	tokenEndpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		req.Body = http.MaxBytesReader(w, req.Body, 4096)
		if err := req.ParseForm(); err != nil {
			http.Error(w, "invalid token request", http.StatusBadRequest)
			return
		}

		client, secret, ok := req.BasicAuth()
		if !ok || client != "test-client" || secret != "test-secret" || req.Form.Get("grant_type") != "refresh_token" || req.Form.Get("refresh_token") != "test-refresh" {
			http.Error(w, "invalid refresh credentials", http.StatusUnauthorized)
			return
		}

		version := refreshes.Add(1)

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token": fmt.Sprintf("alice:refreshed-%d", version),
			"token_type":   "Bearer", "expires_in": 3600, "refresh_token": "test-refresh",
		})
	}))
	t.Cleanup(tokenEndpoint.Close)

	config := &oauth2.Config{
		ClientID: "test-client", ClientSecret: "test-secret",
		Endpoint: oauth2.Endpoint{TokenURL: tokenEndpoint.URL, AuthStyle: oauth2.AuthStyleInHeader},
	}
	newSource := func() oauth2.TokenSource {
		return config.TokenSource(t.Context(), &oauth2.Token{
			AccessToken: "expired", RefreshToken: "test-refresh", Expiry: time.Now().Add(-time.Hour),
		})
	}
	firstContext := context.WithValue(t.Context(), oauthContractSourceKey{}, newSource())
	secondContext := context.WithValue(t.Context(), oauthContractSourceKey{}, newSource())
	registry, err := NewRegistry("remote", peer.url, WithCredentials(func(ctx context.Context) (Credentials, error) {
		source, ok := ctx.Value(oauthContractSourceKey{}).(oauth2.TokenSource)
		if !ok {
			return Credentials{}, errors.New("missing trusted OAuth source")
		}

		return NewCredentials(WithPartitionKey("tenant/alice"), WithTokenSource(source))
	}))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, registry.Close()) })
	assert.Zero(t, refreshes.Load(), "registry construction must not resolve OAuth tokens")

	captured, err := registry.Snapshot(firstContext)
	require.NoError(t, err)
	assert.Equal(t, int32(1), refreshes.Load())

	first, err := captured.Execute(firstContext, registryRequest("remote__shared"))
	require.NoError(t, err)
	require.False(t, first.IsError)
	assert.Contains(t, string(first.Result), "Bearer alice:refreshed-1 shared")
	assert.Equal(t, int32(1), refreshes.Load(), "x/oauth2 must reuse its still-valid token")

	second, err := captured.Execute(secondContext, registryRequest("remote__shared"))
	require.NoError(t, err)
	require.False(t, second.IsError)
	assert.Contains(t, string(second.Result), "Bearer alice:refreshed-2 shared")
	assert.Equal(t, int32(2), refreshes.Load())

	calls := peer.requests("tools/call")
	require.Len(t, calls, 2)
	assert.Equal(t, "Bearer alice:refreshed-1", calls[0].auth)
	assert.Equal(t, "Bearer alice:refreshed-2", calls[1].auth)
	registry.mu.Lock()
	assert.Len(t, registry.entries, 1, "changed OAuth credentials must not secretly change the application partition key")
	registry.mu.Unlock()
}

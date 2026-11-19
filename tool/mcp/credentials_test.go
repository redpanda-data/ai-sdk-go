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
	"encoding/base64"
	"errors"
	"net/http"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
)

func TestCredentialOptions(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		opts []CredentialOption
		want http.Header
	}{
		{name: "explicit anonymous", want: http.Header{}},
		{name: "bearer", opts: []CredentialOption{WithBearerToken("token")}, want: http.Header{"Authorization": {"Bearer token"}}},
		{name: "API key", opts: []CredentialOption{WithAPIKey("x-api-key", "key")}, want: http.Header{"X-Api-Key": {"key"}}},
		{name: "basic", opts: []CredentialOption{WithBasicAuth("user", "password")}, want: http.Header{"Authorization": {"Basic " + base64.StdEncoding.EncodeToString([]byte("user:password"))}}},
		{name: "multiple headers", opts: []CredentialOption{WithBearerToken("token"), WithAPIKey("X-Key", "key"), WithHeaders(http.Header{"x-delegation": {"one", "two"}})}, want: http.Header{"Authorization": {"Bearer token"}, "X-Key": {"key"}, "X-Delegation": {"one", "two"}}},
		{name: "bearer replaces token source", opts: []CredentialOption{WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "old"})), WithBearerToken("new")}, want: http.Header{"Authorization": {"Bearer new"}}},
		{name: "token source replaces basic", opts: []CredentialOption{WithBasicAuth("user", "password"), WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "token"}))}, want: http.Header{"Authorization": {"Bearer token"}}},
		{name: "headers replace token source", opts: []CredentialOption{WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "old"})), WithHeaders(http.Header{"authorization": {"Custom token"}})}, want: http.Header{"Authorization": {"Custom token"}}},
		{name: "OAuth token type", opts: []CredentialOption{WithTokenSource(oauth2.StaticTokenSource(&oauth2.Token{AccessToken: "token", TokenType: "Custom"}))}, want: http.Header{"Authorization": {"Custom token"}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			credentials, err := NewCredentials(append([]CredentialOption{WithPartitionKey("tenant/user")}, tc.opts...)...)
			require.NoError(t, err)
			assert.Equal(t, "tenant/user", credentials.PartitionKey)

			headers, err := credentials.requestHeaders()
			require.NoError(t, err)
			assert.Equal(t, tc.want, headers)
		})
	}
}

func TestCredentialsRejectInvalidConfiguration(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		opts []CredentialOption
	}{
		{name: "missing partition"},
		{name: "empty partition", opts: []CredentialOption{WithPartitionKey("")}},
		{name: "nil option", opts: []CredentialOption{WithPartitionKey("user"), nil}},
		{name: "empty bearer", opts: []CredentialOption{WithPartitionKey("user"), WithBearerToken("")}},
		{name: "empty API key", opts: []CredentialOption{WithPartitionKey("user"), WithAPIKey("X-Key", "")}},
		{name: "invalid API header", opts: []CredentialOption{WithPartitionKey("user"), WithAPIKey("Bad Header", "key")}},
		{name: "empty basic", opts: []CredentialOption{WithPartitionKey("user"), WithBasicAuth("", "")}},
		{name: "colon basic user", opts: []CredentialOption{WithPartitionKey("user"), WithBasicAuth("user:other", "password")}},
		{name: "nil token source", opts: []CredentialOption{WithPartitionKey("user"), WithTokenSource(nil)}},
		{name: "CRLF header", opts: []CredentialOption{WithPartitionKey("user"), WithBearerToken("token\r\nInjected: secret")}},
		{name: "NUL header", opts: []CredentialOption{WithPartitionKey("user"), WithHeaders(http.Header{"X-Key": {"secret\x00"}})}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := NewCredentials(tc.opts...)
			require.Error(t, err)
			assert.NotContains(t, err.Error(), "secret")
			assert.NotContains(t, err.Error(), "password")
		})
	}
}

func TestCredentialsCloneHeaders(t *testing.T) {
	t.Parallel()

	input := http.Header{"X-Key": {"original"}}
	option := WithHeaders(input)
	credentials, err := NewCredentials(WithPartitionKey("user"), option)
	require.NoError(t, err)

	input["X-Key"][0] = "changed input"

	first, err := credentials.requestHeaders()
	require.NoError(t, err)
	assert.Equal(t, "original", first.Get("X-Key"))
	first["X-Key"][0] = "changed output"

	second, err := credentials.requestHeaders()
	require.NoError(t, err)
	assert.Equal(t, "original", second.Get("X-Key"))
}

type credentialTokenSourceFunc func() (*oauth2.Token, error)

func (f credentialTokenSourceFunc) Token() (*oauth2.Token, error) { return f() }

func TestCredentialsResolveTokenSourceForEachRequest(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	source := credentialTokenSourceFunc(func() (*oauth2.Token, error) {
		if calls.Add(1) == 1 {
			return &oauth2.Token{AccessToken: "first"}, nil
		}

		return &oauth2.Token{AccessToken: "second"}, nil
	})
	credentials, err := NewCredentials(WithPartitionKey("stable-user"), WithTokenSource(source))
	require.NoError(t, err)
	assert.Zero(t, calls.Load(), "construction must not fetch tokens")

	first, err := credentials.requestHeaders()
	require.NoError(t, err)
	assert.Equal(t, "Bearer first", first.Get("Authorization"))

	second, err := credentials.requestHeaders()
	require.NoError(t, err)
	assert.Equal(t, "Bearer second", second.Get("Authorization"))
	assert.Equal(t, int32(2), calls.Load())
	assert.Equal(t, "stable-user", credentials.PartitionKey)
}

func TestCredentialsSupportOAuthReuseAndRefresh(t *testing.T) {
	t.Parallel()

	var refreshes atomic.Int32
	source := oauth2.ReuseTokenSource(&oauth2.Token{AccessToken: "expired", Expiry: time.Now().Add(-time.Hour)}, credentialTokenSourceFunc(func() (*oauth2.Token, error) {
		refreshes.Add(1)

		return &oauth2.Token{AccessToken: "refreshed", Expiry: time.Now().Add(time.Hour)}, nil
	}))
	credentials, err := NewCredentials(WithPartitionKey("user"), WithTokenSource(source))
	require.NoError(t, err)

	for range 2 {
		headers, err := credentials.requestHeaders()
		require.NoError(t, err)
		assert.Equal(t, "Bearer refreshed", headers.Get("Authorization"))
	}

	assert.Equal(t, int32(1), refreshes.Load())
}

func TestCredentialsTokenErrorsPreserveClassificationWithoutSecrets(t *testing.T) {
	t.Parallel()

	failure := errors.New("token refresh response contains secret")
	credentials, err := NewCredentials(WithPartitionKey("user"), WithTokenSource(credentialTokenSourceFunc(func() (*oauth2.Token, error) {
		return nil, failure
	})))
	require.NoError(t, err)

	_, err = credentials.requestHeaders()
	require.ErrorIs(t, err, failure)
	assert.NotContains(t, err.Error(), "secret")
}

func TestCredentialsRejectInvalidOAuthTokens(t *testing.T) {
	t.Parallel()

	for _, token := range []*oauth2.Token{nil, {}, {AccessToken: "secret\r\nInjected: bad"}, {AccessToken: "secret", TokenType: "Bad\nType"}} {
		t.Run("token", func(t *testing.T) {
			t.Parallel()

			credentials, err := NewCredentials(WithPartitionKey("user"), WithTokenSource(oauth2.StaticTokenSource(token)))
			require.NoError(t, err)
			_, err = credentials.requestHeaders()
			require.Error(t, err)
			assert.NotContains(t, err.Error(), "secret")
		})
	}
}

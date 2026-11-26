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

package workload

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testAudience = "https://tools.example.com"

// newTestClient returns an enabled issuer for f and a client for testAudience.
func newTestClient(t *testing.T, f *fakeIssuer) (*Issuer, *http.Client) {
	t.Helper()

	issuer, err := NewIssuer(f.tokenURL(), writeAssertion(t, t.TempDir(), f.assertion), WithExperimentalDPoP())
	require.NoError(t, err)

	issuer.now = f.now

	transport, err := issuer.Transport(testAudience, nil)
	require.NoError(t, err)

	return issuer, &http.Client{Transport: transport}
}

// get sends a GET and returns the response status and headers.
func get(t *testing.T, client *http.Client, url string) (int, http.Header) {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, url, nil)
	require.NoError(t, err)

	resp, err := client.Do(req)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())

	return resp.StatusCode, resp.Header
}

// getStatus sends a GET and returns the response status.
func getStatus(t *testing.T, client *http.Client, url string) int {
	t.Helper()

	status, _ := get(t, client, url)

	return status
}

// getErr sends a GET that the transport is expected to fail.
func getErr(t *testing.T, client *http.Client, url string) error {
	t.Helper()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, url, nil)
	require.NoError(t, err)

	resp, err := client.Do(req)
	if resp != nil {
		require.NoError(t, resp.Body.Close())
	}

	require.Error(t, err)

	return err
}

func TestNewIssuerIsOffByDefault(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []Option
	}{
		{name: "no options"},
		{name: "other options only", opts: []Option{WithHTTPClient(&http.Client{})}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var hits atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
			t.Cleanup(server.Close)

			// The assertion file does not exist: a disabled issuer must not read it.
			issuer, err := NewIssuer(server.URL+"/token", "/nonexistent/token", tt.opts...)
			require.ErrorIs(t, err, ErrExperimentalDisabled)
			assert.Nil(t, issuer)
			assert.Zero(t, hits.Load())
		})
	}
}

func TestClientWithoutTransportSendsNoDPoP(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)

	// The SDK's default client path: nothing adds Authorization or DPoP, and no
	// token request is made, so the service turns the request away.
	status, header := get(t, http.DefaultClient, f.server.URL+"/svc/tools")
	assert.Equal(t, http.StatusUnauthorized, status)
	assert.Equal(t, `DPoP algs="ES256"`, header.Get("WWW-Authenticate"))

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Zero(t, f.tokenRequests)
}

func TestTokenEndpointRejectsMissingDPoP(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)

	form := "grant_type=client_credentials&client_assertion_type=urn%3Aietf%3Aparams%3Aoauth%3Aclient-assertion-type%3Ajwt-bearer&client_assertion=sa-jwt-1&audience=x"
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, f.tokenURL(), strings.NewReader(form))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())

	// The fake enforces the contract: no proof, no token.
	assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
	assert.NotEmpty(t, resp.Header.Get("DPoP-Nonce"))
}

func TestTransportRetriesOnceForNonceAndKeepsIt(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	issuer, client := newTestClient(t, f)

	status, header := get(t, client, f.server.URL+"/svc/tools")
	require.Equal(t, http.StatusNoContent, status)
	assert.Equal(t, testAudience, header.Get("X-Audience"))

	f.mu.Lock()
	assert.Equal(t, 2, f.tokenRequests, "first request has no nonce, the retry carries it")
	f.mu.Unlock()
	assert.Equal(t, "nonce-0", issuer.lastNonce())

	// A second audience on the same endpoint reuses the kept nonce.
	other, err := issuer.Transport("https://other.example.com", nil)
	require.NoError(t, err)

	status, header = get(t, &http.Client{Transport: other}, f.server.URL+"/svc/other")
	require.Equal(t, http.StatusNoContent, status)
	assert.Equal(t, "https://other.example.com", header.Get("X-Audience"))

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Equal(t, 3, f.tokenRequests)
	assert.Equal(t, []string{testAudience, testAudience, "https://other.example.com"}, f.audiences)
}

func TestTransportRetriesNonceOnlyOnce(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	f.alwaysNewNonce = true
	_, client := newTestClient(t, f)

	err := getErr(t, client, f.server.URL+"/svc/tools")

	var tokenErr *TokenError
	require.ErrorAs(t, err, &tokenErr)
	assert.Equal(t, http.StatusBadRequest, tokenErr.StatusCode)
	assert.Equal(t, "use_dpop_nonce", tokenErr.Code)

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Equal(t, 2, f.tokenRequests)
	assert.Zero(t, f.resourceCalls)
}

func TestTransportCachesTokenPerAudienceUntilExpiry(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)

	var mu sync.Mutex
	now := time.Now()
	f.now = func() time.Time {
		mu.Lock()
		defer mu.Unlock()

		return now
	}
	advance := func(d time.Duration) {
		mu.Lock()
		defer mu.Unlock()

		now = now.Add(d)
	}

	issuer, client := newTestClient(t, f)
	tokenRequests := func() int {
		f.mu.Lock()
		defer f.mu.Unlock()

		return f.tokenRequests
	}

	for range 3 {
		require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/tools"))
	}

	assert.Equal(t, 2, tokenRequests(), "one nonce retry, then the cached token")

	// Another audience gets its own token.
	other, err := issuer.Transport("https://other.example.com", nil)
	require.NoError(t, err)
	require.Equal(t, http.StatusNoContent, getStatus(t, &http.Client{Transport: other}, f.server.URL+"/svc/other"))
	assert.Equal(t, 3, tokenRequests())

	// Still cached 30s before the 300s expiry is reached.
	advance(269 * time.Second)
	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/tools"))
	assert.Equal(t, 3, tokenRequests())

	// Replaced once within the refresh margin.
	advance(2 * time.Second)
	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/tools"))
	assert.Equal(t, 4, tokenRequests())
}

func TestTransportFetchesOneTokenForConcurrentRequests(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	_, client := newTestClient(t, f)

	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			// require must not run off the test goroutine, so check errors by hand.
			req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, f.server.URL+"/svc/tools", nil)
			if err != nil {
				t.Error(err)

				return
			}

			resp, err := client.Do(req)
			if err != nil {
				t.Error(err)

				return
			}

			assert.Equal(t, http.StatusNoContent, resp.StatusCode)
			assert.NoError(t, resp.Body.Close())
		})
	}

	wg.Wait()

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Equal(t, 2, f.tokenRequests)
	assert.Equal(t, 16, f.resourceCalls)
}

func TestTransportRereadsAssertionFile(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	f.expiresIn = 1 // refresh margin 0.5s, so each call after 1s fetches again

	dir := t.TempDir()
	path := writeAssertion(t, dir, "sa-jwt-1")

	issuer, err := NewIssuer(f.tokenURL(), path, WithExperimentalDPoP())
	require.NoError(t, err)

	var mu sync.Mutex
	now := time.Now()
	clock := func() time.Time {
		mu.Lock()
		defer mu.Unlock()

		return now
	}
	issuer.now = clock
	f.now = clock

	transport, err := issuer.Transport(testAudience, nil)
	require.NoError(t, err)

	client := &http.Client{Transport: transport}

	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/tools"))

	// Kubernetes rotates the projected token in place.
	f.mu.Lock()
	f.assertion = "sa-jwt-2"
	f.mu.Unlock()
	writeAssertion(t, dir, "sa-jwt-2")

	mu.Lock()
	now = now.Add(time.Second)
	mu.Unlock()

	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/tools"))

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Equal(t, []string{"sa-jwt-1", "sa-jwt-1", "sa-jwt-2"}, f.assertions)
}

func TestTransportBindsProofToRequest(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	_, client := newTestClient(t, f)

	// The fake checks htm, htu without query and fragment, ath and that the
	// proof key matches the token's cnf.jkt.
	req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, f.server.URL+"/svc/mcp?session=1#frag", strings.NewReader(`{}`))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())
	assert.Equal(t, http.StatusNoContent, resp.StatusCode)

	// The caller's request is not modified.
	assert.Empty(t, req.Header.Get("Authorization"))
	assert.Empty(t, req.Header.Get("DPoP"))
}

func TestProofClaims(t *testing.T) {
	t.Parallel()

	key, err := processKey()
	require.NoError(t, err)
	signer, err := newProofSigner(key)
	require.NoError(t, err)

	f := &fakeIssuer{now: time.Now, jtis: make(map[string]bool)}

	tests := []struct {
		name        string
		url         string
		htu         string
		nonce       string
		accessToken string
	}{
		{name: "token request with nonce", url: "https://issuer.example.com/token", htu: "https://issuer.example.com/token", nonce: "n-1"},
		{name: "resource call", url: "https://user@tools.example.com/mcp?x=1#y", htu: "https://tools.example.com/mcp", accessToken: "at-1"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			req, err := http.NewRequestWithContext(t.Context(), http.MethodPost, tt.url, nil)
			require.NoError(t, err)

			proof, err := signer.proof(http.MethodPost, req.URL, time.Now(), tt.nonce, tt.accessToken)
			require.NoError(t, err)

			f.mu.Lock()
			defer f.mu.Unlock()

			claims, jkt, err := f.verifyProof([]string{proof}, http.MethodPost, tt.htu)
			require.NoError(t, err)
			assert.NotEmpty(t, jkt)
			assert.Equal(t, tt.nonce, claims.Nonce)

			if tt.accessToken == "" {
				assert.Empty(t, claims.ATH)
			} else {
				assert.Equal(t, accessTokenHash(tt.accessToken), claims.ATH)
			}
		})
	}
}

func TestTransportSurfacesTokenErrors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		setup func(*fakeIssuer)
		want  *TokenError
	}{
		{
			name:  "invalid client",
			setup: func(f *fakeIssuer) { f.assertion = "another-token" },
			want:  &TokenError{StatusCode: http.StatusUnauthorized, Code: "invalid_client", Description: "unknown service account token"},
		},
		{
			name: "invalid target",
			setup: func(f *fakeIssuer) {
				f.failWith = &TokenError{StatusCode: http.StatusBadRequest, Code: "invalid_target", Description: "no such service"}
			},
			want: &TokenError{StatusCode: http.StatusBadRequest, Code: "invalid_target", Description: "no such service"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			f := newFakeIssuer(t)
			issuer, err := NewIssuer(f.tokenURL(), writeAssertion(t, t.TempDir(), "sa-jwt-1"), WithExperimentalDPoP())
			require.NoError(t, err)
			tt.setup(f)

			transport, err := issuer.Transport(testAudience, nil)
			require.NoError(t, err)

			err = getErr(t, &http.Client{Transport: transport}, f.server.URL+"/svc/tools")

			var tokenErr *TokenError
			require.ErrorAs(t, err, &tokenErr)
			assert.Equal(t, tt.want, tokenErr)
			assert.Contains(t, err.Error(), tt.want.Code)
		})
	}
}

func TestTransportRejectsUnusableTokenResponses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		setup func(*fakeIssuer)
		want  string
	}{
		{name: "bearer token", setup: func(f *fakeIssuer) { f.tokenType = "Bearer" }, want: "want DPoP"},
		{name: "no expiry", setup: func(f *fakeIssuer) { f.expiresIn = 0 }, want: "expires_in"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			f := newFakeIssuer(t)
			tt.setup(f)
			_, client := newTestClient(t, f)

			err := getErr(t, client, f.server.URL+"/svc/tools")
			assert.Contains(t, err.Error(), tt.want)

			f.mu.Lock()
			defer f.mu.Unlock()

			assert.Zero(t, f.resourceCalls)
		})
	}
}

func TestTransportRequiresAudience(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	issuer, err := NewIssuer(f.tokenURL(), writeAssertion(t, t.TempDir(), "sa-jwt-1"), WithExperimentalDPoP())
	require.NoError(t, err)

	transport, err := issuer.Transport("", nil)
	require.Error(t, err)
	assert.Nil(t, transport)
}

func TestTransportRefusesExistingCredentials(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	_, client := newTestClient(t, f)

	for _, header := range []string{"Authorization", "DPoP"} {
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, f.server.URL+"/svc/tools", nil)
		require.NoError(t, err)
		req.Header.Set(header, "Bearer other")

		resp, err := client.Do(req)
		if resp != nil {
			require.NoError(t, resp.Body.Close())
		}

		require.ErrorContains(t, err, "already has an Authorization or DPoP header")
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Zero(t, f.tokenRequests)
	assert.Zero(t, f.resourceCalls)
}

func TestTransportRefusesCrossOriginRedirect(t *testing.T) {
	t.Parallel()

	var elsewhere atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { elsewhere.Add(1) }))
	t.Cleanup(other.Close)

	f := newFakeIssuer(t)
	_, client := newTestClient(t, f)

	redirect := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, other.URL+"/steal", http.StatusFound)
	}))
	t.Cleanup(redirect.Close)

	err := getErr(t, client, redirect.URL+"/svc/tools")
	require.ErrorContains(t, err, "cross-origin redirect refused")
	assert.Zero(t, elsewhere.Load())
}

func TestTransportStopsWaitingWhenContextEnds(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(release) })

	issuer, err := NewIssuer(server.URL+"/token", writeAssertion(t, t.TempDir(), "sa-jwt-1"), WithExperimentalDPoP())
	require.NoError(t, err)

	// The first caller holds the audience's lock while its token request hangs.
	go func() { _, _ = issuer.token(context.Background(), testAudience) }()

	require.Eventually(t, func() bool {
		entry := issuer.entry(testAudience)

		return len(entry.lock) == 1
	}, 5*time.Second, 10*time.Millisecond)

	ctx, cancel := context.WithTimeout(t.Context(), 50*time.Millisecond)
	defer cancel()

	_, err = issuer.token(ctx, testAudience)
	require.ErrorIs(t, err, context.DeadlineExceeded)
}

func TestNewIssuerValidatesConfig(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		endpoint string
		file     string
		opts     []Option
		want     string
	}{
		{name: "plain http remote", endpoint: "http://issuer.example.com/token", file: "f", want: "must use https"},
		{name: "relative", endpoint: "/token", file: "f", want: "must be absolute"},
		{name: "query", endpoint: "https://issuer.example.com/token?a=b", file: "f", want: "query or fragment"},
		{name: "fragment", endpoint: "https://issuer.example.com/token#a", file: "f", want: "query or fragment"},
		{name: "user info", endpoint: "https://u@issuer.example.com/token", file: "f", want: "user information"},
		{name: "no file", endpoint: "https://issuer.example.com/token", want: "assertion file"},
		{name: "nil option", endpoint: "https://issuer.example.com/token", file: "f", opts: []Option{nil}, want: "option 1 is nil"},
		{name: "nil client", endpoint: "https://issuer.example.com/token", file: "f", opts: []Option{WithHTTPClient(nil)}, want: "must not be nil"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			issuer, err := NewIssuer(tt.endpoint, tt.file, append([]Option{WithExperimentalDPoP()}, tt.opts...)...)
			require.ErrorContains(t, err, tt.want)
			require.NotErrorIs(t, err, ErrExperimentalDisabled)
			assert.Nil(t, issuer)
		})
	}

	for _, ok := range []string{"https://issuer.example.com/token", "http://127.0.0.1:1/token", "http://[::1]:1/token", "http://localhost/token"} {
		_, err := NewIssuer(ok, "f", WithExperimentalDPoP())
		require.NoError(t, err, ok)
	}
}

func TestMissingAssertionFileFailsTokenRequest(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	issuer, err := NewIssuer(f.tokenURL(), "/nonexistent/token", WithExperimentalDPoP())
	require.NoError(t, err)

	_, err = issuer.token(t.Context(), testAudience)
	require.ErrorIs(t, err, os.ErrNotExist)

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Zero(t, f.tokenRequests)
}

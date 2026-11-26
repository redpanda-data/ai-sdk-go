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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const (
	nextHop = "https://issuer.example.com/agents/agt_next"
	subject = "subject-1"
)

// newOBOClient returns a client of an enabled issuer for f that calls
// nextHop for the user of subject with the scopes.
func newOBOClient(t *testing.T, f *fakeIssuer, scopes []string) *http.Client {
	t.Helper()

	issuer, err := NewIssuer(f.tokenURL(), writeAssertion(t, t.TempDir(), f.assertion), WithExperimentalDPoP())
	require.NoError(t, err)

	issuer.now = f.now

	transport, err := issuer.OnBehalfOf(subject, nextHop, scopes, nil)
	require.NoError(t, err)

	return &http.Client{Transport: transport}
}

func TestOnBehalfOfExchangesTheSubjectToken(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	client := newOBOClient(t, f, []string{"read", "write"})

	status, header := get(t, client, f.server.URL+"/svc/a")
	require.Equal(t, http.StatusNoContent, status)
	assert.Equal(t, nextHop, header.Get("X-Audience"))
	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/b"))

	f.mu.Lock()
	defer f.mu.Unlock()

	// One nonce retry, then one exchange for both calls.
	assert.Equal(t, []string{subject, subject}, f.subjects)
	assert.Equal(t, [][]string{{"read write"}, {"read write"}}, f.scopes)
	assert.Equal(t, 1, f.issued)
	assert.Equal(t, 2, f.resourceCalls)
}

func TestOnBehalfOfSendsNoScopeWhenNoneIsGiven(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	client := newOBOClient(t, f, nil)
	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/a"))

	f.mu.Lock()
	defer f.mu.Unlock()

	require.NotEmpty(t, f.scopes)

	for _, scope := range f.scopes {
		assert.Nil(t, scope)
	}
}

func TestOnBehalfOfExchangesAgainAfterExpiry(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	now := time.Now()
	f.now = func() time.Time { return now }
	client := newOBOClient(t, f, []string{"read"})

	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/a"))

	now = now.Add(5 * time.Minute)

	require.Equal(t, http.StatusNoContent, getStatus(t, client, f.server.URL+"/svc/a"))

	f.mu.Lock()
	defer f.mu.Unlock()

	assert.Equal(t, 2, f.issued)
}

func TestOnBehalfOfSurfacesRefusals(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	f.failWith = &TokenError{StatusCode: http.StatusBadRequest, Code: "invalid_grant", Description: "the user gave no consent to the agent"}
	client := newOBOClient(t, f, []string{"read"})

	err := getErr(t, client, f.server.URL+"/svc/a")

	var tokenErr *TokenError
	require.ErrorAs(t, err, &tokenErr)
	assert.Equal(t, "invalid_grant", tokenErr.Code)
}

func TestOnBehalfOfValidatesArguments(t *testing.T) {
	t.Parallel()

	f := newFakeIssuer(t)
	issuer, err := NewIssuer(f.tokenURL(), writeAssertion(t, t.TempDir(), f.assertion), WithExperimentalDPoP())
	require.NoError(t, err)

	_, err = issuer.OnBehalfOf("", nextHop, nil, nil)
	require.Error(t, err)

	_, err = issuer.OnBehalfOf("bad token", nextHop, nil, nil)
	require.Error(t, err)

	_, err = issuer.OnBehalfOf(subject, "", nil, nil)
	require.Error(t, err)
}

func TestSubjectToken(t *testing.T) {
	t.Parallel()

	for header, want := range map[string]string{
		"Bearer tok-1":  "tok-1",
		"DPoP tok-2":    "tok-2",
		"bearer  tok-3": "tok-3",
		"Basic abc":     "",
		"Bearer":        "",
		"":              "",
	} {
		req := httptest.NewRequestWithContext(t.Context(), http.MethodGet, "/", nil)
		if header != "" {
			req.Header.Set("Authorization", header)
		}

		got, err := SubjectToken(req)
		if want == "" {
			require.Error(t, err, header)

			continue
		}

		require.NoError(t, err, header)
		assert.Equal(t, want, got, header)
	}
}

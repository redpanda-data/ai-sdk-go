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
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// fakeIssuer is a token endpoint and a downstream service that enforce the
// workspace issuer's contract. Handlers record violations as errors so tests
// fail on them; they never call require off the test goroutine.
type fakeIssuer struct {
	t      *testing.T
	server *httptest.Server
	now    func() time.Time

	mu sync.Mutex
	// nonce is the nonce the endpoint currently accepts.
	nonce string
	// alwaysNewNonce makes every request look stale, to test the retry bound.
	alwaysNewNonce bool
	// assertion is the client assertion the endpoint accepts.
	assertion string
	// tokenType and expiresIn shape the success response.
	tokenType string
	expiresIn int64
	// failWith makes the endpoint answer every token request with an error.
	failWith *TokenError

	tokenRequests int
	audiences     []string
	assertions    []string
	resourceCalls int
	// bound maps each issued token to its key thumbprint (cnf.jkt).
	bound   map[string]string
	audOf   map[string]string
	jtis    map[string]bool
	issued  int
	failure []string
}

func newFakeIssuer(t *testing.T) *fakeIssuer {
	t.Helper()

	f := &fakeIssuer{
		t:         t,
		now:       time.Now,
		nonce:     "nonce-0",
		assertion: "sa-jwt-1",
		tokenType: "DPoP",
		expiresIn: 300,
		bound:     make(map[string]string),
		audOf:     make(map[string]string),
		jtis:      make(map[string]bool),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/token", f.serveToken)
	mux.HandleFunc("/svc/", f.serveResource)
	f.server = httptest.NewServer(mux)
	t.Cleanup(f.server.Close)
	t.Cleanup(func() {
		f.mu.Lock()
		defer f.mu.Unlock()

		for _, msg := range f.failure {
			t.Error(msg)
		}
	})

	return f
}

func (f *fakeIssuer) tokenURL() string { return f.server.URL + "/token" }

func (f *fakeIssuer) failf(format string, args ...any) {
	f.failure = append(f.failure, fmt.Sprintf(format, args...))
}

func (f *fakeIssuer) writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	if err := json.NewEncoder(w).Encode(v); err != nil {
		f.failf("encode response: %v", err)
	}
}

func (f *fakeIssuer) writeOAuthError(w http.ResponseWriter, status int, code, description string) {
	f.writeJSON(w, status, errorResponse{Error: code, ErrorDescription: description})
}

func (f *fakeIssuer) serveToken(w http.ResponseWriter, req *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.tokenRequests++
	// The endpoint always sends its current nonce.
	if f.alwaysNewNonce {
		f.nonce = "nonce-" + strconv.Itoa(f.tokenRequests)
	}

	w.Header().Set("DPoP-Nonce", f.nonce)

	if req.Method != http.MethodPost {
		f.failf("token request method %s", req.Method)
	}

	if got := req.Header.Get("Content-Type"); got != "application/x-www-form-urlencoded" {
		f.failf("token request Content-Type %q", got)
	}

	req.Body = http.MaxBytesReader(w, req.Body, 1<<16)
	if err := req.ParseForm(); err != nil {
		f.failf("parse token form: %v", err)
	}

	if len(req.URL.Query()) != 0 {
		f.failf("token request has a query: %s", req.URL.RawQuery)
	}

	form := req.PostForm

	want := map[string]string{
		"grant_type":            "client_credentials",
		"client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
	}
	for name, value := range want {
		if got := form[name]; len(got) != 1 || got[0] != value {
			f.failf("token form %s = %q, want %q", name, got, value)
		}
	}

	if len(form["audience"]) != 1 || form.Get("audience") == "" {
		f.writeOAuthError(w, http.StatusBadRequest, "invalid_target", "exactly one audience is required")

		return
	}

	f.audiences = append(f.audiences, form.Get("audience"))
	f.assertions = append(f.assertions, form.Get("client_assertion"))

	if len(form["client_assertion"]) != 1 || form.Get("client_assertion") != f.assertion {
		f.writeOAuthError(w, http.StatusUnauthorized, "invalid_client", "unknown service account token")

		return
	}

	claims, jkt, err := f.verifyProof(req.Header.Values("DPoP"), http.MethodPost, f.tokenURL())
	if err != nil {
		f.writeOAuthError(w, http.StatusBadRequest, "invalid_dpop_proof", err.Error())

		return
	}

	if claims.ATH != "" {
		f.failf("token request proof has ath")
	}

	if f.failWith != nil {
		f.writeOAuthError(w, f.failWith.StatusCode, f.failWith.Code, f.failWith.Description)

		return
	}

	if f.alwaysNewNonce || claims.Nonce != f.nonce {
		f.writeOAuthError(w, http.StatusBadRequest, "use_dpop_nonce", "use the DPoP-Nonce header")

		return
	}

	f.issued++
	accessToken := "at-" + strconv.Itoa(f.issued)
	f.bound[accessToken] = jkt
	f.audOf[accessToken] = form.Get("audience")

	f.writeJSON(w, http.StatusOK, tokenResponse{AccessToken: accessToken, TokenType: f.tokenType, ExpiresIn: f.expiresIn})
}

func (f *fakeIssuer) serveResource(w http.ResponseWriter, req *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.resourceCalls++

	authorization := req.Header.Get("Authorization")

	accessToken, ok := strings.CutPrefix(authorization, "DPoP ")
	if !ok {
		w.Header().Set("WWW-Authenticate", `DPoP algs="ES256"`)
		http.Error(w, "DPoP token required", http.StatusUnauthorized)

		return
	}

	jkt, ok := f.bound[accessToken]
	if !ok {
		http.Error(w, "unknown token", http.StatusUnauthorized)

		return
	}

	htu := "http://" + req.Host + req.URL.EscapedPath()

	claims, proofJKT, err := f.verifyProof(req.Header.Values("DPoP"), req.Method, htu)
	if err != nil {
		http.Error(w, "invalid DPoP proof: "+err.Error(), http.StatusUnauthorized)

		return
	}

	if proofJKT != jkt {
		http.Error(w, "proof key does not match the token's cnf.jkt", http.StatusUnauthorized)

		return
	}

	if claims.ATH != accessTokenHash(accessToken) {
		http.Error(w, "ath does not match the access token", http.StatusUnauthorized)

		return
	}

	w.Header().Set("X-Audience", f.audOf[accessToken])
	w.WriteHeader(http.StatusNoContent)
}

// verifyProof checks one DPoP proof against RFC 9449 section 4.3 and returns
// its claims and its key's RFC 7638 thumbprint.
func (f *fakeIssuer) verifyProof(values []string, method, htu string) (proofClaims, string, error) {
	if len(values) != 1 {
		return proofClaims{}, "", fmt.Errorf("want one DPoP header, got %d", len(values))
	}

	parts := strings.Split(values[0], ".")
	if len(parts) != 3 {
		return proofClaims{}, "", errors.New("proof is not a compact JWS")
	}

	var header struct {
		Typ string    `json:"typ"`
		Alg string    `json:"alg"`
		JWK publicJWK `json:"jwk"`
	}
	if err := decodeSegment(parts[0], &header); err != nil {
		return proofClaims{}, "", err
	}

	if header.Typ != "dpop+jwt" || header.Alg != "ES256" || header.JWK.Kty != "EC" || header.JWK.Crv != "P-256" {
		return proofClaims{}, "", fmt.Errorf("bad proof header %+v", header)
	}

	x, errX := base64.RawURLEncoding.DecodeString(header.JWK.X)

	y, errY := base64.RawURLEncoding.DecodeString(header.JWK.Y)
	if errX != nil || errY != nil || len(x) != 32 || len(y) != 32 {
		return proofClaims{}, "", errors.New("bad proof JWK coordinates")
	}

	key, err := ecdsa.ParseUncompressedPublicKey(elliptic.P256(), append(append([]byte{4}, x...), y...))
	if err != nil {
		return proofClaims{}, "", fmt.Errorf("proof JWK: %w", err)
	}

	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(signature) != 64 {
		return proofClaims{}, "", errors.New("bad proof signature encoding")
	}

	digest := sha256.Sum256([]byte(parts[0] + "." + parts[1]))
	if !ecdsa.Verify(key, digest[:], new(big.Int).SetBytes(signature[:32]), new(big.Int).SetBytes(signature[32:])) {
		return proofClaims{}, "", errors.New("proof signature does not verify")
	}

	var claims proofClaims
	if err := decodeSegment(parts[1], &claims); err != nil {
		return proofClaims{}, "", err
	}

	switch {
	case claims.JTI == "":
		return proofClaims{}, "", errors.New("proof has no jti")
	case f.jtis[claims.JTI]:
		return proofClaims{}, "", errors.New("proof jti replayed")
	case claims.HTM != method:
		return proofClaims{}, "", fmt.Errorf("htm %q, want %q", claims.HTM, method)
	case claims.HTU != htu:
		return proofClaims{}, "", fmt.Errorf("htu %q, want %q", claims.HTU, htu)
	case f.now().Sub(time.Unix(claims.IAT, 0)).Abs() > time.Minute:
		return proofClaims{}, "", fmt.Errorf("iat %d is not current", claims.IAT)
	}

	f.jtis[claims.JTI] = true

	thumbprint, err := json.Marshal(header.JWK) // members are already in RFC 7638 order
	if err != nil {
		return proofClaims{}, "", err
	}

	sum := sha256.Sum256(thumbprint)

	return claims, base64.RawURLEncoding.EncodeToString(sum[:]), nil
}

func decodeSegment(segment string, v any) error {
	data, err := base64.RawURLEncoding.DecodeString(segment)
	if err != nil {
		return fmt.Errorf("decode JWS segment: %w", err)
	}

	return json.Unmarshal(data, v)
}

// writeAssertion writes a service account token file and returns its path.
func writeAssertion(t *testing.T, dir, assertion string) string {
	t.Helper()

	path := filepath.Join(dir, "token")
	require.NoError(t, os.WriteFile(path, []byte(assertion+"\n"), 0o600))

	return path
}

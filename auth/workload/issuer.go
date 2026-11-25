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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// ErrExperimentalDisabled is returned by NewIssuer unless the caller passes
// WithExperimentalDPoP.
var ErrExperimentalDisabled = errors.New("workload: DPoP workload tokens are experimental and unreleased; pass WithExperimentalDPoP to enable them")

const (
	grantType           = "client_credentials"
	clientAssertionType = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

	// maxTokenResponse bounds the token endpoint's response body.
	maxTokenResponse = 1 << 20
	// maxRefreshMargin is how long before expiry a cached token is replaced.
	maxRefreshMargin = 30 * time.Second
)

// Option configures an Issuer.
type Option func(*config) error

type config struct {
	experimental bool
	httpClient   *http.Client
}

// WithExperimentalDPoP enables the issuer. The package is unreleased; without
// this option NewIssuer returns ErrExperimentalDisabled and does nothing.
func WithExperimentalDPoP() Option {
	return func(c *config) error {
		c.experimental = true

		return nil
	}
}

// WithHTTPClient sets the client for token requests, for example to configure
// TLS or timeouts. The issuer copies it and refuses redirects from the token
// endpoint, so the service account token is never sent elsewhere.
func WithHTTPClient(client *http.Client) Option {
	return func(c *config) error {
		if client == nil {
			return errors.New("workload: HTTP client must not be nil")
		}

		c.httpClient = client

		return nil
	}
}

// Issuer gets DPoP-bound access tokens from one token endpoint. It keeps the
// endpoint's last DPoP nonce and caches one token per audience until shortly
// before it expires. An Issuer is safe for concurrent use.
type Issuer struct {
	tokenURL      *url.URL
	assertionFile string
	client        *http.Client
	signer        *proofSigner
	now           func() time.Time

	nonceMu sync.Mutex
	nonce   string

	tokensMu sync.Mutex
	tokens   map[string]*cachedToken
}

type cachedToken struct {
	// lock is held while a token is read or fetched; a channel so that a
	// waiter can give up when its context ends.
	lock        chan struct{}
	accessToken string
	refreshAt   time.Time
}

// NewIssuer returns an issuer for the token endpoint at tokenURL that
// authenticates with the projected service account token in assertionFile.
// The file is read again on every token request. tokenURL must use https,
// except on a loopback host, and must not have a query or fragment.
//
// NewIssuer returns ErrExperimentalDisabled unless opts include
// WithExperimentalDPoP. It makes no network requests.
func NewIssuer(tokenURL, assertionFile string, opts ...Option) (*Issuer, error) {
	var cfg config

	for i, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("workload: option %d is nil", i)
		}

		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	if !cfg.experimental {
		return nil, ErrExperimentalDisabled
	}

	endpoint, err := parseTokenURL(tokenURL)
	if err != nil {
		return nil, err
	}

	if assertionFile == "" {
		return nil, errors.New("workload: assertion file path must not be empty")
	}

	key, err := processKey()
	if err != nil {
		return nil, fmt.Errorf("workload: generate DPoP key: %w", err)
	}

	signer, err := newProofSigner(key)
	if err != nil {
		return nil, err
	}

	client := &http.Client{}
	if cfg.httpClient != nil {
		*client = *cfg.httpClient
	}

	client.CheckRedirect = func(*http.Request, []*http.Request) error {
		return errors.New("workload: token endpoint redirect refused")
	}

	return &Issuer{
		tokenURL:      endpoint,
		assertionFile: assertionFile,
		client:        client,
		signer:        signer,
		now:           time.Now,
		tokens:        make(map[string]*cachedToken),
	}, nil
}

func parseTokenURL(raw string) (*url.URL, error) {
	endpoint, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("workload: token URL: %w", err)
	}

	switch {
	case endpoint.Host == "":
		return nil, errors.New("workload: token URL must be absolute")
	case endpoint.User != nil:
		return nil, errors.New("workload: token URL must not have user information")
	case endpoint.RawQuery != "" || endpoint.ForceQuery || endpoint.Fragment != "":
		return nil, errors.New("workload: token URL must not have a query or fragment")
	case endpoint.Scheme == "https":
	case endpoint.Scheme == "http" && isLoopback(endpoint.Hostname()):
	default:
		return nil, errors.New("workload: token URL must use https")
	}

	return endpoint, nil
}

func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}

	ip := net.ParseIP(host)

	return ip != nil && ip.IsLoopback()
}

// TokenError is an OAuth error from the token endpoint (RFC 6749 section 5.2),
// such as invalid_client, invalid_target or invalid_dpop_proof.
type TokenError struct {
	// StatusCode is the HTTP status of the response.
	StatusCode int
	// Code is the OAuth error code; empty when the body was not an OAuth error.
	Code string
	// Description is the server's error_description, if any.
	Description string
}

func (e *TokenError) Error() string {
	var b strings.Builder

	fmt.Fprintf(&b, "workload: token endpoint returned HTTP %d", e.StatusCode)

	if e.Code != "" {
		b.WriteString(": " + e.Code)
	}

	if e.Description != "" {
		b.WriteString(": " + e.Description)
	}

	return b.String()
}

// token returns a cached or new access token for audience.
func (i *Issuer) token(ctx context.Context, audience string) (string, error) {
	entry := i.entry(audience)

	select {
	case entry.lock <- struct{}{}:
	case <-ctx.Done():
		return "", ctx.Err()
	}

	defer func() { <-entry.lock }()

	requested := i.now()
	if entry.accessToken != "" && requested.Before(entry.refreshAt) {
		return entry.accessToken, nil
	}

	accessToken, lifetime, err := i.request(ctx, audience)
	if err != nil {
		return "", err
	}

	entry.accessToken = accessToken
	entry.refreshAt = requested.Add(lifetime - min(maxRefreshMargin, lifetime/2))

	return accessToken, nil
}

func (i *Issuer) entry(audience string) *cachedToken {
	i.tokensMu.Lock()
	defer i.tokensMu.Unlock()

	entry, ok := i.tokens[audience]
	if !ok {
		entry = &cachedToken{lock: make(chan struct{}, 1)}
		i.tokens[audience] = entry
	}

	return entry
}

func (i *Issuer) lastNonce() string {
	i.nonceMu.Lock()
	defer i.nonceMu.Unlock()

	return i.nonce
}

func (i *Issuer) setNonce(nonce string) {
	i.nonceMu.Lock()
	defer i.nonceMu.Unlock()

	i.nonce = nonce
}

type tokenResponse struct {
	AccessToken string `json:"access_token"`
	TokenType   string `json:"token_type"`
	ExpiresIn   int64  `json:"expires_in"`
}

type errorResponse struct {
	Error            string `json:"error"`
	ErrorDescription string `json:"error_description"`
}

// request asks the token endpoint for a token for audience. It retries once
// when the endpoint asks for a new DPoP nonce.
func (i *Issuer) request(ctx context.Context, audience string) (string, time.Duration, error) {
	assertion, err := i.readAssertion()
	if err != nil {
		return "", 0, err
	}

	form := url.Values{
		"grant_type":            {grantType},
		"client_assertion_type": {clientAssertionType},
		"client_assertion":      {assertion},
		"audience":              {audience},
	}.Encode()

	for attempt := 0; ; attempt++ {
		status, nonce, body, err := i.post(ctx, form)
		if err != nil {
			return "", 0, err
		}

		if status == http.StatusOK {
			return parseTokenResponse(body)
		}

		tokenErr := &TokenError{StatusCode: status}

		var oauthErr errorResponse
		if json.Unmarshal(body, &oauthErr) == nil {
			tokenErr.Code = oauthErr.Error
			tokenErr.Description = oauthErr.ErrorDescription
		}

		if attempt == 0 && status == http.StatusBadRequest && tokenErr.Code == "use_dpop_nonce" && nonce != "" {
			continue
		}

		return "", 0, tokenErr
	}
}

// post sends one token request with a fresh proof. It returns the response
// status, the DPoP-Nonce it carried and its body.
func (i *Issuer) post(ctx context.Context, form string) (int, string, []byte, error) {
	proof, err := i.signer.proof(http.MethodPost, i.tokenURL, i.now(), i.lastNonce(), "")
	if err != nil {
		return 0, "", nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, i.tokenURL.String(), strings.NewReader(form))
	if err != nil {
		return 0, "", nil, fmt.Errorf("workload: token request: %w", err)
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("DPoP", proof)

	resp, err := i.client.Do(req)
	if err != nil {
		return 0, "", nil, fmt.Errorf("workload: token request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxTokenResponse))
	if err != nil {
		return 0, "", nil, fmt.Errorf("workload: read token response: %w", err)
	}

	nonce := resp.Header.Get("DPoP-Nonce")
	if nonce != "" {
		i.setNonce(nonce)
	}

	return resp.StatusCode, nonce, body, nil
}

func parseTokenResponse(body []byte) (string, time.Duration, error) {
	var token tokenResponse
	if err := json.Unmarshal(body, &token); err != nil {
		return "", 0, fmt.Errorf("workload: decode token response: %w", err)
	}

	switch {
	case token.AccessToken == "" || !validTokenValue(token.AccessToken):
		return "", 0, errors.New("workload: token response has no valid access_token")
	case !strings.EqualFold(token.TokenType, "DPoP"):
		return "", 0, fmt.Errorf("workload: token response has token_type %q, want DPoP", token.TokenType)
	case token.ExpiresIn <= 0:
		return "", 0, errors.New("workload: token response has no positive expires_in")
	}

	return token.AccessToken, time.Duration(token.ExpiresIn) * time.Second, nil
}

// validTokenValue reports whether s can go in an Authorization header: visible
// ASCII only.
func validTokenValue(s string) bool {
	for i := range len(s) {
		if s[i] < 0x21 || s[i] > 0x7e {
			return false
		}
	}

	return true
}

func (i *Issuer) readAssertion() (string, error) {
	data, err := os.ReadFile(i.assertionFile)
	if err != nil {
		return "", fmt.Errorf("workload: read service account token: %w", err)
	}

	assertion := strings.TrimSpace(string(data))
	if assertion == "" {
		return "", fmt.Errorf("workload: service account token file %s is empty", i.assertionFile)
	}

	return assertion, nil
}

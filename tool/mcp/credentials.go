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
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"golang.org/x/oauth2"
)

// Credentials binds an application's authorization partition to HTTP credentials.
// Equal partition keys share client-side connections and any transport catalog
// cache. The application owns cache invalidation: include a credential hash or
// authorization version in PartitionKey when a change must select fresh state.
// A partition is independent of protocol-level MCP sessions.
//
// Headers and TokenSource must be safe to read concurrently. The registry copies
// headers before applying them to requests. TokenSource, when set, supplies the
// Authorization header on each request and is responsible for caching/refresh.
type Credentials struct {
	PartitionKey string
	Headers      http.Header
	TokenSource  oauth2.TokenSource
}

// CredentialsProvider resolves credentials from the application's trusted caller
// context. It must return the same partition for operations belonging to the same
// authorization scope. Missing identity or credentials should return an error,
// not an anonymous fallback. Anonymous access can explicitly use a public key.
type CredentialsProvider func(context.Context) (Credentials, error)

// CredentialOption configures HTTP credentials.
type CredentialOption func(*Credentials) error

// NewCredentials builds credentials. WithPartitionKey is required. Authentication
// options compose with extra headers; the last Authorization-setting option wins.
func NewCredentials(opts ...CredentialOption) (Credentials, error) {
	credentials := Credentials{Headers: make(http.Header)}

	for i, opt := range opts {
		if opt == nil {
			return Credentials{}, fmt.Errorf("mcp: credential option %d is nil", i)
		}

		if err := opt(&credentials); err != nil {
			return Credentials{}, fmt.Errorf("mcp: credential option %d: %w", i, err)
		}
	}

	if err := credentials.Validate(); err != nil {
		return Credentials{}, err
	}

	credentials.Headers = credentials.Headers.Clone()

	return credentials, nil
}

// Validate checks the partition and static header syntax without resolving an
// OAuth token. A TokenSource is resolved only when preparing a request.
func (c Credentials) Validate() error {
	if c.PartitionKey == "" {
		return errors.New("mcp: credential partition key must not be empty")
	}

	for name, values := range c.Headers {
		if !validHeaderName(name) {
			return errors.New("mcp: invalid credential header name")
		}

		for _, value := range values {
			if !validHeaderValue(value) {
				return errors.New("mcp: invalid credential header value")
			}
		}
	}

	return nil
}

// WithPartitionKey selects the application-owned authorization partition. It is
// not necessarily a user ID: callers may also distinguish tenants, delegation,
// roles, scopes or credential generations in the key.
func WithPartitionKey(key string) CredentialOption {
	return func(c *Credentials) error {
		if key == "" {
			return errors.New("mcp: credential partition key must not be empty")
		}

		c.PartitionKey = key

		return nil
	}
}

// WithHeaders adds static HTTP headers, copying their values. A supplied
// Authorization header replaces any previously configured TokenSource.
func WithHeaders(headers http.Header) CredentialOption {
	return func(c *Credentials) error {
		if c.Headers == nil {
			c.Headers = make(http.Header)
		}

		for name, values := range headers {
			if !validHeaderName(name) {
				return errors.New("mcp: invalid credential header name")
			}

			canonical := http.CanonicalHeaderKey(name)

			c.Headers[canonical] = append([]string(nil), values...)
			if canonical == "Authorization" {
				c.TokenSource = nil
			}
		}

		return nil
	}
}

// WithBearerToken sets a static bearer token.
func WithBearerToken(token string) CredentialOption {
	return func(c *Credentials) error {
		if token == "" {
			return errors.New("mcp: bearer token must not be empty")
		}

		return WithHeaders(http.Header{"Authorization": {"Bearer " + token}})(c)
	}
}

// WithAPIKey sets an API key header. Multiple API key options can be combined.
func WithAPIKey(name, value string) CredentialOption {
	return func(c *Credentials) error {
		if value == "" {
			return errors.New("mcp: API key must not be empty")
		}

		return WithHeaders(http.Header{name: {value}})(c)
	}
}

// WithBasicAuth sets HTTP Basic authentication.
func WithBasicAuth(username, password string) CredentialOption {
	return func(c *Credentials) error {
		if username == "" && password == "" {
			return errors.New("mcp: basic authentication credentials must not be empty")
		}

		if strings.Contains(username, ":") {
			return errors.New("mcp: basic authentication username must not contain a colon")
		}

		encoded := base64.StdEncoding.EncodeToString([]byte(username + ":" + password))

		return WithHeaders(http.Header{"Authorization": {"Basic " + encoded}})(c)
	}
}

// WithTokenSource obtains OAuth tokens for each request. Standard oauth2 token
// sources can cache and refresh tokens; callers retain ownership of their source
// and its lifecycle context. This replaces static Authorization headers.
func WithTokenSource(source oauth2.TokenSource) CredentialOption {
	return func(c *Credentials) error {
		if source == nil {
			return errors.New("mcp: OAuth token source must not be nil")
		}

		for name := range c.Headers {
			if strings.EqualFold(name, "Authorization") {
				delete(c.Headers, name)
			}
		}

		c.TokenSource = source

		return nil
	}
}

func (c Credentials) requestHeaders() (http.Header, error) {
	if err := c.Validate(); err != nil {
		return nil, err
	}

	headers := make(http.Header, len(c.Headers))
	for name, values := range c.Headers {
		for _, value := range values {
			headers.Add(name, value)
		}
	}

	if c.TokenSource != nil {
		token, err := c.TokenSource.Token()
		if err != nil {
			return nil, &tokenSourceError{cause: err}
		}

		if token == nil || token.AccessToken == "" {
			return nil, errors.New("mcp: OAuth token source returned an empty token")
		}

		authorization := token.Type() + " " + token.AccessToken
		if !validHeaderValue(authorization) {
			return nil, errors.New("mcp: invalid OAuth authorization header")
		}

		headers.Set("Authorization", authorization)
	}

	return headers, nil
}

type tokenSourceError struct {
	cause error
}

func (*tokenSourceError) Error() string { return "mcp: OAuth token source failed" }

func (e *tokenSourceError) Unwrap() error { return e.cause }

func validHeaderName(name string) bool {
	if name == "" {
		return false
	}

	for i := range len(name) {
		c := name[i]
		if c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || strings.ContainsRune("!#$%&'*+-.^_`|~", rune(c)) {
			continue
		}

		return false
	}

	return true
}

func validHeaderValue(value string) bool {
	for i := range len(value) {
		if value[i] < ' ' && value[i] != '\t' || value[i] == 127 {
			return false
		}
	}

	return true
}

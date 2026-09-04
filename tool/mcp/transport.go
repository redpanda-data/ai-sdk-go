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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/oauth2"
)

// TransportFactory is a function that creates a new MCP transport.
type TransportFactory func() (sdkmcp.Transport, error)

// NewStdioTransport creates a TransportFactory for stdio-based MCP servers.
// The command runs as a subprocess with the given arguments and environment.
// Subprocess lifecycle is managed by the SDK with graceful shutdown on Close.
//
// SECURITY WARNING: The command parameter is executed as a system command.
// Only use values from trusted, static configuration sources. Never construct
// this parameter from user-provided input to prevent command injection attacks.
//
// Example:
//
//	factory := NewStdioTransport("npx", []string{"-y", "@modelcontextprotocol/server-everything"}, nil)
func NewStdioTransport(command string, args []string, env []string) TransportFactory {
	return func() (sdkmcp.Transport, error) {
		// Resolve command path if not absolute
		cmdPath := command
		if !filepath.IsAbs(command) {
			absPath, err := exec.LookPath(command)
			if err != nil {
				return nil, fmt.Errorf("command not found in PATH: %w", err)
			}

			cmdPath = absPath
		}

		// nosemgrep: go.lang.security.audit.dangerous-exec-command.dangerous-exec-command
		cmd := exec.Command(cmdPath, args...)

		// Append custom environment variables to parent process environment.
		// Custom vars override system vars (last value wins for duplicate keys).
		// This preserves critical system variables like PATH, HOME, etc.
		if env != nil {
			cmd.Env = append(os.Environ(), env...)
		}

		return &sdkmcp.CommandTransport{
			Command:           cmd,
			TerminateDuration: 5 * time.Second,
		}, nil
	}
}

// HTTPTransportOption configures an HTTP-based transport.
type HTTPTransportOption func(*httpTransportConfig)

// httpTransportConfig holds configuration for HTTP and SSE transports.
type httpTransportConfig struct {
	httpClient   *http.Client
	oauthConfig  *oauth2.Config
	oauthContext context.Context
	headers      map[string]string
}

// WithHTTPClient sets a custom HTTP client for the transport.
// When combined with WithOAuth, the custom client's configuration (timeouts, TLS, etc.)
// is preserved and used as the underlying transport for OAuth requests.
//
// See README.md for detailed examples.
func WithHTTPClient(client *http.Client) HTTPTransportOption {
	return func(c *httpTransportConfig) {
		c.httpClient = client
	}
}

// WithOAuth configures OAuth 2.0 authentication with automatic token refresh.
// Can be combined with WithHTTPClient to customize the underlying HTTP client.
//
// MCP OAuth Requirements:
// - Authorization Code Grant (user auth): PKCE required via oauth2.GenerateVerifier()
// - Client Credentials Grant (M2M): Use clientcredentials.Config.Client() with WithHTTPClient()
// - Both flows: Resource Indicators (RFC 8707) via WithResourceIndicator() or EndpointParams
//
// See README.md for complete OAuth examples including PKCE and resource indicators.
func WithOAuth(ctx context.Context, config *oauth2.Config) HTTPTransportOption {
	return func(c *httpTransportConfig) {
		c.oauthConfig = config
		c.oauthContext = ctx
	}
}

// WithHTTPHeaders adds custom HTTP headers to all requests.
// Useful for API keys, custom authentication, or additional metadata.
//
// Headers are applied after OAuth processing (if configured) and will override
// any conflicting headers from the base client.
//
// Example:
//
//	factory := NewStreamableTransport("https://mcp.context7.com/mcp",
//	    WithHTTPHeaders(map[string]string{
//	        "CONTEXT7_API_KEY": os.Getenv("CONTEXT7_API_KEY"),
//	        "X-Custom-Header": "value",
//	    }),
//	)
func WithHTTPHeaders(headers map[string]string) HTTPTransportOption {
	return func(c *httpTransportConfig) {
		if c.headers == nil {
			c.headers = make(map[string]string)
		}

		maps.Copy(c.headers, headers)
	}
}

// WithResourceIndicator adds RFC 8707 resource indicator to scope the access token
// to a specific MCP server, preventing token misuse across different servers.
//
// Required by MCP OAuth specification (2025-03-26). The resource parameter should
// match the MCP server's endpoint URL.
//
// Use with oauth2.Config.Exchange() for Authorization Code Grant, or use
// EndpointParams in clientcredentials.Config for Client Credentials Grant.
//
// See README.md for detailed examples.
//
// Reference: https://datatracker.ietf.org/doc/html/rfc8707
func WithResourceIndicator(resource string) oauth2.AuthCodeOption {
	return oauth2.SetAuthURLParam("resource", resource)
}

// authStatusProbe records whether an HTTP response rejected a request on
// authorization grounds, and which JSON-RPC method was rejected. It exists so
// the client can classify a rejection from the status the server actually sent,
// rather than by parsing the error text: the go-sdk reports a rejected
// handshake as `rejected by transport: sending "initialize": Forbidden`, which
// carries the status only as prose.
//
// One probe belongs to one transport, and the client builds a fresh transport
// for every connect attempt, so a recorded denial always describes the session
// (or the failed attempt) that transport served.
type authStatusProbe struct {
	// status holds the first denial status seen, 0 if none.
	status atomic.Int64
	// method holds the JSON-RPC method that first denial was for, empty when
	// it could not be determined.
	method atomic.Value // string
}

// observe records code if it denies the request in a way reconnecting cannot
// fix: 401 (unauthenticated), 403 (authenticated but not permitted) and 424
// (the gateway needs a user OAuth connection first). method is the JSON-RPC
// method of the denied request, empty if unknown.
func (p *authStatusProbe) observe(code int, method string) {
	switch code {
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusFailedDependency:
	default:
		return
	}

	if p.status.CompareAndSwap(0, int64(code)) {
		p.method.Store(method)
	}
}

// denialStatus returns the recorded status, or 0 if no request was denied.
func (p *authStatusProbe) denialStatus() int {
	return int(p.status.Load())
}

// denialMethod returns the JSON-RPC method of the recorded denial, or "".
func (p *authStatusProbe) denialMethod() string {
	method, _ := p.method.Load().(string)

	return method
}

// clientOwnedMethods are the requests a client makes for itself rather than for
// whoever asked it to do something: the handshake and the capability listings.
// A denial of one of these describes the client's own access, so it retires the
// client. A denial of anything else -- tools/call above all -- may belong to one
// caller of a session that several share, and must not.
var clientOwnedMethods = map[string]bool{
	"initialize":               true,
	"tools/list":               true,
	"prompts/list":             true,
	"resources/list":           true,
	"resources/templates/list": true,
	"ping":                     true,
}

// authProbeRoundTripper reports response status to a probe, together with the
// JSON-RPC method it was answering. It only observes: the response and error
// pass through unchanged.
type authProbeRoundTripper struct {
	base  http.RoundTripper
	probe *authStatusProbe
}

func (t *authProbeRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// Read the method before the body is consumed by the send, and only when a
	// denial could be recorded from it.
	method, body, buffered := rpcMethod(req)
	if buffered {
		req = req.Clone(req.Context())
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
		req.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(body)), nil
		}
	}

	resp, err := t.base.RoundTrip(req)
	if err == nil && resp != nil {
		t.probe.observe(resp.StatusCode, method)
	}

	return resp, err
}

// maxBufferedRPCBody caps how much of a request body is read to recover its
// JSON-RPC method. MCP messages are small; a larger body is left alone.
const maxBufferedRPCBody = 1 << 20

// rpcMethod reads the JSON-RPC method of an outgoing request, returning it
// with the buffered body when it had to consume one, so the caller can replace
// it, and whether it did.
func rpcMethod(req *http.Request) (string, []byte, bool) {
	if req.Method != http.MethodPost || req.Body == nil || req.Body == http.NoBody {
		return "", nil, false
	}

	if req.ContentLength > maxBufferedRPCBody {
		return "", nil, false
	}

	body, err := io.ReadAll(io.LimitReader(req.Body, maxBufferedRPCBody+1))
	_ = req.Body.Close()

	if err != nil || len(body) == 0 || len(body) > maxBufferedRPCBody {
		// The body cannot be restored, so the request has to go out as it is;
		// an empty reader is the honest representation of what is left.
		return "", body, len(body) <= maxBufferedRPCBody
	}

	var msg struct {
		Method string `json:"method"`
	}

	if err := json.Unmarshal(body, &msg); err != nil {
		return "", body, true
	}

	return msg.Method, body, true
}

// authDenialStatus reports the status with which the server denied a request on
// this transport, or 0 if none did.
//
// It returns 0 for any transport that cannot observe HTTP status: a stdio
// subprocess, or one built by a caller's own TransportFactory rather than by
// NewStreamableTransport or NewSSETransport. Those clients keep the old
// behaviour of reconnecting regardless of the reason.
func authDenialStatus(transport sdkmcp.Transport) int {
	probe := transportProbe(transport)
	if probe == nil {
		return 0
	}

	return probe.denialStatus()
}

// clientOwnedDenial reports the status of a denial of one of the client's own
// requests on this transport -- the handshake or a capability listing -- and 0
// for no denial, an unknown method, or a denial that may belong to a single
// caller of a shared session.
func clientOwnedDenial(transport sdkmcp.Transport) int {
	probe := transportProbe(transport)
	if probe == nil {
		return 0
	}

	status := probe.denialStatus()
	if status == 0 || !clientOwnedMethods[probe.denialMethod()] {
		return 0
	}

	return status
}

// transportProbe finds the status probe of a transport built here, or nil for
// one that cannot observe HTTP status.
func transportProbe(transport sdkmcp.Transport) *authStatusProbe {
	var httpClient *http.Client

	switch t := transport.(type) {
	case *sdkmcp.StreamableClientTransport:
		httpClient = t.HTTPClient
	case *sdkmcp.SSEClientTransport:
		httpClient = t.HTTPClient
	}

	if httpClient == nil {
		return nil
	}

	// The probe is installed outermost by newProbedHTTPClient, so one
	// assertion finds it without unwrapping whatever it wraps.
	probed, ok := httpClient.Transport.(*authProbeRoundTripper)
	if !ok {
		return nil
	}

	return probed.probe
}

// newProbedHTTPClient builds the transport's HTTP client with its responses
// observed by an authStatusProbe, installed as the outermost round tripper so
// authDenialStatus can find it again.
func newProbedHTTPClient(opts []HTTPTransportOption) *http.Client {
	client := newHTTPTransportClient(opts)

	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}

	probed := *client
	probed.Transport = &authProbeRoundTripper{base: base, probe: &authStatusProbe{}}

	return &probed
}

// headerRoundTripper wraps an http.RoundTripper to inject custom headers.
type headerRoundTripper struct {
	base    http.RoundTripper
	headers map[string]string
}

func (h *headerRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	for k, v := range h.headers {
		req.Header.Set(k, v)
	}

	return h.base.RoundTrip(req)
}

// newHTTPTransportClient creates an HTTP client from options.
// When OAuth is configured, it wraps the custom HTTP client (if provided)
// with OAuth token handling, preserving custom timeouts, TLS config, etc.
func newHTTPTransportClient(opts []HTTPTransportOption) *http.Client {
	cfg := &httpTransportConfig{
		httpClient: http.DefaultClient,
	}

	for _, opt := range opts {
		opt(cfg)
	}

	client := cfg.httpClient

	// If OAuth is configured, inject custom HTTP client into context
	if cfg.oauthConfig != nil {
		ctx := cfg.oauthContext
		// Only inject if a custom client was provided (not default)
		if client != http.DefaultClient {
			ctx = context.WithValue(ctx, oauth2.HTTPClient, client)
		}

		client = cfg.oauthConfig.Client(ctx, nil)
	}

	// If custom headers are configured, wrap the transport
	if len(cfg.headers) > 0 {
		transport := client.Transport
		if transport == nil {
			transport = http.DefaultTransport
		}

		client = &http.Client{
			Transport: &headerRoundTripper{
				base:    transport,
				headers: cfg.headers,
			},
			Timeout: client.Timeout,
		}
	}

	return client
}

// NewStreamableTransport creates a TransportFactory for bidirectional HTTP streaming
// (2025-03-26 spec). Includes automatic reconnection with exponential backoff
// (5 retries, 1-30s delays).
//
// Supports WithHTTPClient and WithOAuth options. See README.md for examples.
func NewStreamableTransport(endpoint string, opts ...HTTPTransportOption) TransportFactory {
	return func() (sdkmcp.Transport, error) {
		return &sdkmcp.StreamableClientTransport{
			Endpoint:   endpoint,
			HTTPClient: newProbedHTTPClient(opts),
		}, nil
	}
}

// NewSSETransport creates a TransportFactory for Server-Sent Events (SSE) streaming
// (2024-11-05 spec).
//
// Supports WithHTTPClient and WithOAuth options. See README.md for examples.
func NewSSETransport(endpoint string, opts ...HTTPTransportOption) TransportFactory {
	return func() (sdkmcp.Transport, error) {
		return &sdkmcp.SSEClientTransport{
			Endpoint:   endpoint,
			HTTPClient: newProbedHTTPClient(opts),
		}, nil
	}
}

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
	"fmt"
	"maps"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
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
			HTTPClient: newHTTPTransportClient(opts),
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
			HTTPClient: newHTTPTransportClient(opts),
		}, nil
	}
}

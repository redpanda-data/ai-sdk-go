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
	"errors"
	"net/http"
	"net/url"
	"strings"
)

func (r *Registry) sameOrigin(target *url.URL) bool {
	return strings.EqualFold(target.Scheme, r.endpoint.Scheme) && strings.EqualFold(target.Host, r.endpoint.Host)
}

func (r *Registry) httpClient(partition string) *http.Client {
	client := &http.Client{}
	if r.config.httpClient != nil {
		*client = *r.config.httpClient
	}

	base := client.Transport
	if base == nil {
		base = http.DefaultTransport
	}

	client.Transport = &registryRoundTripper{registry: r, partition: partition, base: base}
	checkRedirect := client.CheckRedirect
	client.CheckRedirect = func(req *http.Request, via []*http.Request) error {
		if !r.sameOrigin(req.URL) {
			return errors.New("mcp: cross-origin redirect refused")
		}

		if checkRedirect != nil {
			return checkRedirect(req, via)
		}

		if len(via) >= 10 {
			return errors.New("mcp: too many redirects")
		}

		return nil
	}

	return client
}

// MCP versions are ISO dates, so lexical order identifies support for parameter headers.
const registryParameterHeadersProtocol = "2026-07-28"

type registryRoundTripper struct {
	registry  *Registry
	partition string
	base      http.RoundTripper
}

func (t *registryRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if !t.registry.sameOrigin(req.URL) {
		return nil, errors.New("mcp: cross-origin request refused")
	}

	ctx, err := registryCredentialContext(req.Context())
	if err != nil {
		return nil, err
	}

	credentials, err := t.registry.credentials(ctx, t.partition)
	if err != nil {
		return nil, err
	}

	headers, err := credentials.requestHeaders()
	if err != nil {
		return nil, err
	}

	cloned := req.Clone(ctx)
	for name, values := range headers {
		cloned.Header[name] = append([]string(nil), values...)
	}

	if parameters, ok := req.Context().Value(registryParameterHeadersKey{}).(http.Header); ok && cloned.Header.Get("Mcp-Protocol-Version") >= registryParameterHeadersProtocol && cloned.Header.Get("Mcp-Method") == "tools/call" {
		// Replace, rather than merge, headers derived from the SDK's mutable cache.
		for name := range cloned.Header {
			if strings.HasPrefix(strings.ToLower(name), "mcp-param-") {
				delete(cloned.Header, name)
			}
		}

		for name, values := range parameters {
			cloned.Header[name] = append([]string(nil), values...)
		}
	}

	return t.base.RoundTrip(cloned)
}

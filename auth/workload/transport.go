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
	"errors"
	"net/http"
	"net/url"
	"strings"
)

// Transport returns an http.RoundTripper that sends each request through base
// (http.DefaultTransport when nil) with "Authorization: DPoP <token>" for
// audience and a DPoP proof bound to the request method, URL and token.
// audience names the one downstream service the token is for.
//
// The transport refuses a request that already has an Authorization or DPoP
// header, and a redirect to another origin. It does not retry a request the
// service rejects.
func (i *Issuer) Transport(audience string, base http.RoundTripper) (http.RoundTripper, error) {
	if audience == "" {
		return nil, errors.New("workload: audience must not be empty")
	}

	if base == nil {
		base = http.DefaultTransport
	}

	return roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return i.roundTrip(req, audience, base)
	}), nil
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func (i *Issuer) roundTrip(req *http.Request, audience string, base http.RoundTripper) (*http.Response, error) {
	if req.Header.Get("Authorization") != "" || req.Header.Get("DPoP") != "" {
		closeBody(req)

		return nil, errors.New("workload: request already has an Authorization or DPoP header")
	}

	// http.Client sets Response on a request that follows a redirect.
	if req.Response != nil && req.Response.Request != nil && !sameOrigin(req.URL, req.Response.Request.URL) {
		closeBody(req)

		return nil, errors.New("workload: cross-origin redirect refused")
	}

	accessToken, err := i.token(req.Context(), audience)
	if err != nil {
		closeBody(req)

		return nil, err
	}

	method := req.Method
	if method == "" {
		method = http.MethodGet
	}

	proof, err := i.signer.proof(method, req.URL, i.now(), "", accessToken)
	if err != nil {
		closeBody(req)

		return nil, err
	}

	cloned := req.Clone(req.Context())
	cloned.Header.Set("Authorization", "DPoP "+accessToken)
	cloned.Header.Set("DPoP", proof)

	return base.RoundTrip(cloned)
}

func sameOrigin(a, b *url.URL) bool {
	return strings.EqualFold(a.Scheme, b.Scheme) && strings.EqualFold(a.Host, b.Host)
}

// closeBody closes the request body, which RoundTrip must do even on error.
func closeBody(req *http.Request) {
	if req.Body != nil {
		_ = req.Body.Close()
	}
}

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

// Package shouldretry reads the x-should-retry response header. A server sends
// x-should-retry: false when it knows another attempt would fail too. AI
// Gateway sends it after every member of a provider group has failed.
package shouldretry

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Header is the name of the response header that tells a client whether to
// retry.
const Header = "X-Should-Retry"

// Forbidden reports whether h has x-should-retry set to false.
func Forbidden(h http.Header) bool {
	return strings.EqualFold(strings.TrimSpace(h.Get(Header)), "false")
}

// ForbiddenBy reports whether resp has x-should-retry set to false. It returns
// false for a nil resp.
func ForbiddenBy(resp *http.Response) bool {
	return resp != nil && Forbidden(resp.Header)
}

// Apply sets Retryable to false on the *llm.ProviderError inside err when
// forbidden is true. It returns err either way.
func Apply(err error, forbidden bool) error {
	if !forbidden {
		return err
	}

	if pe, ok := errors.AsType[*llm.ProviderError](err); ok {
		pe.Retryable = false
	}

	return err
}

// An SDK can report an error without the response it came from. A stream
// that starts with a 200 and then sends an error event is one case, and genai
// drops the response headers from every error. So a provider gets a flag from
// Watch before it sends a request. Transport, Middleware or Doer, whichever the
// provider's SDK accepts, sets the flag when the response has
// x-should-retry: false. The provider passes the flag to Apply for any error
// the call returns.
type watchKey struct{}

// Watch returns a copy of ctx that carries a new flag, and the flag itself.
// The flag is set when any response to a request made with that context has
// x-should-retry: false, whatever its status.
func Watch(ctx context.Context) (context.Context, *atomic.Bool) {
	flag := new(atomic.Bool)

	return context.WithValue(ctx, watchKey{}, flag), flag
}

// record sets the flag on req's context when resp has x-should-retry: false.
func record(req *http.Request, resp *http.Response) {
	if !ForbiddenBy(resp) {
		return
	}

	if flag, ok := req.Context().Value(watchKey{}).(*atomic.Bool); ok {
		flag.Store(true)
	}
}

// Transport is an http.RoundTripper that sets the flag from Watch when a
// response has x-should-retry: false.
type Transport struct {
	// Base sends the request. If Base is nil, Transport uses
	// http.DefaultTransport.
	Base http.RoundTripper
}

func (t Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	base := t.Base
	if base == nil {
		base = http.DefaultTransport
	}

	resp, err := base.RoundTrip(req)
	if err == nil {
		record(req, resp)
	}

	return resp, err
}

// Middleware sets the flag from Watch when a response has
// x-should-retry: false. Its type matches option.Middleware in the Anthropic
// and OpenAI SDKs, so a provider passes it with option.WithMiddleware.
func Middleware(req *http.Request, next func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	resp, err := next(req)
	if err == nil {
		record(req, resp)
	}

	return resp, err
}

// Doer is the HTTP client interface the AWS SDK takes.
type Doer interface {
	Do(req *http.Request) (*http.Response, error)
}

// WrapDoer returns a Doer that sends requests through d and sets the flag
// from Watch when a response has x-should-retry: false.
func WrapDoer(d Doer) Doer {
	return doer{base: d}
}

type doer struct {
	base Doer
}

func (d doer) Do(req *http.Request) (*http.Response, error) {
	resp, err := d.base.Do(req)
	if err == nil {
		record(req, resp)
	}

	return resp, err
}

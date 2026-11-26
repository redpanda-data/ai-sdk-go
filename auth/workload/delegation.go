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
	"errors"
	"net/http"
	"net/url"
	"strings"
)

const (
	exchangeGrant = "urn:ietf:params:oauth:grant-type:token-exchange"
	accessTypeURN = "urn:ietf:params:oauth:token-type:access_token"
)

// SubjectToken returns the token that a caller sent to the agent in the
// Authorization header of req, with the Bearer or the DPoP scheme. The
// workspace gateway sends a user's token for the agent as a Bearer token; an
// agent earlier in a chain sends its on-behalf-of token with DPoP. Pass the
// result to OnBehalfOf. The agent does not need to verify the token: the
// issuer does, and refuses a token that is not for this agent.
func SubjectToken(req *http.Request) (string, error) {
	scheme, token, ok := strings.Cut(req.Header.Get("Authorization"), " ")
	token = strings.TrimSpace(token)

	if !ok || token == "" || (!strings.EqualFold(scheme, "Bearer") && !strings.EqualFold(scheme, "DPoP")) || !validTokenValue(token) {
		return "", errors.New("workload: request has no Bearer or DPoP token")
	}

	return token, nil
}

// OnBehalfOf returns an http.RoundTripper that calls audience for the user of
// subjectToken: the token the agent received for this request (SubjectToken).
// It exchanges subjectToken at the token endpoint (RFC 8693, with the service
// account token as the client assertion and a DPoP proof) for an
// on-behalf-of token whose audience is the next hop, and sends each request
// with that token and a DPoP proof, like Transport.
//
// audience is the next hop: a service, or another agent of the workspace.
// scopes are the scopes to ask for. The issuer intersects them with the scope
// of subjectToken, the agent's permissions and the user's consent, so the new
// token never has more than subjectToken. With no scopes, the token has an
// empty scope.
//
// The transport exchanges on the first request and keeps the token until
// shortly before it expires, then exchanges subjectToken again. It does not
// share tokens with other transports, because each subject token is for one
// user. A refusal comes back as a *TokenError.
func (i *Issuer) OnBehalfOf(subjectToken, audience string, scopes []string, base http.RoundTripper) (http.RoundTripper, error) {
	if subjectToken == "" || !validTokenValue(subjectToken) {
		return nil, errors.New("workload: subject token must not be empty")
	}

	if audience == "" {
		return nil, errors.New("workload: audience must not be empty")
	}

	if base == nil {
		base = http.DefaultTransport
	}

	entry := &cachedToken{lock: make(chan struct{}, 1)}
	scope := strings.Join(scopes, " ")

	return roundTripFunc(func(req *http.Request) (*http.Response, error) {
		return i.roundTrip(req, func(ctx context.Context) (string, error) {
			return i.delegated(ctx, entry, subjectToken, audience, scope)
		}, base)
	}), nil
}

// delegated returns the token in entry, or exchanges subjectToken for a new
// one.
func (i *Issuer) delegated(ctx context.Context, entry *cachedToken, subjectToken, audience, scope string) (string, error) {
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

	assertion, err := i.readAssertion()
	if err != nil {
		return "", err
	}

	form := url.Values{
		"grant_type":            {exchangeGrant},
		"subject_token":         {subjectToken},
		"subject_token_type":    {accessTypeURN},
		"requested_token_type":  {accessTypeURN},
		"client_assertion_type": {clientAssertionType},
		"client_assertion":      {assertion},
		"audience":              {audience},
	}
	if scope != "" {
		form.Set("scope", scope)
	}

	accessToken, lifetime, err := i.send(ctx, form)
	if err != nil {
		return "", err
	}

	entry.accessToken = accessToken
	entry.refreshAt = requested.Add(lifetime - min(maxRefreshMargin, lifetime/2))

	return accessToken, nil
}

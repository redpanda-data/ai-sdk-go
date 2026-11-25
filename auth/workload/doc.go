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

// Package workload authenticates an agent running in Kubernetes to the
// services it calls, with DPoP-bound access tokens from a workspace token
// issuer.
//
// EXPERIMENTAL: this package is unreleased and its API may change or go away.
// NewIssuer returns ErrExperimentalDisabled unless the caller passes
// WithExperimentalDPoP, so nothing changes for a caller that does not opt in.
//
// The agent exchanges its projected service account token, read from a file
// on each token request because Kubernetes rotates it, for an access token
// scoped to one downstream service (the audience). Every token request and
// every call to that service carries a DPoP proof (RFC 9449) signed with a
// P-256 key that lives only in this process, and the issuer binds each token
// to that key.
//
//	issuer, err := workload.NewIssuer(
//	    "https://issuer.example.com/token",
//	    "/var/run/secrets/tokens/workload",
//	    workload.WithExperimentalDPoP(),
//	)
//	if err != nil {
//	    return err
//	}
//
//	transport, err := issuer.Transport("https://tools.example.com", nil)
//	if err != nil {
//	    return err
//	}
//
//	client := &http.Client{Transport: transport}
//
// The transport sends the token on every request it carries, so put it only
// on a client for the audience's service. It refuses to follow a redirect to
// another origin with the token.
package workload

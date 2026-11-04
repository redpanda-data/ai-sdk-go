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

package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// seen is one request a fake host received.
type seen struct {
	method, path, rawPath string
	header                http.Header
	body                  map[string]any
}

// fakeHosts answers every host's requests from one server: answer picks the
// status and body for each request, and every request is kept.
func fakeHosts(t *testing.T, answer func(r *http.Request, body map[string]any) (int, string)) (*hosts, func() []seen) {
	t.Helper()

	var (
		mu   sync.Mutex
		reqs []seen
	)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)

		var body map[string]any
		_ = json.Unmarshal(raw, &body)

		mu.Lock()

		reqs = append(reqs, seen{r.Method, r.URL.Path, r.URL.RawPath, r.Header.Clone(), body})
		mu.Unlock()

		status, reply := answer(r, body)
		// AWS names an error's type in a header, as Bedrock does.
		if awsTypes[status] != "" && (strings.HasPrefix(r.URL.Path, "/model/") || strings.HasPrefix(r.URL.Path, "/foundation-model-availability/")) {
			w.Header().Set("X-Amzn-Errortype", awsTypes[status]+":http://internal.amazon.com/coral/com.amazon.bedrock/")
		}

		w.Header().Set("Request-Id", "req_anthropic")
		w.Header().Set("X-Request-Id", "req_openai")
		w.Header().Set("X-Amzn-Requestid", "req-aws")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	t.Cleanup(srv.Close)

	h := &hosts{
		client:       srv.Client(),
		anthropicURL: srv.URL, anthropicKey: "sk-ant",
		openaiURL: srv.URL, openaiKey: "sk-openai",
		geminiURL: srv.URL, geminiKey: "gemini-key",
		awsURL: func(string, string) string { return srv.URL },
		awsCredentials: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: "AKIDPROBE", SecretAccessKey: "secret"}, nil
		}),
	}

	return h, func() []seen {
		mu.Lock()
		defer mu.Unlock()

		return append([]seen(nil), reqs...)
	}
}

// awsTypes are the error types the fake Bedrock answers with, by status.
var awsTypes = map[int]string{403: "AccessDeniedException", 404: "ResourceNotFoundException"}

// subscribed is Bedrock's availability answer for a model the account may
// call.
const subscribed = `{"agreementAvailability": {"status": "AVAILABLE"}, "authorizationStatus": "AUTHORIZED",
	"entitlementAvailability": "AVAILABLE", "regionAvailability": "AVAILABLE"}`

// TestHostsSendATinyRequest: each call is one short prompt with a small
// output cap, to the host's real path, with its key; Bedrock's two requests
// are SigV4-signed for the probe's region, with the model ID escaped as
// AWS's SDKs escape it.
func TestHostsSendATinyRequest(t *testing.T) {
	t.Parallel()

	h, requests := fakeHosts(t, func(r *http.Request, _ map[string]any) (int, string) {
		if strings.HasPrefix(r.URL.Path, "/foundation-model-availability/") {
			return 200, subscribed
		}

		return 200, `{"responseId": "resp-gemini"}`
	})

	probes := []Probe{
		{Provider: "anthropic", ID: "claude-z", call: "anthropic"},
		{Provider: "openai", ID: "gpt-z", call: "openai_responses"},
		{Provider: "openai", ID: "gpt-old", call: "openai_chat"},
		{Provider: "google", ID: "gemini-z", call: "gemini"},
		{Provider: "bedrock", ID: "eu.anthropic.claude-z-v1:0", Region: "eu-central-1", call: "bedrock"},
	}

	got := make([]string, 0, len(probes))

	for _, p := range probes {
		r := h.probe(t.Context(), p)
		got = append(got, r.ID+" "+r.Outcome+" "+r.Detail+" "+r.RequestID)
	}

	assert.Equal(t, []string{
		"claude-z callable 200 req_anthropic",
		"gpt-z callable 200 req_openai",
		"gpt-old callable 200 req_openai",
		"gemini-z callable 200 resp-gemini",
		"eu.anthropic.claude-z-v1:0 callable 200 req-aws",
	}, got)

	reqs := requests()
	require.Len(t, reqs, 6)

	anthropic, responses, chat, gemini, availability, converse := reqs[0], reqs[1], reqs[2], reqs[3], reqs[4], reqs[5]

	assert.Equal(t, "POST /v1/messages", anthropic.method+" "+anthropic.path)
	assert.Equal(t, "sk-ant", anthropic.header.Get("X-Api-Key"))
	assert.Equal(t, "2023-06-01", anthropic.header.Get("Anthropic-Version"))
	assert.Equal(t, "claude-z", anthropic.body["model"])
	assert.InDelta(t, 8, anthropic.body["max_tokens"], 0)

	assert.Equal(t, "POST /v1/responses", responses.method+" "+responses.path)
	assert.Equal(t, "Bearer sk-openai", responses.header.Get("Authorization"))
	assert.Equal(t, "gpt-z", responses.body["model"])
	assert.InDelta(t, 16, responses.body["max_output_tokens"], 0)

	assert.Equal(t, "POST /v1/chat/completions", chat.method+" "+chat.path)
	assert.Equal(t, "gpt-old", chat.body["model"])
	assert.InDelta(t, 16, chat.body["max_completion_tokens"], 0)

	assert.Equal(t, "POST /v1beta/models/gemini-z:generateContent", gemini.method+" "+gemini.path)
	assert.Equal(t, "gemini-key", gemini.header.Get("X-Goog-Api-Key"))
	config, _ := gemini.body["generationConfig"].(map[string]any)
	assert.InDelta(t, 8, config["maxOutputTokens"], 0)

	// The pre-check names the foundation model, without the profile prefix.
	assert.Equal(t, "GET /foundation-model-availability/anthropic.claude-z-v1:0", availability.method+" "+availability.path)
	assert.Equal(t, "/foundation-model-availability/anthropic.claude-z-v1%3A0", availability.rawPath)
	assert.Contains(t, availability.header.Get("Authorization"), "Credential=AKIDPROBE/")
	assert.Contains(t, availability.header.Get("Authorization"), "/eu-central-1/bedrock/aws4_request")

	assert.Equal(t, "POST /model/eu.anthropic.claude-z-v1:0/converse", converse.method+" "+converse.path)
	assert.Equal(t, "/model/eu.anthropic.claude-z-v1%3A0/converse", converse.rawPath)
	assert.Contains(t, converse.header.Get("Authorization"), "/eu-central-1/bedrock/aws4_request")
	inference, _ := converse.body["inferenceConfig"].(map[string]any)
	assert.InDelta(t, 8, inference["maxTokens"], 0)
}

// TestBedrockInvokesOnlyAfterThePreCheck: a model is invoked only when the
// read-only availability check says the account is entitled, has the
// agreement and is authorized, because the CI role may subscribe to AWS
// Marketplace on a first call. Any other answer, or no answer, calls
// nothing.
func TestBedrockInvokesOnlyAfterThePreCheck(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name         string
		status       int
		availability string
		want         string
		invoked      bool
	}{
		{"subscribed", 200, subscribed, "callable 200", true},
		{"no entitlement", 200, strings.Replace(subscribed, `"entitlementAvailability": "AVAILABLE"`, `"entitlementAvailability": "NOT_AVAILABLE"`, 1), "not_subscribed entitlement NOT_AVAILABLE", false},
		{"agreement pending", 200, strings.Replace(subscribed, `{"status": "AVAILABLE"}`, `{"status": "PENDING"}`, 1), "not_subscribed agreement PENDING", false},
		{"not authorized", 200, strings.Replace(subscribed, `"AUTHORIZED"`, `"NOT_AUTHORIZED"`, 1), "not_subscribed authorization NOT_AUTHORIZED", false},
		{"empty answer", 200, `{}`, "not_subscribed entitlement none / agreement none / authorization none", false},
		{"pre-check denied", 403, `{"message": "not authorized to perform: bedrock:GetFoundationModelAvailability"}`, "inconclusive pre-check 403 AccessDeniedException", false},
		{"unknown model", 404, `{"message": "Model not found"}`, "not_found pre-check 404 ResourceNotFoundException", false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			h, requests := fakeHosts(t, func(r *http.Request, _ map[string]any) (int, string) {
				if strings.HasPrefix(r.URL.Path, "/foundation-model-availability/") {
					return tc.status, tc.availability
				}

				return 200, `{}`
			})

			r := h.probe(t.Context(), Probe{Provider: "bedrock", ID: "us.anthropic.claude-z", Region: "us-east-1", call: "bedrock"})
			assert.Equal(t, tc.want, r.Outcome+" "+r.Detail)

			invoked := false
			for _, req := range requests() {
				invoked = invoked || strings.HasSuffix(req.path, "/converse")
			}

			assert.Equal(t, tc.invoked, invoked)
		})
	}
}

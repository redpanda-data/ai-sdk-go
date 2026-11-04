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
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	signerv4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// prompt is every probe's whole input; with the small output caps below, a
// call costs a fraction of a cent on any model.
const prompt = "Reply OK"

// hosts sends the probes' requests over plain HTTP, not through the SDK's
// providers, which only call models their catalog already lists. The URLs
// are fields so tests can point them at a fake.
type hosts struct {
	client                     *http.Client
	anthropicURL, anthropicKey string
	openaiURL, openaiKey       string
	geminiURL, geminiKey       string
	// awsURL is a Bedrock endpoint: service "bedrock" (the control plane) or
	// "bedrock-runtime", in region; signed adds the path.
	awsURL         func(service, region string) string
	awsCredentials aws.CredentialsProvider
}

// answer is what a host said: its status (0 when it didn't answer), the
// error code and message when it refused, and its ID for the request.
type answer struct {
	status        int
	code, message string
	requestID     string
	body          []byte
}

// probe sends one probe's request and returns the probe with its outcome.
func (h *hosts) probe(ctx context.Context, p Probe) Probe {
	user := map[string]any{"role": "user", "content": prompt}

	switch p.wire.Protocol {
	case catalog.WireProtocolAnthropicMessages:
		if h.anthropicKey == "" {
			return skip(p, "no Anthropic key in this run")
		}

		body := map[string]any{"model": p.ID, "max_tokens": 8, "messages": []any{user}}

		return result(p, h.post(ctx, h.anthropicURL+"/v1/messages", body, "Request-Id",
			map[string]string{"X-Api-Key": h.anthropicKey, "Anthropic-Version": "2023-06-01"}))
	case catalog.WireProtocolOpenAIResponses, catalog.WireProtocolOpenAIChatCompletions:
		if h.openaiKey == "" {
			return skip(p, "no OpenAI key in this run")
		}

		// 16 is the Responses API's smallest output cap.
		path, body := "/v1/responses", map[string]any{"model": p.ID, "input": prompt, "max_output_tokens": 16}
		if p.wire.Protocol == catalog.WireProtocolOpenAIChatCompletions {
			path, body = "/v1/chat/completions", map[string]any{"model": p.ID, "messages": []any{user}, "max_completion_tokens": 16}
		}

		return result(p, h.post(ctx, h.openaiURL+path, body, "X-Request-Id", map[string]string{"Authorization": "Bearer " + h.openaiKey}))
	case catalog.WireProtocolGeminiGenerateContent:
		if h.geminiKey == "" {
			return skip(p, "no Gemini API key in this run")
		}

		body := map[string]any{
			"contents":         []any{map[string]any{"parts": []any{map[string]any{"text": prompt}}}},
			"generationConfig": map[string]any{"maxOutputTokens": 8},
		}

		return result(p, h.post(ctx, h.geminiURL+"/v1beta/models/"+url.PathEscape(p.ID)+":generateContent", body, "",
			map[string]string{"X-Goog-Api-Key": h.geminiKey}))
	case catalog.WireProtocolBedrockConverse:
		return h.bedrock(ctx, p)
	}

	return skip(p, noRequest)
}

// bedrock calls a Bedrock offering only after a read-only check that the
// account already subscribes to its model: the CI role may subscribe to
// AWS Marketplace on a first call, and a probe must never do that. A failed
// or refused check calls nothing.
func (h *hosts) bedrock(ctx context.Context, p Probe) Probe {
	if h.awsCredentials == nil {
		return skip(p, "no AWS credentials in this run")
	}

	// The check takes the foundation model, a profile's ID without its prefix.
	model := p.ID
	if prefix, rest, ok := strings.Cut(p.ID, "."); ok && reconcile.PrefixNames[prefix] != "" {
		model = rest
	}

	check := h.signed(ctx, http.MethodGet, "bedrock", p.Region, "/foundation-model-availability/"+pathLabel(model), nil)
	if check.status != http.StatusOK {
		outcome := classify(check.status, check.code, check.message)
		if outcome != outcomeNotFound {
			outcome = outcomeInconclusive
		}

		return done(p, outcome, "pre-check "+detail(check), check.requestID)
	}

	//nolint:tagliatelle // AWS's field names
	var available struct {
		Agreement struct {
			Status string `json:"status"`
		} `json:"agreementAvailability"`
		Authorization string `json:"authorizationStatus"`
		Entitlement   string `json:"entitlementAvailability"`
	}

	_ = json.Unmarshal(check.body, &available)

	var missing []string

	for _, field := range []struct{ name, value, want string }{
		{"entitlement", available.Entitlement, "AVAILABLE"},
		{"agreement", available.Agreement.Status, "AVAILABLE"},
		{"authorization", available.Authorization, "AUTHORIZED"},
	} {
		if field.value != field.want {
			missing = append(missing, field.name+" "+cmp.Or(field.value, "none"))
		}
	}

	if len(missing) > 0 {
		return done(p, outcomeNotSubscribed, strings.Join(missing, " / "), check.requestID)
	}

	body := map[string]any{
		"messages":        []any{map[string]any{"role": "user", "content": []any{map[string]any{"text": prompt}}}},
		"inferenceConfig": map[string]any{"maxTokens": 8},
	}

	return result(p, h.signed(ctx, http.MethodPost, "bedrock-runtime", p.Region, "/model/"+pathLabel(p.ID)+"/converse", body))
}

// pathLabel escapes an ID for a Bedrock path as AWS's SDKs escape a path
// label, ":" too; the signer escapes the path once more, as every AWS
// service but S3 expects.
func pathLabel(id string) string {
	return strings.ReplaceAll(url.PathEscape(id), ":", "%3A")
}

// newRequest is a request with body as JSON, or with no body when it is
// nil, and the bytes it sends.
func newRequest(ctx context.Context, method, target string, body any) (*http.Request, []byte, error) {
	var payload []byte

	if body != nil {
		var err error
		if payload, err = json.Marshal(body); err != nil {
			return nil, nil, err
		}
	}

	req, err := http.NewRequestWithContext(ctx, method, target, bytes.NewReader(payload))
	if err != nil {
		return nil, nil, err
	}

	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	return req, payload, nil
}

// post sends a JSON request with headers, reading the request's ID from
// idHeader, or from the body when that is empty.
func (h *hosts) post(ctx context.Context, target string, body any, idHeader string, headers map[string]string) answer {
	req, _, err := newRequest(ctx, http.MethodPost, target, body)
	if err != nil {
		return answer{message: err.Error()}
	}

	for key, value := range headers {
		req.Header.Set(key, value)
	}

	return h.send(req, idHeader)
}

// signed sends a SigV4-signed request to a Bedrock service in region, for
// an already escaped path.
func (h *hosts) signed(ctx context.Context, method, service, region, path string, body any) answer {
	req, payload, err := newRequest(ctx, method, h.awsURL(service, region)+path, body)
	if err != nil {
		return answer{message: err.Error()}
	}

	creds, err := h.awsCredentials.Retrieve(ctx)
	if err != nil {
		return answer{message: err.Error()}
	}

	hash := sha256.Sum256(payload)
	if err := signerv4.NewSigner().SignHTTP(ctx, creds, req, hex.EncodeToString(hash[:]), "bedrock", region, time.Now()); err != nil {
		return answer{message: err.Error()}
	}

	return h.send(req, "X-Amzn-Requestid")
}

// send makes the request and reads the host's answer in any of the four
// hosts' shapes: an error object with a type, code or status (Anthropic,
// OpenAI, Gemini), or an error type header and a message (AWS).
func (h *hosts) send(req *http.Request, idHeader string) answer {
	resp, err := h.client.Do(req) //nolint:gosec // the providers' fixed hosts; plan checks each ID against the offering pattern
	if err != nil {
		return answer{message: err.Error()}
	}
	defer resp.Body.Close()

	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))

	//nolint:tagliatelle // Gemini's responseId
	var body struct {
		Error struct {
			Type    string `json:"type"`
			Code    any    `json:"code"`
			Status  string `json:"status"`
			Message string `json:"message"`
		} `json:"error"`
		Message    string `json:"message"`
		ResponseID string `json:"responseId"`
	}

	_ = json.Unmarshal(raw, &body)

	a := answer{status: resp.StatusCode, body: raw, message: body.Error.Message + body.Message, requestID: body.ResponseID}
	if idHeader != "" {
		a.requestID = resp.Header.Get(idHeader)
	}

	if resp.StatusCode >= http.StatusBadRequest {
		code, _ := body.Error.Code.(string)
		awsType, _, _ := strings.Cut(resp.Header.Get("X-Amzn-Errortype"), ":")
		a.code = cmp.Or(awsType, body.Error.Status, code, body.Error.Type)
	}

	return a
}

// result is a probe with the outcome of the host's answer.
func result(p Probe, a answer) Probe {
	return done(p, classify(a.status, a.code, a.message), detail(a), a.requestID)
}

func done(p Probe, outcome, what, requestID string) Probe {
	p.Outcome, p.Detail, p.RequestID = outcome, shown(what), shown(requestID)

	return p
}

// detail says what the host answered in a few words: the status and the
// error code, never the message, which can quote anything.
func detail(a answer) string {
	if a.status == 0 {
		return "no answer"
	}

	return strings.TrimSpace(fmt.Sprintf("%d %s", a.status, a.code))
}

// shown keeps a host's value to what reconcile.SafeValue lets into the
// public PR body: other characters dropped, cut to 64, and empty when it
// still isn't safe.
func shown(s string) string {
	s = strings.Map(func(r rune) rune {
		if reconcile.ValuePattern.MatchString(string(r)) {
			return r
		}

		return -1
	}, s)

	if s = s[:min(len(s), 64)]; !reconcile.SafeValue(s) {
		return ""
	}

	return s
}

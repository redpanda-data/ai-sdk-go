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

package fireworks_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

type sdkRuntimeBody struct {
	io.ReadCloser

	data     bytes.Buffer
	finished time.Time
}

func (b *sdkRuntimeBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)

	_, _ = b.data.Write(p[:n])
	if err != nil && b.finished.IsZero() {
		b.finished = time.Now()
	}

	return n, err
}

func (b *sdkRuntimeBody) Close() error {
	if b.finished.IsZero() {
		b.finished = time.Now()
	}

	return b.ReadCloser.Close()
}

type sdkRuntimeWire struct {
	key       string
	header    http.Header
	limit     int
	maxOutput int
	allowed   int
	current   contractCase
	report    contractManifest
	bodies    []*sdkRuntimeBody
	path      string
}

func (w *sdkRuntimeWire) RoundTrip(request *http.Request) (*http.Response, error) {
	if request.URL.Scheme != "https" || request.URL.Host != "api.fireworks.ai" || request.URL.Path != "/inference/v1/messages" {
		return nil, errors.New("unexpected SDK runtime HTTP target")
	}

	if request.Header.Get("Authorization") != "Bearer "+w.key || request.Header.Get("X-Api-Key") != "" {
		return nil, errors.New("SDK runtime must send only bearer authorization")
	}

	if len(w.report.Evidence) >= w.allowed {
		return nil, errors.New("unexpected SDK retry or extra HTTP request")
	}

	raw, err := io.ReadAll(io.LimitReader(request.Body, int64(w.limit)+1))
	if err != nil {
		return nil, errors.New("could not inspect SDK runtime request")
	}

	if len(raw) > w.limit {
		return nil, errors.New("SDK runtime request exceeded the bounded probe size")
	}

	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil || contractNumber(body["max_tokens"]) != int64(w.maxOutput) {
		return nil, errors.New("SDK runtime did not preserve the explicit output-token bound")
	}

	request.Body = io.NopCloser(bytes.NewReader(raw))
	maps.Copy(request.Header, w.header)

	header := map[string]string{"Content-Type": request.Header.Get("Content-Type"), "Anthropic-Version": request.Header.Get("Anthropic-Version")}
	for name := range w.header {
		header[name] = request.Header.Get(name)
	}

	e := contractExchange{Method: request.Method, Path: request.URL.RequestURI(), Authorization: "valid", Request: raw, RequestHeaders: header, StartedAt: time.Now().UTC()}
	response, err := http.DefaultTransport.RoundTrip(request)
	e.ElapsedMS = time.Since(e.StartedAt).Milliseconds()

	row := contractEvidence{Case: w.current, Exchanges: []contractExchange{e}, Capture: contractCaptureProvenance{SDKRevision: w.report.SDKRevision, CapturedAt: e.StartedAt, RequestSHA256: []string{contractRequestHash(raw)}}}
	if err != nil {
		row.Exchanges[0].ErrorClass = "transport"
		w.report.Evidence = append(w.report.Evidence, row)
		w.bodies = append(w.bodies, nil)

		return nil, errors.New("SDK runtime upstream transport failed")
	}

	row.Exchanges[0].Status = response.StatusCode
	row.Exchanges[0].ResponseHeaders = map[string]string{"Content-Type": response.Header.Get("Content-Type")}
	observedBody := &sdkRuntimeBody{ReadCloser: response.Body}
	response.Body = observedBody

	w.report.Evidence = append(w.report.Evidence, row)
	w.bodies = append(w.bodies, observedBody)

	return response, nil
}

func (w *sdkRuntimeWire) checkpoint(t *testing.T) {
	t.Helper()

	if (t.Failed() || t.Skipped()) && !strings.HasSuffix(w.report.RunKind, "_failed") {
		w.report.RunKind += "_failed"
	}

	report := w.report
	report.Evidence = make([]contractEvidence, len(w.report.Evidence))

	for n := range w.report.Evidence {
		if w.bodies[n] != nil {
			w.report.Evidence[n].Exchanges[0].Response = w.bodies[n].data.String()
			if !w.bodies[n].finished.IsZero() {
				w.report.Evidence[n].Exchanges[0].ElapsedMS = w.bodies[n].finished.Sub(w.report.Evidence[n].Exchanges[0].StartedAt).Milliseconds()
			}
		}

		report.Evidence[n] = w.report.Evidence[n]
		report.Evidence[n].Exchanges = make([]contractExchange, len(w.report.Evidence[n].Exchanges))
		report.Evidence[n].Capture.RequestSHA256 = nil

		for j, exchange := range w.report.Evidence[n].Exchanges {
			exchange.ResponseHeaders = maps.Clone(exchange.ResponseHeaders)
			report.Evidence[n].Exchanges[j] = contractSanitize(t, exchange, w.key)
			report.Evidence[n].Capture.RequestSHA256 = append(report.Evidence[n].Capture.RequestSHA256, contractRequestHash(report.Evidence[n].Exchanges[j].Request))
		}
	}

	contractWriteEvidence(t, w.path, report, nil)
}

func newSDKRuntimeWire(t *testing.T, key, path string, limit int) *sdkRuntimeWire {
	t.Helper()
	revision, err := exec.CommandContext(t.Context(), "git", "rev-parse", "HEAD").Output()
	require.NoError(t, err)

	return &sdkRuntimeWire{key: key, path: path, limit: limit, report: contractManifest{Version: 1, Scope: "sdk_generate_and_generate_events", RunKind: "live_sdk_first_binding", CapturedAt: time.Now().UTC(), SDKRevision: strings.TrimSpace(string(revision)), Blocked: []string{"native_verified records native final-usage and price comparison with SDK usage, not SDK feature qualification; a failed run or failed row does not qualify the feature"}}}
}

func sdkRuntimeCall(t *testing.T, wire *sdkRuntimeWire, model llm.Model, request *llm.Request, stream bool, feature string) *llm.Response {
	t.Helper()

	before := len(wire.report.Evidence)
	wire.allowed = before + 1

	t.Cleanup(func() {
		if before == len(wire.report.Evidence)-1 && (t.Failed() || t.Skipped()) {
			wire.report.Evidence[before].Failed = true
		}

		wire.checkpoint(t)
	})

	wire.current = contractCase{Model: model.Name(), API: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect, Stream: stream, Feature: feature}
	defer wire.checkpoint(t)

	response := sdkRuntimeGenerate(t, model, request, stream, wire.key)
	require.Len(t, wire.report.Evidence, before+1, "one SDK invocation must issue exactly one upstream request")
	wire.checkpoint(t)
	e := wire.report.Evidence[before].Exchanges[0]
	sdkRuntimeAssertUsage(t, wire.current, e, response)
	raw := contractParse(t, wire.current, e)
	contractAssertPrice(t, wire.current, raw)
	wire.report.KnownRunUSD += contractCost(wire.current, raw.Input, raw.Cached, raw.Output)
	wire.report.Evidence[before].NativeVerified = true

	t.Logf("SDK HTTP model=%s stream=%t feature=%s input=%d cache=%d output=%d finish=%s", model.Name(), stream, feature, response.Usage.InputTokens, response.Usage.CachedInputTokens, response.Usage.OutputTokens+response.Usage.ReasoningTokens, response.FinishReason)

	return response
}

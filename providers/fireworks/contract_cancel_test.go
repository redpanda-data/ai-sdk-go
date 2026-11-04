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
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func contractAbort(t *testing.T, client *http.Client, req *http.Request, body []byte, headers map[string]string, key string) contractExchange {
	t.Helper()

	ctx, cancel := context.WithCancel(req.Context())
	defer cancel()
	var written atomic.Bool
	trace := &httptrace.ClientTrace{WroteRequest: func(info httptrace.WroteRequestInfo) {
		if info.Err == nil {
			written.Store(true)
			time.AfterFunc(25*time.Millisecond, cancel)
		}
	}}
	req = req.WithContext(httptrace.WithClientTrace(ctx, trace))
	resp, err := client.Do(req) //nolint:gosec // Only the fixed Fireworks origin or a strict local replay server reaches this helper.
	e := contractExchange{Method: req.Method, Path: req.URL.Path, Authorization: "valid", RequestHeaders: headers, Request: body, ResponseHeaders: map[string]string{}}

	if resp != nil {
		defer resp.Body.Close()

		e.Status = resp.StatusCode
		e.ResponseHeaders, e.ResponseHeaderNames = contractResponseMetadata(resp.Header, key)
		b, readErr := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		e.Response, e.ResponseTruncated = contractCancelPrefix(b, resp.Header.Get("Content-Type"))

		if readErr != nil {
			err = readErr
			e.ResponseTruncated = true
		}
	}

	e.RequestWritten = written.Load()

	e.Cancelled = errors.Is(err, context.Canceled)
	if !e.RequestWritten {
		e.ErrorClass = "cancel_request_not_written"
	} else if !e.Cancelled {
		e.ErrorClass = "cancellation_not_observed"
	}

	return e
}

func contractCancelPrefix(raw []byte, contentType string) (string, bool) {
	if !strings.Contains(contentType, "text/event-stream") {
		if json.Valid(raw) {
			return string(raw), false
		}

		return "", len(raw) > 0
	}

	lines := strings.SplitAfter(string(raw), "\n")
	var kept strings.Builder

	for _, line := range lines {
		if data, ok := strings.CutPrefix(line, "data:"); ok {
			data = strings.TrimSpace(data)
			if data != contractSSEDone && !json.Valid([]byte(data)) {
				return kept.String(), true
			}
		}

		kept.WriteString(line)
	}

	return kept.String(), false
}

func contractCancelledUsage(c contractCase, e contractExchange) (contractResult, bool) {
	if !c.Stream {
		if !json.Valid([]byte(e.Response)) {
			return contractResult{}, false
		}

		var body map[string]any
		if err := json.Unmarshal([]byte(e.Response), &body); err != nil {
			return contractResult{}, false
		}

		r := contractParseObject(c.API, body)

		return r, r.HasInput && r.HasOutput
	}

	r := contractResult{}
	final := false

	for line := range strings.SplitSeq(e.Response, "\n") {
		data, ok := strings.CutPrefix(line, "data:")
		if !ok || strings.TrimSpace(data) == contractSSEDone {
			continue
		}

		var body map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(data)), &body); err != nil {
			continue
		}

		if model := contractString(body["model"]); model != "" {
			r.Model = model
		}

		if message := contractObject(body["message"]); message != nil {
			r.Model = contractString(message["model"])
			contractUsage(&r, c.API, contractObject(message["usage"]))
		}

		contractUsage(&r, c.API, contractObject(body["usage"]))

		if body["type"] == "message_delta" && contractObject(body["usage"])["output_tokens"] != nil {
			final = true
		}

		for _, choice := range contractArray(body["choices"]) {
			if contractObject(choice)["finish_reason"] != nil {
				final = true
			}
		}

		if response := contractObject(body["response"]); response != nil {
			if body["type"] == "response.completed" || body["type"] == "response.incomplete" {
				r = contractParseObject(c.API, response)
				final = true
			}
		}
	}

	return r, final && r.HasInput && r.HasOutput
}

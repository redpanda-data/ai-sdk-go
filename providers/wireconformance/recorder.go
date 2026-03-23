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

package wireconformance

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"sync"
)

// CapturedExchange holds a single HTTP request captured by the recording transport.
type CapturedExchange struct {
	Method      string
	URL         string
	Path        string
	RequestBody json.RawMessage
	Headers     http.Header
}

// RecordingTransport is an http.RoundTripper that captures outgoing request
// bodies and returns a canned response. No real network calls are made.
type RecordingTransport struct {
	// CannedStatusCode is returned for every request. Defaults to 200.
	CannedStatusCode int
	// CannedBody is the response body returned for every request.
	CannedBody []byte
	// CannedHeaders are merged into every response.
	CannedHeaders http.Header

	mu       sync.Mutex
	captured []CapturedExchange
}

// RoundTrip implements http.RoundTripper. It records the request and returns
// the canned response without touching the network.
func (rt *RecordingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var body []byte
	if req.Body != nil {
		var err error
		body, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		req.Body.Close()
	}

	exchange := CapturedExchange{
		Method:      req.Method,
		URL:         req.URL.String(),
		Path:        req.URL.Path,
		RequestBody: json.RawMessage(body),
		Headers:     req.Header.Clone(),
	}

	rt.mu.Lock()
	rt.captured = append(rt.captured, exchange)
	rt.mu.Unlock()

	statusCode := rt.CannedStatusCode
	if statusCode == 0 {
		statusCode = http.StatusOK
	}

	respHeaders := make(http.Header)
	respHeaders.Set("Content-Type", "application/json")
	for k, vs := range rt.CannedHeaders {
		for _, v := range vs {
			respHeaders.Add(k, v)
		}
	}

	cannedBody := rt.CannedBody
	if cannedBody == nil {
		cannedBody = []byte(`{}`)
	}

	return &http.Response{
		StatusCode: statusCode,
		Header:     respHeaders,
		Body:       io.NopCloser(bytes.NewReader(cannedBody)),
	}, nil
}

// Captured returns a copy of all captured exchanges.
func (rt *RecordingTransport) Captured() []CapturedExchange {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	out := make([]CapturedExchange, len(rt.captured))
	copy(out, rt.captured)
	return out
}

// Reset clears all captured exchanges.
func (rt *RecordingTransport) Reset() {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	rt.captured = nil
}

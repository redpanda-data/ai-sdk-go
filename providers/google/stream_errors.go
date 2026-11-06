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

package google

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/shouldretry"
)

// wrapHTTPClient returns a copy of c for genai to use. The copy reads
// x-should-retry, and rewrites stream error events so that genai reports them
// as errors. It doesn't change c.
func wrapHTTPClient(c *http.Client) *http.Client {
	wrapped := *c
	wrapped.Transport = streamErrorTransport{base: shouldretry.Transport{Base: c.Transport}}

	return &wrapped
}

// genai reports a stream event that is a bare {"error": ...} object as an
// APIError. It reads the same object behind "data:" as a response with no
// candidates, and drops the error. AI Gateway sends a Gemini stream error
// as a data: event, and Google's Python and JS SDKs report that event as an
// error. So streamErrorTransport removes the "data:" prefix from any event
// whose JSON has a top-level "error".
type streamErrorTransport struct {
	base http.RoundTripper
}

func (t streamErrorTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := t.base.RoundTrip(req)
	if err != nil || !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		return resp, err
	}

	resp.Body = &errorEventReader{lines: bufio.NewReader(resp.Body), body: resp.Body}

	return resp, nil
}

// errorEventReader passes a stream through line by line and rewrites its
// error events.
type errorEventReader struct {
	lines   *bufio.Reader
	body    io.Closer
	pending []byte
	err     error
}

func (r *errorEventReader) Read(p []byte) (int, error) {
	for len(r.pending) == 0 {
		if r.err != nil {
			return 0, r.err
		}

		line, err := r.lines.ReadBytes('\n')
		r.err = err
		r.pending = unwrapErrorEvent(line)
	}

	n := copy(p, r.pending)
	r.pending = r.pending[n:]

	return n, nil
}

func (r *errorEventReader) Close() error {
	return r.body.Close()
}

// unwrapErrorEvent returns line without its "data:" prefix when the data is a
// JSON object with a top-level "error". It returns any other line unchanged.
func unwrapErrorEvent(line []byte) []byte {
	data, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok || !bytes.Contains(data, []byte(`"error"`)) {
		return line
	}

	data = bytes.TrimSpace(data)

	var event struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(data, &event) != nil || len(event.Error) == 0 || string(event.Error) == "null" {
		return line
	}

	return append(data, '\n')
}

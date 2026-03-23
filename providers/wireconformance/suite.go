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

// Package wireconformance provides infrastructure for wire-level conformance
// testing of AI SDK providers. It captures HTTP requests made by both the
// native provider SDK and the ai-sdk-go provider, then structurally diffs
// them to detect silent field drops, wrong mappings, or missing headers.
//
// Tests using this package do not make real API calls. A recording HTTP
// transport returns canned responses, making tests free, fast, and
// deterministic.
package wireconformance

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

// WireScenario defines a single wire conformance test case.
type WireScenario struct {
	// Name identifies the scenario in test output.
	Name string

	// NativeCall uses the native provider SDK (configured with the recording
	// transport) to make an API call. The call may return an error from
	// response parsing (since the canned response is minimal) -- that's fine,
	// we only care about the captured request.
	NativeCall func(t *testing.T, transport *RecordingTransport)

	// SDKCall uses ai-sdk-go (configured with the recording transport) to
	// make the equivalent API call.
	SDKCall func(t *testing.T, transport *RecordingTransport)

	// IgnorePaths are JSON paths to ignore when diffing request bodies.
	// Use dotted paths for nested fields, "field.*" for wildcard.
	IgnorePaths []string

	// IgnoreHeaders are header names to skip when comparing headers.
	IgnoreHeaders []string

	// FixHint tells the agent where to look to fix diffs.
	FixHint string
}

// RunScenario executes a single wire scenario: captures both requests, diffs them.
func RunScenario(t *testing.T, scenario WireScenario, cannedResponse []byte) {
	t.Helper()
	t.Run(scenario.Name, func(t *testing.T) {
		// Set up two independent recording transports with the same canned response.
		nativeTransport := &RecordingTransport{
			CannedStatusCode: http.StatusOK,
			CannedBody:       cannedResponse,
		}
		sdkTransport := &RecordingTransport{
			CannedStatusCode: http.StatusOK,
			CannedBody:       cannedResponse,
		}

		// Run both calls.
		scenario.NativeCall(t, nativeTransport)
		scenario.SDKCall(t, sdkTransport)

		// Get captured exchanges.
		nativeCaptured := nativeTransport.Captured()
		sdkCaptured := sdkTransport.Captured()

		require.NotEmpty(t, nativeCaptured, "native SDK made no HTTP requests")
		require.NotEmpty(t, sdkCaptured, "ai-sdk-go made no HTTP requests")
		require.Equal(t, len(nativeCaptured), len(sdkCaptured),
			"native SDK made %d requests, ai-sdk-go made %d", len(nativeCaptured), len(sdkCaptured))

		// Build ignore set.
		ignores := make(map[string]bool)
		for _, p := range scenario.IgnorePaths {
			ignores[p] = true
		}

		// Diff each request pair.
		for i := range nativeCaptured {
			native := nativeCaptured[i]
			sdk := sdkCaptured[i]

			// Compare request bodies.
			diffs := DiffJSON(native.RequestBody, sdk.RequestBody, ignores)
			if len(diffs) > 0 {
				report := FormatDiffs(scenario.Name, diffs, scenario.FixHint)
				t.Errorf("wire conformance request body mismatch:\n%s", report)
			}

			// Compare headers (optional).
			headerIgnores := map[string]bool{
				"User-Agent":    true,
				"Authorization": true,
				"Content-Type":  true,
				"Content-Length": true,
				"Accept":        true,
				"Accept-Encoding": true,
			}
			for _, h := range scenario.IgnoreHeaders {
				headerIgnores[http.CanonicalHeaderKey(h)] = true
			}

			headerDiffs := diffHeaders(native.Headers, sdk.Headers, headerIgnores)
			if len(headerDiffs) > 0 {
				report := FormatDiffs(scenario.Name+" [headers]", headerDiffs, scenario.FixHint)
				t.Errorf("wire conformance header mismatch:\n%s", report)
			}
		}
	})
}

// diffHeaders compares two sets of HTTP headers, ignoring specified header names.
func diffHeaders(native, sdk http.Header, ignores map[string]bool) []FieldDiff {
	var diffs []FieldDiff

	allKeys := make(map[string]bool)
	for k := range native {
		allKeys[http.CanonicalHeaderKey(k)] = true
	}
	for k := range sdk {
		allKeys[http.CanonicalHeaderKey(k)] = true
	}

	for key := range allKeys {
		if ignores[key] {
			continue
		}

		nVals := native.Values(key)
		sVals := sdk.Values(key)

		nStr := headerValStr(nVals)
		sStr := headerValStr(sVals)

		if nStr == sStr {
			continue
		}

		diff := FieldDiff{Path: "Header:" + key}
		switch {
		case len(nVals) > 0 && len(sVals) == 0:
			diff.Kind = DiffMissing
			diff.Expected = []byte(`"` + nStr + `"`)
		case len(nVals) == 0 && len(sVals) > 0:
			diff.Kind = DiffExtra
			diff.Actual = []byte(`"` + sStr + `"`)
		default:
			diff.Kind = DiffChanged
			diff.Expected = []byte(`"` + nStr + `"`)
			diff.Actual = []byte(`"` + sStr + `"`)
		}

		diffs = append(diffs, diff)
	}

	return diffs
}

func headerValStr(vals []string) string {
	if len(vals) == 0 {
		return ""
	}
	if len(vals) == 1 {
		return vals[0]
	}
	result := ""
	for i, v := range vals {
		if i > 0 {
			result += ", "
		}
		result += v
	}
	return result
}

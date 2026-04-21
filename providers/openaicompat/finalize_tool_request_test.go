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

package openaicompat

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestFinalizeToolRequest covers the same class of wedge as the anthropic
// upstream fix: tool_calls deltas that stopped accumulating mid-JSON (e.g.
// finish_reason=length on a call that had only sent `{"q":`) must not escape
// the provider with truncated bytes.
func TestFinalizeToolRequest(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name     string
		in       json.RawMessage
		wantOK   bool
		wantArgs string
	}{
		{"empty coerces to {}", nil, true, "{}"},
		{"zero-length coerces to {}", json.RawMessage(""), true, "{}"},
		{"valid passes through", json.RawMessage(`{"q":"SELECT 1"}`), true, `{"q":"SELECT 1"}`},
		{"truncated is dropped", json.RawMessage(`{"q":`), false, `{"q":`},
		{"garbage is dropped", json.RawMessage(`not json`), false, `not json`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			tr := &llm.ToolRequest{
				ID:        "call_1",
				Name:      "query",
				Arguments: tc.in,
			}

			ok := finalizeToolRequest(tr)
			assert.Equal(t, tc.wantOK, ok)
			assert.Equal(t, tc.wantArgs, string(tr.Arguments))
		})
	}
}

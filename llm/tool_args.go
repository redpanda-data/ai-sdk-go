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

package llm

import "encoding/json"

// FinalizeToolArgs validates a tool-argument buffer accumulated from a
// streaming provider's tool-use delta events. It's the shared guard every
// provider that accumulates tool_use input byte-by-byte (anthropic's
// input_json_delta, bedrock's ContentBlockDelta toolUse input,
// openaicompat's tool_calls[].function.arguments) runs at block
// finalization:
//
//   - empty accumulation coerces to `{}` — the wire form these providers
//     send for no-arg tool calls, so coercion is safe;
//   - invalid JSON signals that the stream was cut short mid-accumulation
//     (typically stop_reason=max_tokens during parallel tool use) and
//     returns ok=false so the caller can drop the block.
//
// Without this guard, truncated bytes like `{"query":` reach the agent as
// ToolRequest.Arguments, get persisted to session state, and wedge every
// subsequent replay at json.Unmarshal with "unexpected end of JSON input".
// Provider stream finalizers must call this and propagate the false result
// as a block drop.
func FinalizeToolArgs(args []byte) (json.RawMessage, bool) {
	if len(args) == 0 {
		return json.RawMessage("{}"), true
	}

	if !json.Valid(args) {
		return nil, false
	}

	return json.RawMessage(args), true
}

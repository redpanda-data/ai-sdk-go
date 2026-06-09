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

package tool

import (
	"context"
	"encoding/json"
)

// Tool is the contract every executable tool implements. The 90% path —
// simple synchronous functions — should be reached through tool.Func +
// tool.Done; tools that need pause/resume reach for Call.Resume and the
// Execution return type directly.
//
// Concrete implementations should:
//
//   - Return a stable Name and a model-visible Description.
//   - Return a JSON Schema for arguments via InputSchema (use tool.Spec
//   - tool.Func if you don't want to write the schema by hand).
//   - Treat call.Resume == nil as "first entry" and non-nil as "the
//     runtime re-entered after a previous Await with ResumeWithReentry".
//   - Return tool.Execution with an Output payload that the model can
//     parse. To pause, attach an Await — see tool/await.go for the
//     accepted Reason/Resume pairs.
//
// Failures reach the model through one of two channels:
//
//   - Return a non-nil error: the runtime encodes the message as a tool
//     error response ({"error": ...} with is_error set) and ignores the
//     returned Execution. Use this for plain failures.
//   - Return a structured failure payload in Execution.Output with a nil
//     error: the response is delivered as a regular (non-error) result
//     whose JSON describes the failure. Use this when the model should
//     read structured failure detail (status codes, retry hints) instead
//     of a bare message.
//
// Pause is not an error: attach an Await and return a nil error.
type Tool interface {
	// Name is the model-visible tool name. Stable across invocations.
	Name() string

	// Description is the model-visible description. Async-capable tools
	// should leave async-related guidance to tool.Spec + the registry —
	// don't re-state it manually.
	Description() string

	// InputSchema returns the JSON Schema describing call arguments.
	// May be nil for tools with no arguments.
	InputSchema() json.RawMessage

	// Execute runs the tool with the given Call. Errors propagate
	// model-visible failures; returning a non-nil Execution with a
	// non-nil Await pauses the invocation instead.
	Execute(ctx context.Context, call Call) (Execution, error)
}

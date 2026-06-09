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

// Result is the typed analogue of Execution used by tool.Func authors:
// instead of constructing JSON by hand, return a Result[Out] with a
// typed Value. The registry marshals Value to Execution.Output.
//
// Use Done for synchronous completion, Pending for long-running external
// work, and NeedInput for conversational pauses.
type Result[T any] struct {
	// Value is the typed model-visible result. The registry marshals it
	// to JSON when reconciling into an Execution.
	Value T

	// Await is non-nil when the call has not reached its terminal
	// result. Build it via Pending / NeedInput or by hand.
	Await *Await

	// Actions carry side effects to the runner (artifact emission, etc.).
	Actions []Action
}

// Done returns a Result containing a final terminal value. Equivalent to
// `Result[T]{Value: v, Actions: actions}`.
func Done[T any](v T, actions ...Action) Result[T] {
	return Result[T]{Value: v, Actions: actions}
}

// Pending returns a Result that pauses execution awaiting an external
// result (Reason=tool_result, Resume=tool_response). The provided v is
// the placeholder Output sent to the model while the runtime waits for
// the caller to submit the final result.
//
// Apply AwaitOption helpers to set the user-visible message, correlation
// ID, optional timeout, or opaque tool-private state.
func Pending[T any](v T, opts ...AwaitOption) Result[T] {
	a := &Await{Reason: AwaitReasonToolResult, Resume: ResumeWithToolResponse}
	for _, opt := range opts {
		opt(a)
	}

	return Result[T]{Value: v, Await: a}
}

// NeedInput returns a Result that pauses for conversational user input
// (Reason=user_input, Resume=message). The message argument is the UI
// prompt the runner surfaces; it is also set on the Await so adapters
// such as A2A can read it from PendingCallSummary. The placeholder v is
// what travels to the model in the meantime.
func NeedInput[T any](v T, message string, opts ...AwaitOption) Result[T] {
	a := &Await{
		Reason:  AwaitReasonUserInput,
		Resume:  ResumeWithMessage,
		Message: message,
	}

	for _, opt := range opts {
		opt(a)
	}

	return Result[T]{Value: v, Await: a}
}

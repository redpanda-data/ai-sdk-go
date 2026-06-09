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

// Package tool provides a runtime for registering and executing tools
// that LLM agents can call, including pause/resume for long-running and
// human-in-the-loop tools.
//
// # Defining tools
//
// The 90% path is tool.Func: a typed function with an inferred JSON
// schema. Return tool.Done for synchronous results:
//
//	type AddInput struct {
//		A int `json:"a"`
//		B int `json:"b"`
//	}
//
//	type AddOutput struct {
//		Sum int `json:"sum"`
//	}
//
//	var addTool = tool.Must(tool.Func(
//		tool.Spec{Name: "add", Description: "Add two integers."},
//		func(ctx context.Context, in AddInput) (tool.Result[AddOutput], error) {
//			return tool.Done(AddOutput{Sum: in.A + in.B}), nil
//		},
//	))
//
// Implement the Tool interface directly when you need full control —
// custom re-entry logic, dynamic schemas, wrapping external systems.
// Implement SpecProvider as well so Definition() and the registry see
// your structured Spec, and Unwrapper if you decorate another Tool.
//
// # Registry
//
//	registry := tool.NewRegistry()
//	_ = registry.Register(addTool, tool.WithTimeout(30*time.Second))
//
//	res := registry.Run(ctx, inv, toolRequest)
//	if res.Err != nil { /* model-visible tool error */ }
//	if res.Execution.Await != nil { /* paused; persist pending state */ }
//	responsePart := res.Response()
//
// Run returns the typed ExecutionResult so callers can distinguish
// pauses from terminal results; ExecutionResult.Response reconciles it
// into the model-visible llm.ToolResponsePart.
//
// # Pausing (async tools)
//
// A tool pauses by returning a Result with a non-nil Await — built via
// tool.Pending (external job, resumed with the final output) or
// tool.NeedInput (conversational pause). Tools that need custom
// re-entry logic implement Tool directly and read Call.Resume. Declare the pause behavior
// on Spec.Async (AsyncExternalResult, AsyncApproval, ...) so the model
// is told not to re-call the tool while a call is pending and the
// registry validates emitted pauses.
//
// The agent loop persists pauses as session.PendingToolCall records;
// runner.Resume delivers results, runner.Progress records non-terminal
// updates, and runner.Cancel aborts. See the runner package.
package tool

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
	"encoding/json"
	"maps"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Spec declares everything the registry needs to know about a tool at
// registration time. It is consumed by tool.Func and any other Tool
// implementation that wants the Spec-based helpers (Definition, async
// description hint, metadata round-trip).
type Spec struct {
	// Name is the model-visible tool name. Must be non-empty.
	Name string

	// Description is the model-visible description. AsyncSpec.Hint, if
	// set, is appended verbatim.
	Description string

	// InputSchema is the JSON Schema for tool arguments. Optional for
	// typed Func; required for raw Tool implementations that do not infer
	// a schema themselves.
	InputSchema json.RawMessage

	// OutputSchema is an optional JSON Schema for the tool's output.
	// Currently advisory; included in Definition.Metadata as
	// "output_schema".
	OutputSchema json.RawMessage

	// Type is the provider-facing tool kind (e.g. "function",
	// "extension"). Empty defaults to "function" at definition time.
	Type string

	// Async, when non-nil, declares the tool as async-capable. The
	// runtime uses this for LLM-facing guidance ("don't call again while
	// pending") and to validate pause modes the tool emits.
	Async *AsyncSpec

	// Metadata is round-tripped to llm.ToolDefinition.Metadata for
	// provider-side annotations. It is not sent as part of the
	// description prompt unless a provider chooses to.
	Metadata map[string]any
}

// AsyncSpec declares that a tool may pause via Execution.Await. It does
// not commit the tool to pause every call: tools can mix sync and async
// returns based on input. The registry uses AsyncSpec to:
//
//   - Validate that any returned Await uses the declared Reason/Resume.
//   - Append Hint to the model-visible description so the model knows
//     not to re-call the same tool while the previous call is pending.
type AsyncSpec struct {
	// Reason classifies the expected pause. The actual returned Await
	// must use this Reason (or a compatible one when the tool dispatches
	// dynamically — currently the runtime checks Reason equality).
	Reason AwaitReason

	// Resume declares the expected resume mode. Same equality rule as
	// Reason.
	Resume ResumeMode

	// Hint is appended to the model-visible description. When empty, a
	// reason-appropriate default is used.
	Hint string
}

// AsyncOption configures an AsyncSpec built by the factory helpers.
type AsyncOption func(*AsyncSpec)

// WithAsyncHint overrides the default model-visible hint.
func WithAsyncHint(hint string) AsyncOption {
	return func(s *AsyncSpec) { s.Hint = hint }
}

// AsyncExternalResult returns an AsyncSpec for a long-running external
// job that resumes when the caller submits the final tool response.
func AsyncExternalResult(opts ...AsyncOption) *AsyncSpec {
	return newAsyncSpec(AwaitReasonToolResult, ResumeWithToolResponse, opts...)
}

// AsyncUserInput returns an AsyncSpec for a conversational pause that
// resumes when the user sends their next message.
func AsyncUserInput(opts ...AsyncOption) *AsyncSpec {
	return newAsyncSpec(AwaitReasonUserInput, ResumeWithMessage, opts...)
}

// AsyncApproval returns an AsyncSpec for an approval gate that resumes
// by re-entering the tool/interceptor with the approval decision.
func AsyncApproval(opts ...AsyncOption) *AsyncSpec {
	return newAsyncSpec(AwaitReasonApproval, ResumeWithReentry, opts...)
}

// AsyncElicitation returns an AsyncSpec for an MCP elicitation pause.
func AsyncElicitation(opts ...AsyncOption) *AsyncSpec {
	return newAsyncSpec(AwaitReasonElicitation, ResumeWithReentry, opts...)
}

// AsyncHandoff returns an AsyncSpec for an agent-as-tool handoff pause.
func AsyncHandoff(opts ...AsyncOption) *AsyncSpec {
	return newAsyncSpec(AwaitReasonHandoff, ResumeWithReentry, opts...)
}

func newAsyncSpec(reason AwaitReason, resume ResumeMode, opts ...AsyncOption) *AsyncSpec {
	s := &AsyncSpec{Reason: reason, Resume: resume}
	for _, opt := range opts {
		opt(s)
	}

	return s
}

// defaultAsyncHint returns a model-visible hint for a given reason. The
// hint is appended to the tool description so the model understands the
// pause semantics.
func defaultAsyncHint(reason AwaitReason) string {
	switch reason {
	case AwaitReasonToolResult:
		return "\n\nThis tool may pause while external work runs. Do not call it again for the same operation; the SDK will provide the result with the original call ID when ready."
	case AwaitReasonUserInput:
		return "\n\nThis tool may pause for user input. The user's next message is the answer; do not call this tool again unless you need to ask for something different."
	case AwaitReasonApproval:
		return "\n\nThis tool may pause for human approval. The SDK will continue the same call ID once a decision is provided."
	case AwaitReasonElicitation:
		return "\n\nThis tool may pause for additional input from the connected MCP server. The SDK resumes the call once the elicited value is provided."
	case AwaitReasonHandoff:
		return "\n\nThis tool may pause while a delegated sub-agent waits on input. The SDK resumes the call once the sub-agent has a result."
	default:
		return ""
	}
}

// Definition builds the provider-facing llm.ToolDefinition for a Tool.
// It honors Spec.Description, appends AsyncSpec.Hint (or a default), and
// folds Spec.Metadata into the result.
//
// Free-function rather than a method on Tool so that helpers built on
// top of Spec (like Func) own description shaping in one place; raw
// Tool implementations get a sensible default by virtue of implementing
// Name/Description/InputSchema.
func Definition(t Tool) llm.ToolDefinition {
	if t == nil {
		return llm.ToolDefinition{}
	}

	desc := t.Description()
	typeName := "function"

	var meta map[string]any

	if sp, ok := t.(specProvider); ok {
		desc, typeName, meta = applySpecToDefinition(desc, sp.toolSpec())
	}

	return llm.ToolDefinition{
		Name:        t.Name(),
		Description: desc,
		Parameters:  t.InputSchema(),
		Metadata:    meta,
		Type:        typeName,
	}
}

// applySpecToDefinition layers the spec's async hint, type override,
// metadata, and output schema onto the base description/type/meta
// triple. Pulled out of Definition to keep cyclomatic complexity flat
// when the spec carries multiple optional pieces.
func applySpecToDefinition(desc string, spec Spec) (string, string, map[string]any) {
	typeName := "function"
	if spec.Type != "" {
		typeName = spec.Type
	}

	desc = appendAsyncHint(desc, spec.Async)

	var meta map[string]any
	if spec.Metadata != nil {
		meta = maps.Clone(spec.Metadata)
	}

	if spec.OutputSchema != nil {
		if meta == nil {
			meta = make(map[string]any, 1)
		}

		meta["output_schema"] = spec.OutputSchema
	}

	return desc, typeName, meta
}

// appendAsyncHint returns desc with the configured async hint appended.
// If the hint is empty, the default for the reason is used; if the hint
// is already present, desc is returned unchanged.
func appendAsyncHint(desc string, async *AsyncSpec) string {
	if async == nil {
		return desc
	}

	hint := async.Hint
	if hint == "" {
		hint = defaultAsyncHint(async.Reason)
	}

	if hint == "" || strings.Contains(desc, hint) {
		return desc
	}

	return desc + hint
}

// specProvider is implemented by Tool wrappers that want Definition() to
// honor their Spec metadata. Helpers in this package such as Func
// implement it via an unexported method so consumers cannot fake out
// the registry by lying about their Spec.
type specProvider interface {
	toolSpec() Spec
}

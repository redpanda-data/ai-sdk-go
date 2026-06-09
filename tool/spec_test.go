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

package tool_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

type echoInput struct {
	Text string `json:"text"`
}

type echoOutput struct {
	Text string `json:"text"`
}

func newAsyncEchoTool(t *testing.T) tool.Tool {
	t.Helper()

	tl, err := tool.Func(
		tool.Spec{
			Name:        "echo_async",
			Description: "Echoes text via an external job.",
			Async:       tool.AsyncExternalResult(),
			Metadata:    map[string]any{"risk": "low"},
		},
		func(_ context.Context, in echoInput) (tool.Result[echoOutput], error) {
			return tool.Pending(echoOutput(in)), nil
		},
	)
	require.NoError(t, err)

	return tl
}

// loggingWrapper is a minimal Tool decorator. It forwards the Tool
// surface and exposes the wrapped tool via Unwrap so SpecOf and
// Definition keep working through it.
type loggingWrapper struct {
	inner tool.Tool
	calls int
}

func (w *loggingWrapper) Name() string                 { return w.inner.Name() }
func (w *loggingWrapper) Description() string          { return w.inner.Description() }
func (w *loggingWrapper) InputSchema() json.RawMessage { return w.inner.InputSchema() }
func (w *loggingWrapper) Unwrap() tool.Tool            { return w.inner }

func (w *loggingWrapper) Execute(ctx context.Context, call tool.Call) (tool.Execution, error) {
	w.calls++
	return w.inner.Execute(ctx, call)
}

func TestSpecOf_FollowsUnwrapChain(t *testing.T) {
	t.Parallel()

	inner := newAsyncEchoTool(t)
	wrapped := &loggingWrapper{inner: inner}
	doubleWrapped := &loggingWrapper{inner: wrapped}

	spec, ok := tool.SpecOf(doubleWrapped)
	require.True(t, ok, "SpecOf must follow Unwrap chains")
	assert.Equal(t, "echo_async", spec.Name)
	require.NotNil(t, spec.Async)
	assert.Equal(t, tool.AwaitReasonToolResult, spec.Async.Reason)
}

func TestDefinition_PreservedThroughWrapper(t *testing.T) {
	t.Parallel()

	inner := newAsyncEchoTool(t)
	wrapped := &loggingWrapper{inner: inner}

	direct := tool.Definition(inner)
	viaWrapper := tool.Definition(wrapped)

	assert.Equal(t, direct.Description, viaWrapper.Description,
		"wrapping must not strip the async hint from the model-visible description")
	assert.Equal(t, direct.Metadata, viaWrapper.Metadata)
	assert.Equal(t, direct.Type, viaWrapper.Type)
}

// rawMismatchedTool implements Tool + SpecProvider directly and emits a
// pause that contradicts its declared AsyncSpec.
type rawMismatchedTool struct{}

func (rawMismatchedTool) Name() string                 { return "mismatched" }
func (rawMismatchedTool) Description() string          { return "Declares approval, pauses external." }
func (rawMismatchedTool) InputSchema() json.RawMessage { return nil }
func (rawMismatchedTool) ToolSpec() tool.Spec {
	return tool.Spec{Name: "mismatched", Async: tool.AsyncApproval()}
}

func (rawMismatchedTool) Execute(context.Context, tool.Call) (tool.Execution, error) {
	return tool.Execution{
		Output: json.RawMessage(`{}`),
		Await:  &tool.Await{Reason: tool.AwaitReasonToolResult, Resume: tool.ResumeWithToolResponse},
	}, nil
}

func TestRegistry_EnforcesAsyncSpecForRawTools(t *testing.T) {
	t.Parallel()

	reg := tool.NewRegistry()
	require.NoError(t, reg.Register(rawMismatchedTool{}))

	res := reg.Run(context.Background(), tool.InvocationInfo{},
		&llm.ToolRequestPart{ID: "c1", Name: "mismatched"})

	require.ErrorIs(t, res.Err, tool.ErrAwaitInvalid)
	assert.Contains(t, res.Err.Error(), "does not match declared")
}

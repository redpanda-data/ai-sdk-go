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

package llmagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	nativeCompatParams = `{"type":"object","properties":{"q":{"type":"string"}}}`
	nativeCompatAlpha  = "alpha"
)

type nativeCompatBlockingTool struct {
	def     llm.ToolDefinition
	result  json.RawMessage
	started chan struct{}
	release <-chan struct{}
	calls   atomic.Int32
}

func (b *nativeCompatBlockingTool) Definition() llm.ToolDefinition { return b.def }

func (b *nativeCompatBlockingTool) Execute(ctx context.Context, _ json.RawMessage) (json.RawMessage, error) {
	b.calls.Add(1)

	select {
	case b.started <- struct{}{}:
	default:
	}

	select {
	case <-b.release:
		return b.result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

type nativeCompatModelInterceptor func(*agent.ModelCallInfo)

func (f nativeCompatModelInterceptor) InterceptModel(_ context.Context, info *agent.ModelCallInfo, next agent.ModelCallHandler) agent.ModelCallHandler {
	f(info)

	return next
}

type nativeCompatView struct {
	Description string
	Parameters  string
	Metadata    string
}

func nativeCompatMetadata() map[string]any {
	return map[string]any{
		"policy": map[string]any{"scope": "pristine"},
		"tags":   []string{"pristine"},
		"items":  []any{map[string]any{"scope": "pristine"}},
	}
}

func nativeCompatDefinition(name, description string) llm.ToolDefinition {
	return llm.ToolDefinition{Name: name, Description: description, Parameters: json.RawMessage(nativeCompatParams), Metadata: nativeCompatMetadata()}
}

func nativeCompatPristine(t *testing.T, description string) nativeCompatView {
	t.Helper()

	return nativeCompatCapture(t, nativeCompatDefinition("", description))
}

func nativeCompatCapture(t *testing.T, def llm.ToolDefinition) nativeCompatView {
	t.Helper()

	metadata, err := json.Marshal(def.Metadata)
	if err != nil {
		t.Errorf("marshal metadata: %v", err)
	}

	return nativeCompatView{Description: def.Description, Parameters: string(def.Parameters), Metadata: string(metadata)}
}

func nativeCompatCaptureTool(t *testing.T, defs []llm.ToolDefinition, name string) nativeCompatView {
	t.Helper()

	for _, def := range defs {
		if def.Name == name {
			return nativeCompatCapture(t, def)
		}
	}

	t.Errorf("tool %q missing from %v", name, defNames(defs))

	return nativeCompatView{}
}

// nativeCompatMutate edits a definition in place at every depth a caller can
// reach: scalar fields, schema bytes, and top-level and nested metadata.
func nativeCompatMutate(t *testing.T, def *llm.ToolDefinition, by string) {
	t.Helper()

	def.Description = by
	for i := range def.Parameters {
		def.Parameters[i] = 'x'
	}

	def.Metadata["added"] = by

	if policy, ok := def.Metadata["policy"].(map[string]any); assert.True(t, ok) {
		policy["scope"] = by
	}

	if tags, ok := def.Metadata["tags"].([]string); assert.True(t, ok) {
		tags[0] = by
	}

	if items, ok := def.Metadata["items"].([]any); assert.True(t, ok) {
		if item, ok := items[0].(map[string]any); assert.True(t, ok) {
			item["scope"] = by
		}
	}
}

func nativeCompatMutateFetch(t *testing.T, defs []llm.ToolDefinition, by string) {
	t.Helper()

	for i := range defs {
		if defs[i].Name == "fetch" {
			nativeCompatMutate(t, &defs[i], by)

			return
		}
	}

	t.Errorf("tool fetch missing from %v", defNames(defs))
}

func nativeCompatSourceView(t *testing.T, source tool.Registry, name string) nativeCompatView {
	t.Helper()

	frozen, err := tool.Snapshot(t.Context(), source)
	require.NoError(t, err)

	defs, err := frozen.List(t.Context())
	require.NoError(t, err)

	return nativeCompatCaptureTool(t, defs, name)
}

func nativeCompatText(text string) *llm.Response {
	return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(text)), FinishReason: llm.FinishReasonStop}
}

func nativeCompatConversation(req *llm.Request) string {
	for _, msg := range req.Messages {
		if msg.Role == llm.RoleUser {
			return msg.TextContent()
		}
	}

	return ""
}

func nativeCompatAnswered(req *llm.Request) int {
	var answered int
	for _, msg := range req.Messages {
		answered += len(msg.ToolResponses())
	}

	return answered
}

func nativeCompatLastAnswered(req *llm.Request) bool {
	return len(req.Messages) > 0 && len(req.Messages[len(req.Messages)-1].ToolResponses()) > 0
}

func nativeCompatCalls(model *fakellm.FakeModel, conversation string) []fakellm.Call {
	var calls []fakellm.Call

	for _, call := range model.CallsMatching(fakellm.Any()) {
		if nativeCompatConversation(call.Request) == conversation {
			calls = append(calls, call)
		}
	}

	return calls
}

func nativeCompatToolResponses(events []agent.Event) []agent.ToolResponseEvent {
	var out []agent.ToolResponseEvent

	for _, event := range events {
		if response, ok := event.(agent.ToolResponseEvent); ok {
			out = append(out, response)
		}
	}

	return out
}

func nativeCompatOverflow() error {
	return fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{Base: llm.ErrContextOverflow, Code: "invalid_request_error", Message: "prompt is too long"})
}

// One batch is planned against one native snapshot. While its first call is
// still running, the source replaces two tools (one with a different response
// policy) and removes the third, and a second invocation of the same agent
// resolves the changed catalog. The batch's remaining calls still dispatch to
// the captured executables and policies; the batch's next model step and the
// concurrent invocation see only the current registrations.
func TestToolsNativeCompatBatchKeepsCapturedToolsWhileSourceChanges(t *testing.T) {
	t.Parallel()

	for _, concurrency := range []int{1, 3} {
		t.Run(fmt.Sprintf("concurrency=%d", concurrency), func(t *testing.T) {
			t.Parallel()

			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()

			mutated := make(chan struct{})
			payload := fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("second payload ", 20))
			first := &nativeCompatBlockingTool{
				def: llm.ToolDefinition{Name: "first", Description: "original first", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"source":"original first"}`),
				started: make(chan struct{}, 1), release: mutated,
			}
			second := &stubTool{def: llm.ToolDefinition{Name: "second", Description: "original second", Parameters: json.RawMessage(`{"type":"object"}`)}, result: payload}
			third := &stubTool{def: llm.ToolDefinition{Name: "third", Description: "original third", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"source":"original third"}`)}
			firstNext := &stubTool{def: llm.ToolDefinition{Name: "first", Description: "replacement first", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"source":"replacement first"}`)}
			secondNext := &stubTool{def: llm.ToolDefinition{Name: "second", Description: "replacement second", Parameters: json.RawMessage(`{"type":"object"}`)}, result: payload}

			source := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, source.Register(first))
			require.NoError(t, source.Register(second, tool.WithMaxResponseTokens(5), tool.WithResponseTooLargeMessage("Captured second policy.")))
			require.NoError(t, source.Register(third))

			model := fakellm.NewFakeModel()
			model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
				if nativeCompatLastAnswered(req) {
					return nativeCompatText("done"), nil
				}

				id := nativeCompatConversation(req)

				return respondWith(toolCall(id+"-first", "first", `{}`), toolCall(id+"-second", "second", `{}`), toolCall(id+"-third", "third", `{}`))(req, cc)
			})

			// Holds alpha's later calls until the source has changed, so with
			// any concurrency they dispatch through the batch's snapshot after
			// the mutation rather than racing it.
			gate := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				if info.Inv.Session().ID == nativeCompatAlpha && info.Req.Name != "first" {
					select {
					case <-mutated:
					case <-ctx.Done():
						return nil, ctx.Err()
					}
				}

				return next(ctx, info)
			})
			ag, err := New("native-batch", "prompt", model, WithTools(source), WithToolConcurrency(concurrency), WithInterceptors(gate))
			require.NoError(t, err)

			alpha := &session.State{ID: nativeCompatAlpha, Messages: userMessage(nativeCompatAlpha)}
			alphaDone := make(chan error, 1)

			go func() {
				alphaDone <- toolsRunError(ctx, ag, agent.NewInvocationMetadata(alpha, ag.Info()))
			}()

			select {
			case <-first.started:
			case <-ctx.Done():
				require.FailNow(t, "alpha's first call never started")
			}

			require.NoError(t, source.Unregister("first"))
			require.NoError(t, source.Register(firstNext))
			require.NoError(t, source.Unregister("second"))
			require.NoError(t, source.Register(secondNext))
			require.NoError(t, source.Unregister("third"))

			beta := &session.State{ID: "beta", Messages: userMessage("beta")}
			require.NoError(t, toolsRunError(ctx, ag, agent.NewInvocationMetadata(beta, ag.Info())))
			close(mutated)

			select {
			case err := <-alphaDone:
				require.NoError(t, err)
			case <-ctx.Done():
				require.FailNow(t, "alpha did not finish")
			}

			assert.Equal(t, int32(1), first.calls.Load())
			assert.Equal(t, int32(1), second.calls.Load())
			assert.Equal(t, int32(1), third.calls.Load(), "a removed tool already in the batch snapshot still runs")
			assert.Equal(t, int32(1), firstNext.calls.Load())
			assert.Equal(t, int32(1), secondNext.calls.Load())

			require.Len(t, toolsResults(alpha, "first"), 1)
			assert.JSONEq(t, `{"source":"original first"}`, string(toolsResults(alpha, "first")[0].Result))
			require.Len(t, toolsResults(alpha, "second"), 1)
			assert.Contains(t, string(toolsResults(alpha, "second")[0].Result), "Captured second policy.", "the captured response policy must apply")
			assert.NotContains(t, string(toolsResults(alpha, "second")[0].Result), "second payload")
			require.Len(t, toolsResults(alpha, "third"), 1)
			assert.False(t, toolsResults(alpha, "third")[0].IsError)
			assert.JSONEq(t, `{"source":"original third"}`, string(toolsResults(alpha, "third")[0].Result))

			require.Len(t, toolsResults(beta, "first"), 1)
			assert.JSONEq(t, `{"source":"replacement first"}`, string(toolsResults(beta, "first")[0].Result))
			require.Len(t, toolsResults(beta, "second"), 1)
			assert.False(t, toolsResults(beta, "second")[0].IsError)
			assert.JSONEq(t, string(payload), string(toolsResults(beta, "second")[0].Result), "the replacement registered without a size policy")
			require.Len(t, toolsResults(beta, "third"), 1)
			assert.True(t, toolsResults(beta, "third")[0].IsError)
			assert.Contains(t, string(toolsResults(beta, "third")[0].Result), tool.ErrToolNotFound.Error())

			alphaCalls := nativeCompatCalls(model, nativeCompatAlpha)
			require.Len(t, alphaCalls, 2)
			assert.Equal(t, []string{"first", "second", "third"}, requestToolNames(alphaCalls[0].Request))
			assert.Equal(t, "original second", defByName(t, alphaCalls[0].Request.Tools, "second").Description)
			assert.Equal(t, []string{"first", "second"}, requestToolNames(alphaCalls[1].Request))
			assert.Equal(t, "replacement first", defByName(t, alphaCalls[1].Request.Tools, "first").Description)
			assert.Equal(t, "replacement second", defByName(t, alphaCalls[1].Request.Tools, "second").Description)

			betaCalls := nativeCompatCalls(model, "beta")
			require.Len(t, betaCalls, 2)

			for _, call := range betaCalls {
				assert.Equal(t, []string{"first", "second"}, requestToolNames(call.Request))
				assert.Equal(t, "replacement first", defByName(t, call.Request.Tools, "first").Description)
			}
		})
	}
}

// A fake model, a model interceptor, and a tool interceptor each edit the
// definitions they are handed, in place and at every depth, during the first
// step. None of it may reach the native source, the next step's snapshot, or
// an independent invocation of the same agent.
func TestToolsNativeCompatStepMutationsStayInStep(t *testing.T) {
	t.Parallel()

	fetch := &stubTool{def: nativeCompatDefinition("fetch", "pristine fetch")}
	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(fetch))

	var modelViews []nativeCompatView

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
		modelViews = append(modelViews, nativeCompatCaptureTool(t, req.Tools, "fetch"))

		switch answered := nativeCompatAnswered(req); {
		case nativeCompatConversation(req) != nativeCompatAlpha || answered == 2:
			return nativeCompatText("done"), nil
		case answered == 0:
			nativeCompatMutateFetch(t, req.Tools, "model")

			return respondWith(toolCall("step-0", "fetch", `{}`))(req, cc)
		default:
			return respondWith(toolCall("step-1", "fetch", `{}`))(req, cc)
		}
	})

	var toolViews []nativeCompatView

	firstStep := func(inv *agent.InvocationMetadata) bool {
		return inv.Session().ID == nativeCompatAlpha && inv.Turn() == 0
	}
	modelEdits := nativeCompatModelInterceptor(func(info *agent.ModelCallInfo) {
		if firstStep(info.InvocationMetadata) {
			nativeCompatMutateFetch(t, info.Req.Tools, "model interceptor")
		}
	})
	toolEdits := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
		if !assert.NotNil(t, info.Definition) {
			return next(ctx, info)
		}

		toolViews = append(toolViews, nativeCompatCapture(t, *info.Definition))
		if firstStep(info.Inv) {
			nativeCompatMutate(t, info.Definition, "tool interceptor")
		}

		return next(ctx, info)
	})
	ag, err := New("native-mutations", "prompt", model, WithTools(source), WithInterceptors(modelEdits, toolEdits))
	require.NoError(t, err)

	alpha := &session.State{ID: nativeCompatAlpha, Messages: userMessage(nativeCompatAlpha)}
	runAgent(t, ag, alpha)

	beta := &session.State{ID: "beta", Messages: userMessage("beta")}
	runAgent(t, ag, beta)

	pristine := nativeCompatPristine(t, "pristine fetch")
	require.Len(t, modelViews, 4)
	assert.Equal(t, pristine, modelViews[1], "the next step must resolve a fresh snapshot")
	assert.Equal(t, pristine, modelViews[2])
	assert.Equal(t, pristine, modelViews[3], "an independent invocation must not see another run's edits")
	require.Len(t, toolViews, 2)
	assert.Equal(t, pristine, toolViews[1], "the next step's tool interceptors must see its own snapshot")
	assert.Equal(t, pristine, nativeCompatCapture(t, fetch.Definition()), "the registered tool's own definition must stay intact")
	assert.Equal(t, pristine, nativeCompatSourceView(t, source, "fetch"))
	assert.Equal(t, int32(2), fetch.calls.Load())
}

// A reactive overflow retry reuses its step's snapshot. Edits the first
// attempt's model or model interceptor make to the request in place must not
// reach the retry or the tool interceptors of that step: the retry must send
// what the snapshot captured, nested metadata included. tool.Snapshot copies
// metadata recursively; the per-attempt request must keep that guarantee.
func TestToolsNativeCompatReactiveRetrySendsCapturedDefinitions(t *testing.T) {
	t.Parallel()

	for _, mutator := range []string{"model", "model interceptor"} {
		t.Run(mutator, func(t *testing.T) {
			t.Parallel()

			fetch := &stubTool{def: nativeCompatDefinition("fetch", "captured fetch"), result: json.RawMessage(`{"source":"captured"}`)}
			replacement := &stubTool{def: nativeCompatDefinition("fetch", "replacement fetch"), result: json.RawMessage(`{"source":"replacement"}`)}
			source := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, source.Register(fetch))

			var views []nativeCompatView

			model := fakellm.NewFakeModel(fakellm.WithContextWindow(200_000))
			model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				views = append(views, nativeCompatCaptureTool(t, req.Tools, "fetch"))
				if mutator == "model" {
					nativeCompatMutateFetch(t, req.Tools, "attempt one")
				}

				require.NoError(t, source.Unregister("fetch"))
				require.NoError(t, source.Register(replacement))

				return nil, nativeCompatOverflow()
			})
			model.When(fakellm.CallNumber(2)).ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
				views = append(views, nativeCompatCaptureTool(t, req.Tools, "fetch"))

				return respondWith(toolCall("retry", "fetch", `{}`))(req, cc)
			})
			model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				views = append(views, nativeCompatCaptureTool(t, req.Tools, "fetch"))

				return nativeCompatText("done"), nil
			})

			var applications int

			modelEdits := nativeCompatModelInterceptor(func(info *agent.ModelCallInfo) {
				applications++
				if mutator == "model interceptor" && applications == 1 {
					nativeCompatMutateFetch(t, info.Req.Tools, "attempt one")
				}
			})

			var toolViews []nativeCompatView

			toolCapture := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				if assert.NotNil(t, info.Definition) {
					toolViews = append(toolViews, nativeCompatCapture(t, *info.Definition))
				}

				return next(ctx, info)
			})
			ag, err := New("native-retry", "prompt", model, WithCompaction(CompactionConfig{}), WithTools(source), WithInterceptors(modelEdits, toolCapture))
			require.NoError(t, err)

			sess := &session.State{ID: "retry", Messages: userMessage("start")}

			for i := range 4 {
				id := fmt.Sprintf("old_%d", i)
				sess.Messages = append(sess.Messages,
					llm.NewMessage(llm.RoleAssistant, toolCall(id, "fetch", `{}`)),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, "fetch", fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", 16_000)), false)),
				)
			}

			sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))
			events := runAgent(t, ag, sess)

			var reactive bool

			for _, event := range events {
				if compacted, ok := event.(agent.CompactionEvent); ok && compacted.Report.Phase == agent.CompactionPhaseReactive {
					reactive = true
				}
			}

			require.True(t, reactive, "the first attempt must take the reactive retry path")
			assert.Equal(t, 3, applications)

			captured := nativeCompatPristine(t, "captured fetch")
			require.Len(t, views, 3)

			if mutator == "model" {
				assert.Equal(t, captured, views[0])
			}

			assert.Equal(t, captured.Description, views[1].Description, "the retry must reuse the step's snapshot, not the replaced source")
			assert.Equal(t, captured.Parameters, views[1].Parameters)
			assert.Equal(t, captured.Metadata, views[1].Metadata, "first-attempt edits to nested metadata leaked into the retry request")
			require.Len(t, toolViews, 1)
			assert.Equal(t, captured, toolViews[0], "first-attempt edits leaked into the retried step's tool definitions")
			assert.Equal(t, nativeCompatPristine(t, "replacement fetch"), views[2], "the next step must resolve the current source")

			assert.Equal(t, int32(1), fetch.calls.Load(), "the retried step dispatches to its captured executable")
			assert.Zero(t, replacement.calls.Load())

			results := toolsResults(sess, "fetch")
			require.NotEmpty(t, results)
			assert.Equal(t, "retry", results[len(results)-1].ID)
			assert.JSONEq(t, `{"source":"captured"}`, string(results[len(results)-1].Result))
			assert.Equal(t, captured, nativeCompatCapture(t, fetch.Definition()))
			assert.Equal(t, nativeCompatPristine(t, "replacement fetch"), nativeCompatSourceView(t, source, "fetch"))
		})
	}
}

// Canceling the invocation after the model answers, before its calls are
// dispatched, runs none of the snapshot's tools, answers every call as
// interrupted in request order, and resolves no further catalog. A later
// invocation on the same session runs against the source as it is then.
func TestToolsNativeCompatCancelBeforeDispatchRunsNothing(t *testing.T) {
	t.Parallel()

	write := &stubTool{def: llm.ToolDefinition{Name: "write", Description: "original write", Parameters: json.RawMessage(`{"type":"object"}`)}}
	read := &stubTool{def: llm.ToolDefinition{Name: "read", Description: "original read", Parameters: json.RawMessage(`{"type":"object"}`)}}
	writeNext := &stubTool{def: llm.ToolDefinition{Name: "write", Description: "replacement write", Parameters: json.RawMessage(`{"type":"object"}`)}, result: json.RawMessage(`{"source":"replacement write"}`)}
	source := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, source.Register(write))
	require.NoError(t, source.Register(read))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(toolCall("w1", "write", `{}`), toolCall("r1", "read", `{}`)))
	model.When(fakellm.CallNumber(2)).ThenRespondWith(respondWith(toolCall("w2", "write", `{}`)))
	model.When(fakellm.Any()).ThenRespondText("done")

	var intercepted atomic.Int32

	counter := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
		intercepted.Add(1)

		return next(ctx, info)
	})

	var resolutions atomic.Int32

	counted := toolsContextRegistry{snapshot: func(ctx context.Context) (tool.Registry, error) {
		resolutions.Add(1)

		return tool.Snapshot(ctx, source)
	}}
	ag, err := New("native-cancel", "prompt", model, WithTools(counted), WithToolConcurrency(2), WithInterceptors(counter))
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()

	sess := &session.State{ID: "cancel", Messages: userMessage("go")}

	var (
		events []agent.Event
		runErr error
	)

	for event, err := range ag.Run(ctx, agent.NewInvocationMetadata(sess, ag.Info())) {
		if err != nil {
			runErr = err

			break
		}

		events = append(events, event)
		if _, ok := event.(agent.MessageEvent); ok {
			cancel()
		}
	}

	require.NoError(t, runErr)
	assert.Equal(t, agent.FinishReasonInterrupted, finishReason(events))
	assert.Empty(t, nativeCompatToolResponses(events))
	assert.Zero(t, write.calls.Load())
	assert.Zero(t, read.calls.Load())
	assert.Zero(t, intercepted.Load())
	assert.Equal(t, int32(1), resolutions.Load(), "a canceled invocation must not resolve another step's catalog")
	assert.Len(t, model.CallsMatching(fakellm.Any()), 1)

	require.Len(t, sess.Messages, 3)
	responses := sess.Messages[2].ToolResponses()
	require.Len(t, responses, 2)

	for i, id := range []string{"w1", "r1"} {
		assert.Equal(t, id, responses[i].ID)
		assert.True(t, responses[i].IsError)
		assert.Contains(t, string(responses[i].Result), "interrupted")
	}

	require.NoError(t, source.Unregister("write"))
	require.NoError(t, source.Register(writeNext))

	sess.Messages = append(sess.Messages, userMessage("again")...)
	runAgent(t, ag, sess)

	assert.Zero(t, write.calls.Load(), "interrupted calls are not replayed")
	assert.Zero(t, read.calls.Load())
	assert.Equal(t, int32(1), writeNext.calls.Load())
	assert.Equal(t, []string{`{"source":"replacement write"}`}, toolResultsFor(sess.Messages, "write")[1:])

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3)
	assert.Equal(t, "replacement write", defByName(t, calls[1].Request.Tools, "write").Description)
}

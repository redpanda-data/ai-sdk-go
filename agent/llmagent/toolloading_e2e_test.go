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
	"errors"
	"strings"
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

// End-to-end tests drive the agent loop with a scripted fake model.

const incidentTool = "servicenow__create_incident"

// serviceDeskRegistry: one always-on tool and two deferred ones in different
// groups. The incident tool is returned so tests can check whether it ran.
func serviceDeskRegistry(t *testing.T) (tool.Registry, *stubTool) {
	t.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(&stubTool{def: llm.ToolDefinition{
		Name: "todo_write", Description: "Record the plan for a multi-step task.", Parameters: json.RawMessage(`{"type":"object"}`),
	}}))

	incident := &stubTool{
		def: llm.ToolDefinition{
			Name: incidentTool, Description: "Open a ServiceNow incident on behalf of a caller.",
			Parameters: json.RawMessage(`{"type":"object","properties":{"short_description":{"type":"string"}}}`),
		},
		result: json.RawMessage(`{"number":"INC0012345"}`),
	}
	require.NoError(t, tool.NewGroup(llm.ToolGroup{
		Name:         "servicenow",
		Description:  "ServiceNow incidents and CMDB lookups",
		Instructions: "Resolve the caller's sys_id before opening an incident.",
	}).Defer(incident).Register(registry))

	require.NoError(t, tool.NewGroup(llm.ToolGroup{Name: "jira"}).Defer(&stubTool{def: llm.ToolDefinition{
		Name: "jira__create_issue", Description: "Create a Jira issue in a project.", Parameters: json.RawMessage(`{"type":"object"}`),
	}}).Register(registry))

	return registry, incident
}

func newAgent(t *testing.T, model llm.Model, registry tool.Registry, opts ...Option) *LLMAgent {
	t.Helper()

	ag, err := New("service-desk", "You are a service desk agent.", model, append([]Option{WithTools(registry)}, opts...)...)
	require.NoError(t, err)

	return ag
}

// respondWith scripts a model response carrying several tool calls at once.
func respondWith(parts ...llm.Part) func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
	return func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
		return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, parts...), FinishReason: llm.FinishReasonToolCalls}, nil
	}
}

func toolCall(id, name, args string) *llm.ToolRequestPart {
	return llm.NewToolRequestPart(id, name, json.RawMessage(args))
}

type toolInterceptorFunc func(context.Context, *agent.ToolCallInfo, agent.ToolExecutionNext) (*llm.ToolResponsePart, error)

func (f toolInterceptorFunc) InterceptToolExecution(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
	return f(ctx, info, next)
}

// gateInterceptor holds one tool call open until released, so a test can act
// while that worker is still running.
type gateInterceptor struct {
	tool     string
	started  chan struct{}
	release  chan struct{}
	finished chan struct{}
	ctxErr   error
}

func newGate(toolName string) *gateInterceptor {
	return &gateInterceptor{tool: toolName, started: make(chan struct{}), release: make(chan struct{}), finished: make(chan struct{})}
}

func (g *gateInterceptor) InterceptToolExecution(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
	if info.Req.Name != g.tool {
		return next(ctx, info)
	}

	close(g.started)
	<-g.release

	g.ctxErr = ctx.Err()
	resp, err := next(ctx, info)

	close(g.finished)

	return resp, err
}

// stopAtFirstResult runs the agent until the first tool result and stops the
// consumer there, after the gated worker is inside its interceptor.
func stopAtFirstResult(t *testing.T, ag *LLMAgent, sess *session.State, gate *gateInterceptor) {
	t.Helper()

	for evt, err := range ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})) {
		require.NoError(t, err)

		if _, ok := evt.(agent.ToolResponseEvent); ok {
			<-gate.started

			break
		}
	}
}

// TestEndToEnd walks the whole loop: the model sees only always-on tools plus
// tool_search, searches, gets the deferred tool on the very next call, calls it,
// and the loaded set is left on the session. The catalog never changes; the
// prompt changes exactly when the tools array does.
func TestEndToEnd(t *testing.T) {
	t.Parallel()

	registry, incident := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.Not(fakellm.HasTool(incidentTool))).
		ThenRespondWithToolCall("tool_search", map[string]any{"query": "select:" + incidentTool})
	model.When(fakellm.And(fakellm.HasTool(incidentTool), fakellm.Not(fakellm.LastMessageHasToolResponse(incidentTool)))).
		ThenRespondWithToolCall(incidentTool, map[string]any{"short_description": "Laptop will not boot"})
	model.When(fakellm.Any()).ThenRespondText("Opened INC0012345 for you.")

	sess := &session.State{ID: "e2e", Messages: userMessage("My laptop will not boot")}
	events := runAgent(t, newAgent(t, model, registry), sess)

	assert.Equal(t, agent.FinishReasonStop, finishReason(events))
	assert.Equal(t, int32(1), incident.calls.Load())

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3, "one extra model call for the search")
	assert.Equal(t, []string{"todo_write", "tool_search"}, requestToolNames(calls[0].Request))
	assert.Equal(t, []string{incidentTool, "todo_write", "tool_search"}, requestToolNames(calls[1].Request))
	assert.Equal(t, requestToolNames(calls[1].Request), requestToolNames(calls[2].Request), "the tools array converges")

	prompt := func(i int) string { return calls[i].Request.Messages[0].TextContent() }
	catalog := func(s string) string {
		c, _, _ := strings.Cut(s, "## Tool instructions")

		return strings.TrimSpace(c)
	}

	for i := range calls {
		assert.Equal(t, 1, strings.Count(prompt(i), "## Additional tools"))
		assert.Contains(t, prompt(i), "`jira__create_issue`", "an unloaded tool stays in the catalog")
		assert.Contains(t, prompt(i), "`"+incidentTool+"`", "a loaded tool stays in the catalog too")
		assert.Equal(t, catalog(prompt(0)), catalog(prompt(i)), "the catalog never changes")
	}

	assert.NotContains(t, prompt(0), "Resolve the caller's sys_id")
	assert.Contains(t, prompt(1), "Resolve the caller's sys_id", "the group's instructions arrive with its first visible tool")
	assert.Equal(t, prompt(1), prompt(2))
	assert.Equal(t, []any{incidentTool}, sess.Metadata[loadedToolsMetadataKey])
}

// TestSiblingSearchDoesNotAuthorizeExecution: a response that loads a tool and
// calls it in the same breath wrote the arguments without the schema. The call
// gets a recovery error; the tool runs on the next turn.
func TestSiblingSearchDoesNotAuthorizeExecution(t *testing.T) {
	t.Parallel()

	registry, incident := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
		toolCall("s1", "tool_search", `{"query":"select:`+incidentTool+`"}`),
		toolCall("c1", incidentTool, `{"guessed":"argument"}`),
	))
	model.When(fakellm.CallNumber(2)).ThenRespondWithToolCall(incidentTool, map[string]any{"short_description": "Laptop will not boot"})
	model.When(fakellm.Any()).ThenRespondText("Opened INC0012345 for you.")

	sess := &session.State{ID: "sibling", Messages: userMessage("My laptop will not boot")}
	events := runAgent(t, newAgent(t, model, registry, WithToolConcurrency(2)), sess)

	assert.Equal(t, agent.FinishReasonStop, finishReason(events))
	assert.Equal(t, int32(1), incident.calls.Load(), "only the schema-backed call executes")

	results := toolResultsFor(sess.Messages, incidentTool)
	require.Len(t, results, 2)
	assert.Contains(t, results[0], "tool_not_loaded")
	assert.Contains(t, results[1], "INC0012345")
}

// TestSelfHealsAnUnloadedCall: a model that skips the search and calls a
// deferred tool directly gets one error, one round trip, and then succeeds.
func TestSelfHealsAnUnloadedCall(t *testing.T) {
	t.Parallel()

	registry, incident := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	// Call 1 is blind; call 2 retries with the schema in hand.
	model.When(fakellm.Or(fakellm.FirstCall(), fakellm.CallNumber(2))).
		ThenRespondWithToolCall(incidentTool, map[string]any{"short_description": "Laptop will not boot"})
	model.When(fakellm.Any()).ThenRespondText("Opened INC0012345 for you.")

	sess := &session.State{ID: "selfheal", Messages: userMessage("My laptop will not boot")}
	runAgent(t, newAgent(t, model, registry), sess)

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 3, "recovery costs exactly one extra round trip")
	assert.Contains(t, requestToolNames(calls[1].Request), incidentTool, "the retry has the schema")
	assert.Equal(t, int32(1), incident.calls.Load())
	assert.Contains(t, toolResultsFor(sess.Messages, incidentTool)[0], "tool_not_loaded")
}

// TestInterceptorsGovernLoads: a load is an effect of the tool_search call the
// interceptor chain lets through, as the chain left it.
func TestInterceptorsGovernLoads(t *testing.T) {
	t.Parallel()

	const originalQuery = `{"query":"select:` + incidentTool + `"}`

	tests := []struct {
		name        string
		callName    string
		interceptor toolInterceptorFunc
		wantLoads   []any
		wantError   bool
	}{
		{
			name: "deny without executing",
			interceptor: func(context.Context, *agent.ToolCallInfo, agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				return nil, errors.New("search denied")
			},
			wantError: true,
		},
		{
			name: "mock result without executing",
			interceptor: func(_ context.Context, info *agent.ToolCallInfo, _ agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				return llm.NewToolResponsePart(info.Req.ID, info.Req.Name, json.RawMessage(`{"loaded":[]}`), false), nil
			},
		},
		{
			name: "rewrite arguments",
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req.Arguments = json.RawMessage(`{"query":"select:jira__create_issue"}`)

				return next(ctx, info)
			},
			wantLoads: []any{"jira__create_issue"},
		},
		{
			// Call identity is immutable even when the new name is a framework tool.
			name:     "rewrite ordinary tool into search is refused",
			callName: "todo_write",
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req.Name = "tool_search"

				return next(ctx, info)
			},
			wantError: true,
		},
		{
			name: "retry with a different selection loads both",
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				if _, err := next(ctx, info); err != nil {
					return nil, err
				}

				info.Req.Arguments = json.RawMessage(`{"query":"select:jira__create_issue"}`)

				return next(ctx, info)
			},
			wantLoads: []any{"jira__create_issue", incidentTool},
		},
		{
			name: "post-processing error keeps executed effects",
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				if _, err := next(ctx, info); err != nil {
					return nil, err
				}

				return nil, errors.New("postprocessing failed")
			},
			wantLoads: []any{incidentTool},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			callName := tt.callName
			if callName == "" {
				callName = "tool_search"
			}

			registry, _ := serviceDeskRegistry(t)
			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(toolCall("search-1", callName, originalQuery)))
			model.When(fakellm.Any()).ThenRespondText("Done.")

			sess := &session.State{ID: "interception", Messages: userMessage("Do the task.")}
			events := runAgent(t, newAgent(t, model, registry, WithInterceptors(tt.interceptor)), sess)
			assert.Equal(t, agent.FinishReasonStop, finishReason(events))

			if len(tt.wantLoads) == 0 {
				assert.Empty(t, sess.Metadata[loadedToolsMetadataKey])
			} else {
				assert.Equal(t, tt.wantLoads, sess.Metadata[loadedToolsMetadataKey])
			}

			calls := model.CallsMatching(fakellm.Any())
			require.Len(t, calls, 2)

			for _, name := range []string{incidentTool, "jira__create_issue"} {
				wantVisible := false
				for _, load := range tt.wantLoads {
					wantVisible = wantVisible || load == name
				}

				assert.Equal(t, wantVisible, strings.Contains(strings.Join(requestToolNames(calls[1].Request), ","), name), name)
			}

			// Rewrites change what executes, not the assistant message in the session.
			requests := sess.Messages[1].ToolRequests()
			require.Len(t, requests, 1)
			assert.Equal(t, callName, requests[0].Name)
			assert.JSONEq(t, originalQuery, string(requests[0].Arguments))

			responses := sess.Messages[2].ToolResponses()
			require.Len(t, responses, 1)
			assert.Equal(t, tt.wantError, responses[0].IsError)
		})
	}
}

// TestAdmissionOrderedAfterInterception: two searches whose schemas fit the
// window separately but not together. The first is denied by its interceptor
// while it is held open; the second must wait for it and then get the room the
// first did not take.
func TestAdmissionOrderedAfterInterception(t *testing.T) {
	t.Parallel()

	registry, _ := serviceDeskRegistry(t)
	for _, name := range []string{"export_a", "export_b"} {
		require.NoError(t, registry.Register(&stubTool{def: llm.ToolDefinition{
			Name: name, Parameters: json.RawMessage(`{"type":"object","description":"` + strings.Repeat("padding ", 2600) + `"}`),
		}}, tool.WithDeferred()))
	}

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(20_000))
	model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(
		toolCall("first", "tool_search", `{"query":"select:export_a"}`),
		toolCall("second", "tool_search", `{"query":"select:export_b"}`),
		toolCall("ordinary", "todo_write", `{}`),
	))
	model.When(fakellm.Any()).ThenRespondText("Done.")

	started, release, second := make(chan struct{}), make(chan struct{}), make(chan struct{})
	interceptor := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
		switch info.Req.ID {
		case "first":
			close(started)

			select {
			case <-release:
			case <-ctx.Done():
				return nil, ctx.Err()
			}

			return nil, errors.New("first search denied")
		case "second":
			close(second)
		}

		return next(ctx, info)
	})

	sess := &session.State{ID: "ordered", Messages: userMessage("Export everything.")}
	ag := newAgent(t, model, registry, WithToolConcurrency(3), WithInterceptors(interceptor))

	for ev, err := range ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})) {
		require.NoError(t, err)

		if response, ok := ev.(agent.ToolResponseEvent); ok && response.Response.ID == "ordinary" {
			<-started

			select {
			case <-second:
				t.Fatal("second search overtook the first search's interceptor")
			default:
			}

			close(release)
		}
	}

	assert.Equal(t, []any{"export_b"}, sess.Metadata[loadedToolsMetadataKey],
		"a denied search leaves the room for the next search")
}

// TestOutstandingSearchCannotMutateASavedSession: when the consumer stops
// mid-burst, a tool_search still running cannot change the session the runner
// is about to save, every call still has a result, and the worker is cancelled.
func TestOutstandingSearchCannotMutateASavedSession(t *testing.T) {
	t.Parallel()

	registry, _ := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(respondWith(
		toolCall("t1", "todo_write", `{}`),
		toolCall("s1", "tool_search", `{"query":"select:`+incidentTool+`"}`),
	))

	gate := newGate("tool_search")
	sess := &session.State{ID: "stopped", Messages: userMessage("hello")}
	stopAtFirstResult(t, newAgent(t, model, registry, WithToolConcurrency(2), WithInterceptors(gate)), sess, gate)

	saved := sess.Clone()
	assert.Empty(t, saved.Metadata[loadedToolsMetadataKey])
	responses := saved.Messages[len(saved.Messages)-1].ToolResponses()
	require.Len(t, responses, 2, "every request needs a result before this transcript can resume")
	assert.False(t, responses[0].IsError)
	assert.Contains(t, string(responses[1].Result), "interrupted")

	close(gate.release)

	select {
	case <-gate.finished:
	case <-time.After(5 * time.Second):
		t.Fatal("the gated search never finished")
	}

	require.Error(t, gate.ctxErr, "an outstanding worker is cancelled when execution returns")
	assert.Empty(t, sess.Metadata[loadedToolsMetadataKey], "an uncollected result's loads are never committed")
}

// TestOversizedLoads: neither two searches in one response nor a blind call to
// an oversized tool can push the session past what the model's window holds.
// Refusals show in the results, nothing refused is persisted, the session
// continues.
func TestOversizedLoads(t *testing.T) {
	t.Parallel()

	// Window 20k: reserve 4096, usable 15904, target 9542. Each big schema is
	// ~7k estimated tokens: one fits, two do not; the giant one never fits.
	big := `{"type":"object","properties":{"blob":{"type":"string","description":"` + strings.Repeat("padding ", 2600) + `"}}}`
	giant := `{"type":"object","properties":{"blob":{"type":"string","description":"` + strings.Repeat("padding ", 6000) + `"}}}`

	run := func(t *testing.T, first func(*llm.Request, *fakellm.CallContext) (*llm.Response, error)) *session.State {
		t.Helper()

		registry := tool.NewRegistry(tool.RegistryConfig{})
		for name, params := range map[string]string{"legacy__export_a": big, "legacy__export_b": big, "legacy__export_all": giant} {
			require.NoError(t, registry.Register(&stubTool{def: llm.ToolDefinition{Name: name, Description: "Export.", Parameters: json.RawMessage(params)}}, tool.WithDeferred()))
		}

		model := fakellm.NewFakeModel(fakellm.WithContextWindow(20_000))
		model.When(fakellm.FirstCall()).ThenRespondWith(first)
		model.When(fakellm.Any()).ThenRespondText("Done what I could.")

		sess := &session.State{ID: "admission", Messages: userMessage("export everything")}
		ag := newAgent(t, model, registry)
		require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

		sess.Messages = append(sess.Messages, userMessage("thanks")...)
		require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)), "still usable afterwards")

		return sess
	}

	t.Run("two searches", func(t *testing.T) {
		t.Parallel()

		sess := run(t, respondWith(
			toolCall("s1", "tool_search", `{"query":"select:legacy__export_a"}`),
			toolCall("s2", "tool_search", `{"query":"select:legacy__export_b"}`),
		))

		results := toolResultsFor(sess.Messages, "tool_search")
		require.Len(t, results, 2)
		assert.Contains(t, results[0], `"loaded":["legacy__export_a"]`)
		assert.Contains(t, results[1], `"too_large":["legacy__export_b"]`)
		assert.Equal(t, []any{"legacy__export_a"}, sess.Metadata[loadedToolsMetadataKey])
	})

	t.Run("blind call to an oversized tool", func(t *testing.T) {
		t.Parallel()

		sess := run(t, respondWith(toolCall("c1", "legacy__export_all", `{"guessed":true}`)))

		assert.Contains(t, toolResultsFor(sess.Messages, "legacy__export_all")[0], "tool_too_large")
		assert.Empty(t, sess.Metadata[loadedToolsMetadataKey])
	})
}

// TestRegistryChangesBetweenTurns: MCP sync adds and removes tools while a
// session is live. A removed tool leaves the request although the session still
// lists it as loaded, a new deferred tool appears in the catalog and loads, a
// stale call gets a plain error, and a tool that comes back keeps its status.
func TestRegistryChangesBetweenTurns(t *testing.T) {
	t.Parallel()

	registry, incident := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.CallNumber(1)).ThenRespondWithToolCall("tool_search", map[string]any{"query": "select:" + incidentTool})
	model.When(fakellm.CallNumber(2)).ThenRespondWithToolCall(incidentTool, map[string]any{"short_description": "boot"})
	model.When(fakellm.CallNumber(3)).ThenRespondText("Opened.")
	model.When(fakellm.CallNumber(4)).ThenRespondWith(respondWith(
		toolCall("stale", incidentTool, `{"short_description":"again"}`),
		toolCall("s2", "tool_search", `{"query":"select:confluence__get_page"}`),
	))
	model.When(fakellm.CallNumber(5)).ThenRespondWithToolCall("confluence__get_page", map[string]any{"id": "7781"})
	model.When(fakellm.Any()).ThenRespondText("Done.")

	ag := newAgent(t, model, registry)
	sess := &session.State{ID: "churn", Messages: userMessage("open an incident")}
	runAgent(t, ag, sess)
	require.Equal(t, int32(1), incident.calls.Load())

	// The sync: the incident tool is gone, a Confluence tool arrived.
	require.NoError(t, registry.Unregister(incidentTool))

	page := &stubTool{def: llm.ToolDefinition{Name: "confluence__get_page", Description: "Fetch a page.", Parameters: json.RawMessage(`{"type":"object"}`)}}
	require.NoError(t, registry.Register(page, tool.WithDeferred(), tool.WithGroup(llm.ToolGroup{Name: "confluence"})))

	sess.Messages = append(sess.Messages, userMessage("now the page")...)
	require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

	calls := model.CallsMatching(fakellm.Any())
	require.Len(t, calls, 6)
	assert.Equal(t, []string{"todo_write", "tool_search"}, requestToolNames(calls[3].Request), "the removed tool leaves the request")
	assert.Contains(t, calls[3].Request.Messages[0].TextContent(), "confluence__get_page")
	assert.NotContains(t, calls[3].Request.Messages[0].TextContent(), incidentTool)

	assert.Contains(t, toolResultsFor(sess.Messages, incidentTool)[1], "not found")
	assert.Equal(t, int32(1), incident.calls.Load(), "the removed tool did not run again")
	assert.Equal(t, int32(1), page.calls.Load())
	assert.ElementsMatch(t, []any{"confluence__get_page", incidentTool}, sess.Metadata[loadedToolsMetadataKey],
		"the session keeps the removed name, so the tool is back the moment a sync restores it")

	require.NoError(t, registry.Register(incident, tool.WithDeferred(), tool.WithGroup(llm.ToolGroup{Name: "servicenow"})))

	sess.Messages = append(sess.Messages, userMessage("thanks")...)
	runAgent(t, ag, sess)

	last, err := model.LastCall()
	require.NoError(t, err)
	assert.Equal(t, []string{"confluence__get_page", incidentTool, "todo_write", "tool_search"}, requestToolNames(last.Request))
}

// TestToolVanishesBetweenResponseAndExecution: the registry loses a loaded tool
// after the model decided to call it. The call fails with an ordinary error and
// the session stays consistent.
func TestToolVanishesBetweenResponseAndExecution(t *testing.T) {
	t.Parallel()

	registry, incident := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.CallNumber(1)).ThenRespondWithToolCall("tool_search", map[string]any{"query": "select:" + incidentTool})
	model.When(fakellm.CallNumber(2)).ThenRespondWith(func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
		require.NoError(t, registry.Unregister(incidentTool)) // the sync lands mid-answer

		return respondWith(toolCall("c1", incidentTool, `{}`))(nil, nil)
	})
	model.When(fakellm.Any()).ThenRespondText("Sorry, that tool is gone.")

	sess := &session.State{ID: "vanish", Messages: userMessage("open an incident")}
	require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, newAgent(t, model, registry), sess)))

	assert.Zero(t, incident.calls.Load())
	assert.Contains(t, toolResultsFor(sess.Messages, incidentTool)[0], "not found")
	assert.Equal(t, []any{incidentTool}, sess.Metadata[loadedToolsMetadataKey])
}

// TestResumesFromSession: a session whose metadata names a loaded tool sends
// that tool's schema on its very first call, with no search.
func TestResumesFromSession(t *testing.T) {
	t.Parallel()

	registry, _ := serviceDeskRegistry(t)
	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("Already loaded.")

	sess := &session.State{
		ID:       "resumed",
		Metadata: map[string]any{loadedToolsMetadataKey: []any{"jira__create_issue"}}, // as a store hands it back
		Messages: userMessage("hello"),
	}
	runAgent(t, newAgent(t, model, registry), sess)

	first, err := model.FirstCall()
	require.NoError(t, err)
	assert.Equal(t, []string{"jira__create_issue", "todo_write", "tool_search"}, requestToolNames(first.Request))
}

// TestNoDeferredToolsSendsEverything is the control: lazy loading is automatic,
// so a registry with nothing deferred produces the request it always did.
func TestNoDeferredToolsSendsEverything(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	group := tool.NewGroup(llm.ToolGroup{Name: "desk", Instructions: "Be brief."})
	for _, name := range []string{"todo_write", incidentTool, "jira__create_issue"} {
		group.Add(&stubTool{def: llm.ToolDefinition{Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object"}`)}})
	}

	require.NoError(t, group.Register(registry))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("Hi.")

	sess := &session.State{ID: "off", Messages: userMessage("hello")}
	runAgent(t, newAgent(t, model, registry), sess)

	first, err := model.FirstCall()
	require.NoError(t, err)
	assert.Equal(t, []string{"jira__create_issue", incidentTool, "todo_write"}, requestToolNames(first.Request))
	assert.NotContains(t, first.Request.Messages[0].TextContent(), "## Additional tools")
	assert.Contains(t, first.Request.Messages[0].TextContent(), "Be brief.", "group instructions apply to always-on tools too")
	assert.Empty(t, sess.Metadata[loadedToolsMetadataKey])
}

func TestRegisteredSearchToolRunsWithoutDeferral(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	search := &stubTool{def: llm.ToolDefinition{Name: toolSearchName, Parameters: json.RawMessage(`{"type":"object"}`)}}
	require.NoError(t, registry.Register(search))

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall(toolSearchName, map[string]any{})
	model.When(fakellm.Any()).ThenRespondText("Done.")
	runAgent(t, newAgent(t, model, registry), &session.State{ID: "ordinary-search", Messages: userMessage("Search.")})
	assert.Equal(t, int32(1), search.calls.Load())
}

func TestInterceptorsCannotChangeCallIdentity(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name        string
		callName    string
		interceptor toolInterceptorFunc
	}{
		{
			name: "ordinary call renamed to unloaded tool", callName: "todo_write",
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req.Name = incidentTool
				return next(ctx, info)
			},
		},
		{
			name: "search renamed to ordinary tool", callName: toolSearchName,
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req.Name = "todo_write"
				return next(ctx, info)
			},
		},
		{
			name: "search ID changed", callName: toolSearchName,
			interceptor: func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req.ID = "different"
				return next(ctx, info)
			},
		},
		{
			name: "mock changes request identity", callName: toolSearchName,
			interceptor: func(_ context.Context, info *agent.ToolCallInfo, _ agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				info.Req = toolCall("different", "todo_write", `{}`)
				return llm.NewToolResponsePart("original", toolSearchName, json.RawMessage(`{}`), false), nil
			},
		},
		{
			name: "mock changes response name", callName: toolSearchName,
			interceptor: func(_ context.Context, info *agent.ToolCallInfo, _ agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				return llm.NewToolResponsePart(info.Req.ID, "different", json.RawMessage(`{}`), false), nil
			},
		},
		{
			name: "mock changes response ID", callName: toolSearchName,
			interceptor: func(_ context.Context, info *agent.ToolCallInfo, _ agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
				return llm.NewToolResponsePart("different", info.Req.Name, json.RawMessage(`{}`), false), nil
			},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry, incident := serviceDeskRegistry(t)
			ordinary, err := registry.Get("todo_write")
			require.NoError(t, err)

			ordinaryTool, ok := ordinary.(*stubTool)
			require.True(t, ok)

			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(respondWith(toolCall("original", tt.callName,
				`{"query":"select:`+incidentTool+`"}`)))
			model.When(fakellm.Any()).ThenRespondText("Done.")

			sess := &session.State{ID: "identity", Messages: userMessage("Act.")}
			runAgent(t, newAgent(t, model, registry, WithInterceptors(tt.interceptor)), sess)

			assert.Zero(t, incident.calls.Load())
			assert.Zero(t, ordinaryTool.calls.Load())
			assert.Empty(t, sess.Metadata[loadedToolsMetadataKey])
			responses := sess.Messages[2].ToolResponses()
			require.Len(t, responses, 1)
			assert.True(t, responses[0].IsError)
			assert.Equal(t, "original", responses[0].ID)
			assert.Equal(t, tt.callName, responses[0].Name)
			assert.Contains(t, string(responses[0].Result), "must preserve")
		})
	}
}

//
// Executor behaviour that applies to every agent, surfaced by lazy loading.
//

// TestConsumerStopLeavesACompleteTranscript: a consumer that breaks out of Run
// mid-burst must not be handed another event (Go panics if an iterator
// continues after its body returned false), and every call still gets a result
// so the transcript can be resumed.
func TestConsumerStopLeavesACompleteTranscript(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	for _, name := range []string{"fast", "slow"} {
		require.NoError(t, registry.Register(&stubTool{def: llm.ToolDefinition{Name: name, Description: name, Parameters: json.RawMessage(`{"type":"object"}`)}}))
	}

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(respondWith(toolCall("f1", "fast", `{}`), toolCall("s1", "slow", `{}`)))

	gate := newGate("slow")
	sess := &session.State{ID: "stop", Messages: userMessage("go")}
	stopAtFirstResult(t, newAgent(t, model, registry, WithToolConcurrency(2), WithInterceptors(gate)), sess, gate)

	close(gate.release)
	<-gate.finished

	responses := sess.Messages[len(sess.Messages)-1].ToolResponses()
	require.Len(t, responses, 2)
	assert.Equal(t, "fast", responses[0].Name)
	assert.Contains(t, string(responses[1].Result), "interrupted")
	require.Error(t, gate.ctxErr, "the outstanding worker was cancelled")
	assert.Len(t, model.CallsMatching(fakellm.Any()), 1, "the loop stopped rather than running on silently")
}

// TestInterceptorEditsDoNotMutateSessionMessage: an interceptor that rewrites
// the request changes what the tool receives, not the stored assistant message.
func TestInterceptorEditsDoNotMutateSessionMessage(t *testing.T) {
	t.Parallel()

	echo := &stubTool{def: llm.ToolDefinition{Name: "echo", Description: "Echo.", Parameters: json.RawMessage(`{"type":"object"}`)}}
	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(echo))

	rewrite := toolInterceptorFunc(func(ctx context.Context, info *agent.ToolCallInfo, next agent.ToolExecutionNext) (*llm.ToolResponsePart, error) {
		info.Req.Arguments = json.RawMessage(`{"a":2}`)

		return next(ctx, info)
	})

	model := fakellm.NewFakeModel()
	model.When(fakellm.FirstCall()).ThenRespondWithToolCall("echo", map[string]any{"a": 1})
	model.When(fakellm.Any()).ThenRespondText("done")

	sess := &session.State{ID: "edit", Messages: userMessage("hi")}
	runAgent(t, newAgent(t, model, registry, WithInterceptors(rewrite)), sess)

	assert.JSONEq(t, `{"a":2}`, string(echo.arguments()[0]), "the tool sees the interceptor's edit")
	assert.JSONEq(t, `{"a":1}`, string(sess.Messages[1].ToolRequests()[0].Arguments), "the session keeps what the model said")
}

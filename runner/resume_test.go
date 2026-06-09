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

package runner_test

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// pausingTool returns AwaitReasonToolResult on first call (so the
// invocation pauses) and a final result on Resume — but Resume goes
// through ResumeWithToolResponse so the tool is never re-entered. The
// runner records the supplied output directly.
type pausingDeployInput struct {
	Version string `json:"version"`
}

type pausingDeployOutput struct {
	DeploymentID string `json:"deployment_id"`
	Status       string `json:"status"`
}

func newPausingTool() tool.Tool {
	return tool.Must(tool.Func(
		tool.Spec{
			Name:        "deploy",
			Description: "Trigger a deployment that pauses until the job finishes.",
			Async:       tool.AsyncExternalResult(),
		},
		func(_ context.Context, in pausingDeployInput) (tool.Result[pausingDeployOutput], error) {
			return tool.Pending(
				pausingDeployOutput{DeploymentID: "dep-1", Status: "queued"},
				tool.WithAwaitMessage("Deployment "+in.Version+" queued."),
				tool.WithCorrelationID("dep-1"),
			), nil
		},
	))
}

func TestRunner_Resume_ExternalResult_EndToEnd(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	registry := tool.NewRegistry()
	require.NoError(t, registry.Register(newPausingTool()))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		// On the first turn the user says "deploy"; emit a tool call.
		// On subsequent turns (post-resume) emit a final assistant text.
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleUser {
				for _, p := range msg.Content {
					if tr, ok := p.(*llm.ToolResponsePart); ok && tr.Name == "deploy" {
						// Tool response is in history → final turn.
						return &llm.Response{
							Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Deployment finished.")),
							FinishReason: llm.FinishReasonStop,
						}, nil
					}
				}
			}
		}

		args, _ := json.Marshal(map[string]string{"version": "v1.2.3"})

		return &llm.Response{
			Message: llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call-1", "deploy", args)),
			FinishReason: llm.FinishReasonToolCalls,
		}, nil
	})

	ag, err := llmagent.New("deployer", "You deploy things.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	store := session.NewInMemoryStore()
	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// First Run: pauses.
	events := collectEvents(t, r.Run(ctx, "user-1", "sess-1", llm.NewMessage(llm.RoleUser, llm.NewTextPart("deploy v1.2.3"))))

	end := findInvocationEndEvent(events)
	require.NotNil(t, end)
	assert.Equal(t, agent.FinishReasonPaused, end.FinishReason)
	require.Len(t, end.PendingCalls, 1)
	assert.Equal(t, "call-1", end.PendingCalls[0].CallID)
	assert.Equal(t, tool.AwaitReasonToolResult, end.PendingCalls[0].Reason)

	// Verify session has the pending entry persisted.
	sess, err := store.Load(ctx, "sess-1")
	require.NoError(t, err)
	require.Contains(t, sess.PendingToolCalls, "call-1")

	// Submit a Resume with a final result.
	finalOutput := json.RawMessage(`{"deployment_id":"dep-1","status":"success","url":"https://example/deploy/1"}`)
	resumeEvents := collectEvents(t, mustResume(ctx, t, r, "user-1", "sess-1", runner.Resumption{CallID: "call-1", Output: finalOutput}))

	end2 := findInvocationEndEvent(resumeEvents)
	require.NotNil(t, end2)
	assert.Equal(t, agent.FinishReasonStop, end2.FinishReason, "agent should complete after resume")

	// Pending cleared, receipt stored, history has the final response.
	sess, err = store.Load(ctx, "sess-1")
	require.NoError(t, err)
	assert.NotContains(t, sess.PendingToolCalls, "call-1")
	require.Contains(t, sess.ResumeReceipts, "call-1")

	// Idempotency: same resume payload again should be acknowledged
	// without mutating session state.
	dupEvents := collectEvents(t, mustResume(ctx, t, r, "user-1", "sess-1", runner.Resumption{CallID: "call-1", Output: finalOutput}))

	sawAck := false

	for _, evt := range dupEvents {
		if _, ok := evt.(agent.ResumeAcknowledgedEvent); ok {
			sawAck = true
			break
		}
	}

	assert.True(t, sawAck, "duplicate Resume with same payload should emit ResumeAcknowledgedEvent")
}

func TestRunner_Resume_Conflict(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	registry := tool.NewRegistry()
	require.NoError(t, registry.Register(newPausingTool()))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		args, _ := json.Marshal(map[string]string{"version": "v1"})

		return &llm.Response{
			Message: llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call-1", "deploy", args)),
			FinishReason: llm.FinishReasonToolCalls,
		}, nil
	})

	ag, err := llmagent.New("deployer", "You deploy things.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	store := session.NewInMemoryStore()
	r, err := runner.New(ag, store)
	require.NoError(t, err)

	// Pause.
	_ = collectEvents(t, r.Run(ctx, "u", "s", llm.NewMessage(llm.RoleUser, llm.NewTextPart("deploy"))))

	// First Resume succeeds.
	_ = collectEvents(t, mustResume(ctx, t, r, "u", "s", runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"status":"ok"}`)}))

	// Second Resume with a DIFFERENT payload must produce a conflict
	// without mutating the session — and the conflict surfaces EAGERLY,
	// so a caller that never ranges the stream still sees it.
	_, err = r.Resume(ctx, "u", "s", runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"status":"different"}`)})
	require.ErrorIs(t, err, runner.ErrResumeConflict)
}


// mustResume asserts the eager phase of Resume succeeded and returns
// the continuation stream.
func mustResume(ctx context.Context, t *testing.T, r *runner.Runner, userID, sessionID string, results ...runner.Resumption) func(func(agent.Event, error) bool) {
	t.Helper()

	stream, err := r.Resume(ctx, userID, sessionID, results...)
	require.NoError(t, err)

	return stream
}

func TestRunner_Resume_MutatesEagerly_WithoutRangingStream(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	registry := tool.NewRegistry()
	require.NoError(t, registry.Register(newPausingTool()))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		return &llm.Response{
			Message: llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call-1", "deploy", json.RawMessage(`{"version":"v1"}`))),
			FinishReason: llm.FinishReasonToolCalls,
		}, nil
	})

	ag, err := llmagent.New("deployer", "You deploy things.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	store := session.NewInMemoryStore()
	r, err := runner.New(ag, store)
	require.NoError(t, err)

	_ = collectEvents(t, r.Run(ctx, "u", "s", llm.NewMessage(llm.RoleUser, llm.NewTextPart("deploy"))))

	// Resume WITHOUT ranging the returned stream: the mutation must
	// already be applied and saved when Resume returns.
	_, err = r.Resume(ctx, "u", "s", runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"status":"ok"}`)})
	require.NoError(t, err)

	sess, err := store.Load(ctx, "s")
	require.NoError(t, err)
	assert.NotContains(t, sess.PendingToolCalls, "call-1", "pending call resolved without ranging the stream")
	assert.Contains(t, sess.ResumeReceipts, "call-1")

	// Unknown call ID surfaces eagerly too.
	_, err = r.Resume(ctx, "u", "s", runner.Resumption{CallID: "nope", Output: json.RawMessage(`{}`)})
	require.ErrorIs(t, err, runner.ErrPendingCallNotFound)
}

// approvalGate is a ToolInterceptor that pauses gated tools for human
// approval and consumes the decision on re-entry, per the
// agent.ToolCallInfo.Resume contract.
type approvalGate struct {
	gated map[string]bool
}

func (g *approvalGate) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (tool.Execution, error) {
	if !g.gated[info.Req.Name] {
		return next(ctx, info)
	}

	if info.Resume == nil {
		// First entry: pause for approval instead of executing.
		return tool.Execution{
			Output: json.RawMessage(`{"status":"awaiting_approval"}`),
			Await: &tool.Await{
				Reason:  tool.AwaitReasonApproval,
				Resume:  tool.ResumeWithReentry,
				Message: "Approve " + info.Req.Name + "?",
			},
		}, nil
	}

	// Re-entry: consume the decision so the tool runs fresh below.
	decision := info.Resume
	info.Resume = nil

	if decision.Error != "" {
		return tool.Execution{}, errors.New(decision.Error)
	}

	return next(ctx, info)
}

// newCountingTool returns a Done-tool that counts executions, plus the
// counter.
func newCountingTool(name string, result tool.Result[map[string]string]) (tool.Tool, *atomic.Int32) {
	var count atomic.Int32

	t := tool.Must(tool.Func(
		tool.Spec{Name: name, Description: "Gated operation."},
		func(_ context.Context, _ struct{}) (tool.Result[map[string]string], error) {
			count.Add(1)
			return result, nil
		},
	))

	return t, &count
}

// newApprovalFixture wires a fake model that calls `name` once and
// finishes after it sees a tool response for it.
func newApprovalFixture(t *testing.T, gated tool.Tool) (*runner.Runner, *session.InMemoryStore) {
	t.Helper()

	registry := tool.NewRegistry()
	require.NoError(t, registry.Register(gated))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		for _, msg := range req.Messages {
			if msg.Role != llm.RoleUser {
				continue
			}

			for _, p := range msg.Content {
				if tr, ok := p.(*llm.ToolResponsePart); ok && tr.Name == gated.Name() {
					return &llm.Response{
						Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Operation handled.")),
						FinishReason: llm.FinishReasonStop,
					}, nil
				}
			}
		}

		return &llm.Response{
			Message: llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart("call-1", gated.Name(), json.RawMessage(`{}`))),
			FinishReason: llm.FinishReasonToolCalls,
		}, nil
	})

	ag, err := llmagent.New("gatekeeper", "You operate carefully.", model,
		llmagent.WithTools(registry),
		llmagent.WithInterceptors(&approvalGate{gated: map[string]bool{gated.Name(): true}}),
	)
	require.NoError(t, err)

	store := session.NewInMemoryStore()
	r, err := runner.New(ag, store)
	require.NoError(t, err)

	return r, store
}

func TestRunner_Resume_ApprovalApproved_RunsToolOnce(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	gated, count := newCountingTool("db_write", tool.Done(map[string]string{"rows": "1"}))
	r, store := newApprovalFixture(t, gated)

	events := collectEvents(t, r.Run(ctx, "u", "s", llm.NewMessage(llm.RoleUser, llm.NewTextPart("write"))))
	end := findInvocationEndEvent(events)
	require.NotNil(t, end)
	require.Equal(t, agent.FinishReasonPaused, end.FinishReason)
	require.Len(t, end.PendingCalls, 1)
	assert.Equal(t, tool.AwaitReasonApproval, end.PendingCalls[0].Reason)
	assert.Equal(t, int32(0), count.Load(), "tool must not run before approval")

	resumeEvents := collectEvents(t, mustResume(ctx, t, r, "approver", "s",
		runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"approved":true}`)}))

	end2 := findInvocationEndEvent(resumeEvents)
	require.NotNil(t, end2)
	assert.Equal(t, agent.FinishReasonStop, end2.FinishReason)
	assert.Equal(t, int32(1), count.Load(), "approved tool must run exactly once")

	sess, err := store.Load(ctx, "s")
	require.NoError(t, err)
	assert.NotContains(t, sess.PendingToolCalls, "call-1")
	assert.Contains(t, sess.ResumeReceipts, "call-1")
}

func TestRunner_Resume_ApprovalDenied_NeverRunsTool(t *testing.T) {
	t.Parallel()

	ctx := context.Background()
	gated, count := newCountingTool("db_write", tool.Done(map[string]string{"rows": "1"}))
	r, store := newApprovalFixture(t, gated)

	_ = collectEvents(t, r.Run(ctx, "u", "s", llm.NewMessage(llm.RoleUser, llm.NewTextPart("write"))))

	resumeEvents := collectEvents(t, mustResume(ctx, t, r, "approver", "s",
		runner.Resumption{CallID: "call-1", Error: "denied by operator bob"}))

	end := findInvocationEndEvent(resumeEvents)
	require.NotNil(t, end)
	assert.Equal(t, int32(0), count.Load(), "denied tool must never execute")

	// The recorded response must be a tool error carrying the denial.
	sess, err := store.Load(ctx, "s")
	require.NoError(t, err)
	assert.NotContains(t, sess.PendingToolCalls, "call-1")

	var denial *llm.ToolResponsePart

	for _, msg := range sess.Messages {
		for _, p := range msg.Content {
			if tr, ok := p.(*llm.ToolResponsePart); ok && tr.ID == "call-1" {
				denial = tr
			}
		}
	}

	require.NotNil(t, denial)
	assert.True(t, denial.IsError)
	assert.Contains(t, string(denial.Result), "denied by operator bob")
}

func TestRunner_Resume_ApprovalThenExternalWork_ChainedPause(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	// The gated tool itself pauses for an external result once approved.
	gated, count := newCountingTool("deploy_prod",
		tool.Pending(map[string]string{"status": "deploying"}, tool.WithCorrelationID("job-7")))
	r, store := newApprovalFixture(t, gated)

	_ = collectEvents(t, r.Run(ctx, "u", "s", llm.NewMessage(llm.RoleUser, llm.NewTextPart("deploy"))))

	// Approve: the interceptor consumes the decision, the tool runs and
	// pauses again awaiting the external job. Same call ID, new reason.
	approveEvents := collectEvents(t, mustResume(ctx, t, r, "approver", "s",
		runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"approved":true}`)}))

	sawChainedPending := false

	for _, evt := range approveEvents {
		if pe, ok := evt.(agent.ToolPendingEvent); ok && pe.PendingCall.Reason == tool.AwaitReasonToolResult {
			sawChainedPending = true
		}
	}

	assert.True(t, sawChainedPending, "chained pause should surface a ToolPendingEvent with the new reason")
	assert.Equal(t, int32(1), count.Load())

	sess, err := store.Load(ctx, "s")
	require.NoError(t, err)
	require.Contains(t, sess.PendingToolCalls, "call-1", "chained pause keeps the pending entry")
	assert.Equal(t, string(tool.AwaitReasonToolResult), sess.PendingToolCalls["call-1"].Reason)
	assert.Equal(t, "job-7", sess.PendingToolCalls["call-1"].CorrelationID, "chained pause must refresh correlation ID")
	assert.NotContains(t, sess.ResumeReceipts, "call-1", "no receipt while the call is still pending")

	// Final resume with the external result completes the call.
	finalEvents := collectEvents(t, mustResume(ctx, t, r, "webhook", "s",
		runner.Resumption{CallID: "call-1", Output: json.RawMessage(`{"status":"success"}`)}))

	end := findInvocationEndEvent(finalEvents)
	require.NotNil(t, end)
	assert.Equal(t, agent.FinishReasonStop, end.FinishReason)

	sess, err = store.Load(ctx, "s")
	require.NoError(t, err)
	assert.NotContains(t, sess.PendingToolCalls, "call-1")
	assert.Contains(t, sess.ResumeReceipts, "call-1")
	assert.Equal(t, int32(1), count.Load(), "tool ran exactly once across the whole flow")
}

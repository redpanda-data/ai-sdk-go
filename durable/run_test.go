// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// fakeProducer captures commands instead of writing to Kafka.
type fakeProducer struct {
	mu   sync.Mutex
	cmds []Command
	fail error
}

func (p *fakeProducer) Produce(_ context.Context, _, _ string, value any) error {
	if p.fail != nil {
		return p.fail
	}

	cmd, ok := value.(Command)
	if !ok {
		return errors.New("fakeProducer: not a command")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	p.cmds = append(p.cmds, cmd)

	return nil
}

func (p *fakeProducer) byType(typ string) []Command {
	p.mu.Lock()
	defer p.mu.Unlock()

	var out []Command

	for _, c := range p.cmds {
		if c.Type == typ {
			out = append(out, c)
		}
	}

	return out
}

// countingTool records how many times it executed.
type countingTool struct {
	name  string
	calls int
	mu    sync.Mutex
	exec  func(args json.RawMessage) (json.RawMessage, error)
}

func (t *countingTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        t.name,
		Description: "test tool",
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
	}
}

func (t *countingTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	t.mu.Lock()
	t.calls++
	t.mu.Unlock()

	if t.exec != nil {
		return t.exec(args)
	}

	return json.RawMessage(`{"ok":true}`), nil
}

func (t *countingTool) count() int {
	t.mu.Lock()
	defer t.mu.Unlock()

	return t.calls
}

func newAgent(t *testing.T, model llm.Model, tools ...tool.Tool) agent.Agent {
	t.Helper()

	reg := tool.NewRegistry(tool.RegistryConfig{})
	for _, tl := range tools {
		require.NoError(t, reg.Register(tl))
	}

	ag, err := llmagent.New("test", "You are a test agent.", model,
		llmagent.WithTools(reg),
		llmagent.WithInterceptors(NewInterceptor()),
		llmagent.WithMaxTurns(5),
	)
	require.NoError(t, err)

	return ag
}

// respondText makes the fake model answer with a complete text message. The
// fake's ThenRespondText streams text deltas but ends the stream without the
// assembled message, so llmagent never appends it to the session.
func respondText(rb *fakellm.RuleBuilder, text string) {
	rb.ThenRespondWith(func(_ *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		return &llm.Response{
			Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(text)),
			FinishReason: llm.FinishReasonStop,
		}, nil
	})
}

func newRC(task Task, prod producer) *runContext {
	return &runContext{task: task, topics: Topics{}, prod: prod}
}

func baseTask() Task {
	return Task{
		TaskToken: "run-1:1",
		RunID:     "run-1",
		Agent:     "test",
		Version:   "v1",
		Attempt:   1,
		Messages:  []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("look it up"))},
	}
}

func TestExecuteAttempt_JournalsEveryMessageAndToolResult(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.LastUserMessageContains("look it up")).Times(1).
		ThenRespondWithToolCall("lookup", map[string]any{})
	respondText(model.When(fakellm.Any()), "done")

	lookup := &countingTool{name: "lookup"}
	prod := &fakeProducer{}
	rc := newRC(baseTask(), prod)

	o := executeAttempt(t.Context(), newAgent(t, model, lookup), rc)

	assert.True(t, o.completed, "outcome: %+v", o)
	assert.Equal(t, agent.FinishReasonStop, o.finishReason)
	assert.Equal(t, 1, lookup.count())

	appended := prod.byType(CommandAppendMessage)
	require.Len(t, appended, 3, "assistant tool call, tool result message, final assistant message")
	assert.Equal(t, []int{1, 2, 3}, []int{appended[0].MessageIndex, appended[1].MessageIndex, appended[2].MessageIndex})
	assert.Equal(t, llm.RoleAssistant, appended[0].Message.Role)
	assert.Len(t, appended[1].Message.ToolResponses(), 1)

	results := prod.byType(CommandRecordToolResult)
	require.Len(t, results, 1)
	assert.Equal(t, "lookup", results[0].ToolResult.Name)

	for _, c := range prod.cmds {
		assert.Equal(t, "run-1:1", c.TaskToken)
		assert.Equal(t, "run-1", c.RunID)
		assert.NotEmpty(t, c.CommandID)
	}
}

func TestExecuteAttempt_ReplaysJournaledToolResult(t *testing.T) {
	t.Parallel()

	// Session ends with an assistant tool request whose result was journaled
	// by a crashed attempt: llmagent recovers the call, the interceptor
	// returns the journaled result, and the tool does not run again.
	model := fakellm.NewFakeModel()
	respondText(model.When(fakellm.Any()), "recovered")

	lookup := &countingTool{name: "lookup"}

	task := baseTask()
	task.Attempt = 2
	task.TaskToken = "run-1:2"
	task.Messages = append(task.Messages, llm.NewMessage(llm.RoleAssistant,
		llm.NewToolRequestPart("call_9", "lookup", json.RawMessage(`{}`))))
	task.ToolResults = map[string]*llm.ToolResponsePart{
		"call_9": {ID: "call_9", Name: "lookup", Result: json.RawMessage(`{"cached":true}`)},
	}

	prod := &fakeProducer{}
	o := executeAttempt(t.Context(), newAgent(t, model, lookup), newRC(task, prod))

	assert.True(t, o.completed)
	assert.Equal(t, 0, lookup.count(), "journaled result must not re-execute the tool")

	appended := prod.byType(CommandAppendMessage)
	require.NotEmpty(t, appended)
	resps := appended[0].Message.ToolResponses()
	require.Len(t, resps, 1)
	assert.JSONEq(t, `{"cached":true}`, string(resps[0].Result))
	assert.Equal(t, 2, appended[0].MessageIndex, "recovery inserts the tool message after the existing two")
}

func TestExecuteAttempt_SuspendsOnWaitForInput(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.LastUserMessageContains("look it up")).Times(1).
		ThenRespondWithToolCall("request_approval", map[string]any{"request": "ship it?"})
	respondText(model.When(fakellm.Any()), "should not be reached")

	approve := InputTool("request_approval", "ask a human", "approval")
	prod := &fakeProducer{}

	o := executeAttempt(t.Context(), newAgent(t, model, approve), newRC(baseTask(), prod))

	require.True(t, o.suspended, "outcome: %+v", o)
	require.NotNil(t, o.awaiting)
	assert.Equal(t, AwaitInput, o.awaiting.Kind)
	assert.Equal(t, "approval", o.awaiting.Name)
	assert.Equal(t, "request_approval", o.awaiting.ToolName)
	assert.NotEmpty(t, o.awaiting.ToolCallID)

	appended := prod.byType(CommandAppendMessage)
	require.Len(t, appended, 1, "only the assistant tool request is journaled; the interrupted tool message is not")
	assert.Equal(t, llm.RoleAssistant, appended[0].Message.Role)
	assert.Empty(t, prod.byType(CommandRecordToolResult))
	assert.Equal(t, 1, model.CallCount(), "the model is not called again after a suspension")
}

func TestExecuteAttempt_ResumesWithDeliveredInput(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		last := req.Messages[len(req.Messages)-1]

		resps := last.ToolResponses()
		if len(resps) != 1 {
			return nil, errors.New("expected the delivered approval as a tool response")
		}

		return &llm.Response{
			Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("approved: "+string(resps[0].Result))),
			FinishReason: llm.FinishReasonStop,
		}, nil
	})

	approve := &countingTool{name: "request_approval", exec: func(json.RawMessage) (json.RawMessage, error) {
		return WaitForInput("approval")
	}}

	task := baseTask()
	task.Attempt = 2
	task.Messages = append(task.Messages, llm.NewMessage(llm.RoleAssistant,
		llm.NewToolRequestPart("call_1", "request_approval", json.RawMessage(`{"request":"ship it?"}`))))
	task.Delivered = map[string]json.RawMessage{"call_1": json.RawMessage(`{"approved":true}`)}

	prod := &fakeProducer{}
	o := executeAttempt(t.Context(), newAgent(t, model, approve), newRC(task, prod))

	assert.True(t, o.completed, "outcome: %+v", o)
	assert.Equal(t, 0, approve.count(), "delivered input replaces execution")

	appended := prod.byType(CommandAppendMessage)
	require.Len(t, appended, 2)
	assert.Contains(t, appended[1].Message.TextContent(), `{"approved":true}`)
}

func TestExecuteAttempt_ModelErrorIsRetryableFailure(t *testing.T) {
	t.Parallel()

	model := fakellm.AlwaysFail(errors.New("upstream 503"))
	prod := &fakeProducer{}

	o := executeAttempt(t.Context(), newAgent(t, model), newRC(baseTask(), prod))

	assert.True(t, o.failed)
	assert.True(t, o.retryable)
	assert.Contains(t, o.err, "503")
}

func TestExecuteAttempt_JournalWriteFailureIsRetryable(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	respondText(model.When(fakellm.Any()), "hello")

	prod := &fakeProducer{fail: errors.New("broker unavailable")}

	o := executeAttempt(t.Context(), newAgent(t, model), newRC(baseTask(), prod))

	assert.True(t, o.failed)
	assert.True(t, o.retryable)
	assert.Contains(t, o.err, "broker unavailable")
}

func TestExecuteAttempt_ShutdownIsInterrupted(t *testing.T) {
	t.Parallel()

	model := &blockingModel{FakeModel: fakellm.NewFakeModel()}

	ctx, cancel := context.WithCancel(t.Context())

	go func() {
		time.Sleep(50 * time.Millisecond)
		cancel()
	}()

	o := executeAttempt(ctx, newAgent(t, model), newRC(baseTask(), &fakeProducer{}))

	assert.True(t, o.interrupted, "outcome: %+v", o)
	assert.False(t, o.failed)
}

// blockingModel blocks every call until the context is cancelled, standing in
// for a slow provider during a worker shutdown.
type blockingModel struct {
	*fakellm.FakeModel
}

func (*blockingModel) Generate(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
	<-ctx.Done()

	return nil, ctx.Err()
}

func (*blockingModel) GenerateEvents(ctx context.Context, _ *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		<-ctx.Done()
		yield(nil, ctx.Err())
	}
}

func TestSuspension_ResultsAndHelpers(t *testing.T) {
	t.Parallel()

	res, err := WaitForInput("x")
	require.NoError(t, err)

	s, ok := ParseSuspension(res)
	require.True(t, ok)
	assert.Equal(t, AwaitInput, s.Kind)
	assert.Equal(t, "x", s.Name)

	at := time.Date(2030, 1, 1, 0, 0, 0, 0, time.UTC)
	res, err = SleepUntil(at)
	require.NoError(t, err)

	s, ok = ParseSuspension(res)
	require.True(t, ok)
	assert.Equal(t, AwaitTimer, s.Kind)
	assert.Equal(t, at, s.FireAt)

	_, ok = ParseSuspension(json.RawMessage(`{"ok":true}`))
	assert.False(t, ok)
	_, ok = ParseSuspension(json.RawMessage(`{"$durable_suspend":{"kind":"input"},"extra":1}`))
	assert.False(t, ok, "marker must be the only key")
	_, ok = ParseSuspension(json.RawMessage(`"text"`))
	assert.False(t, ok)

	res, err = SleepTool().Execute(t.Context(), json.RawMessage(`{"seconds": 1.5}`))
	require.NoError(t, err)

	s, ok = ParseSuspension(res)
	require.True(t, ok)
	assert.WithinDuration(t, time.Now().Add(1500*time.Millisecond), s.FireAt, 2*time.Second)
}

func TestCommandJSON_UsesSnakeCaseDiscriminators(t *testing.T) {
	t.Parallel()

	cmd := NewCommand(CommandRunSuspended, "r")
	cmd.TaskToken = "r:1"
	fire := time.Date(2026, 9, 16, 0, 0, 0, 0, time.UTC)
	cmd.Awaiting = &Awaiting{Kind: AwaitTimer, ToolCallID: "call_2", FireAt: &fire}

	b, err := json.Marshal(cmd)
	require.NoError(t, err)
	assert.Contains(t, string(b), `"type":"run_suspended"`)
	assert.Contains(t, string(b), `"awaiting":{"kind":"timer","tool_call_id":"call_2","fire_at":"2026-09-16T00:00:00Z"}`)

	var back Command
	require.NoError(t, json.Unmarshal(b, &back))
	assert.Equal(t, cmd.Awaiting, back.Awaiting)
}

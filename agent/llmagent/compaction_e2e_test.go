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

package llmagent_test

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TestCompaction_NeverOverflows is the headline property: across long runs on
// small windows, the provider is never handed a request over the window. The
// same configurations overflow without compaction
// (TestContextBudget_OverflowsWithoutCompaction).
func TestCompaction_NeverOverflows(t *testing.T) {
	t.Parallel()

	for _, window := range []int{16_000, 32_000, 200_000} {
		for _, seed := range []uint64{1, 7, 42} {
			for _, calls := range []int{1, 3} {
				t.Run(fmt.Sprintf("window_%d_seed_%d_calls_%d", window, seed, calls), func(t *testing.T) {
					t.Parallel()

					assertNeverOverflows(t, budgetConfig{
						window:       window,
						prompts:      12,
						callsPerTurn: calls,
						minTokens:    200,
						maxTokens:    2_500,
						seed:         seed,
						agentOpts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{})},
					})
				})
			}
		}
	}
}

// TestCompaction_SingleOversizedResult: a result larger than the window can
// only be truncated - the per-turn burst division caps it at collection, so
// the run survives what summarise-the-middle never could.
func TestCompaction_SingleOversizedResult(t *testing.T) {
	t.Parallel()

	assertNeverOverflows(t, budgetConfig{
		window:       8_000,
		prompts:      3,
		callsPerTurn: 1,
		minTokens:    12_000,
		maxTokens:    12_000,
		seed:         99,
		agentOpts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{})},
	})
}

// compactionAgent builds an agent + fake pair for the direct-session tests.
func compactionAgent(t *testing.T, window int, opts ...llmagent.Option) (*llmagent.LLMAgent, *fakellm.FakeModel) {
	t.Helper()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(window))
	model.When(fakellm.Any()).ThenRespondText("done")

	opts = append([]llmagent.Option{llmagent.WithMaxTurns(5)}, opts...)
	ag, err := llmagent.New("compact-agent", "You are a test assistant.", model, opts...)
	require.NoError(t, err)

	return ag, model
}

// runOnce drives one invocation and returns the events and terminal error.
func runOnce(t *testing.T, ag *llmagent.LLMAgent, sess *session.State) ([]agent.Event, error) {
	t.Helper()

	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	events := make([]agent.Event, 0, 64)

	for ev, err := range ag.Run(t.Context(), inv) {
		if err != nil {
			return events, err
		}

		events = append(events, ev)
	}

	return events, nil
}

func compactionEvents(events []agent.Event) []agent.CompactionEvent {
	var out []agent.CompactionEvent

	for _, ev := range events {
		if ce, ok := ev.(agent.CompactionEvent); ok {
			out = append(out, ce)
		}
	}

	return out
}

// applyTimeMarkerInterceptor copies and enriches the request while the model
// interceptor chain is built. A reactive retry must build the chain again or
// this transformation disappears when its Messages slice is replaced.
type applyTimeMarkerInterceptor struct {
	mu           sync.Mutex
	applications int
}

func (i *applyTimeMarkerInterceptor) InterceptModel(
	_ context.Context,
	info *agent.ModelCallInfo,
	next agent.ModelCallHandler,
) agent.ModelCallHandler {
	i.mu.Lock()
	i.applications++
	i.mu.Unlock()

	info.Req.Messages = append([]llm.Message(nil), info.Req.Messages...)
	info.Req.Messages[0] = llm.CloneMessage(info.Req.Messages[0])
	info.Req.Messages[0].Content = append(info.Req.Messages[0].Content, llm.NewTextPart("interceptor marker"))

	return next
}

func (i *applyTimeMarkerInterceptor) applicationCount() int {
	i.mu.Lock()
	defer i.mu.Unlock()

	return i.applications
}

// oldTurn appends one completed tool turn with a result of ~tokens (in the
// fake's 4-chars/token terms) to the session.
func oldTurn(sess *session.State, i, tokens int) {
	id := fmt.Sprintf("old_%d", i)

	payload := fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", tokens*4))
	sess.Messages = append(sess.Messages,
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(id, "fetch_records", json.RawMessage(`{"page":1}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(id, "fetch_records", payload, false)),
	)
}

// TestCompaction_FreshUserMessageTriggersPreCallCompaction: a session already
// near the line plus a fresh user message compacts before the first model
// call and the turn proceeds.
func TestCompaction_FreshUserMessageTriggersPreCallCompaction(t *testing.T) {
	t.Parallel()

	const window = 16_000

	ag, model := compactionAgent(t, window, llmagent.WithCompaction(llmagent.CompactionConfig{}))

	sess := &session.State{ID: "fresh"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("start")))

	for i := range 4 {
		oldTurn(sess, i, 3_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
		llm.NewTextPart(strings.Repeat("new question ", 800))))

	events, err := runOnce(t, ag, sess)
	require.NoError(t, err)

	require.NotEmpty(t, compactionEvents(events), "compaction must run before the first model call")

	calls := model.Calls()
	require.NotEmpty(t, calls)

	for _, call := range calls {
		assert.LessOrEqual(t, call.InputTokens, window)
	}
}

// TestCompaction_IrreducibleButFittingIsSent: a request between trigger and
// hardLimit that cannot reach target is sent, not rejected.
func TestCompaction_IrreducibleButFittingIsSent(t *testing.T) {
	t.Parallel()

	const window = 16_000

	ag, model := compactionAgent(t, window, llmagent.WithCompaction(llmagent.CompactionConfig{}))

	// One irreducible user message: ~13.4k estimated tokens against a 16k
	// window (usable ~11.9k... deliberately between trigger and hardLimit in
	// the agent's own high-counting terms; the fake counts ~25% lower, so the
	// provider accepts it).
	sess := &session.State{ID: "irreducible"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
		llm.NewTextPart(strings.Repeat("evidence ", 3_400))))

	_, err := runOnce(t, ag, sess)
	require.NoError(t, err, "a request that fits under hardLimit must be sent even when target is unreachable")
	require.NotEmpty(t, model.Calls())
}

// TestCompaction_CannotFitTypedError: when the minimum request exceeds the
// usable window, the failure is a typed error with the numbers - before any
// model call, with no retry loop.
func TestCompaction_CannotFitTypedError(t *testing.T) {
	t.Parallel()

	ag, model := compactionAgent(t, 2_000, llmagent.WithCompaction(llmagent.CompactionConfig{}))

	sess := &session.State{ID: "cannot-fit"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
		llm.NewTextPart(strings.Repeat("attachment ", 2_000))))

	_, err := runOnce(t, ag, sess)

	require.Error(t, err)
	require.ErrorIs(t, err, llm.ErrContextOverflow)
	assert.Contains(t, err.Error(), "cannot fit request")
	assert.Contains(t, err.Error(), "usable window")
	assert.Empty(t, model.Calls(), "the request must not reach the provider")
}

// TestCompaction_ReactiveRetry: the provider rejects a request the estimate
// thought fit; the runtime forces a strictly smaller request and retries
// exactly once.
func TestCompaction_ReactiveRetry(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(200_000))

	// First call: the production error shape for a pre-flight overflow.
	overflow := fmt.Errorf("%w: %w", llm.ErrAPICall, &llm.ProviderError{
		Base:    llm.ErrContextOverflow,
		Code:    "invalid_request_error",
		Message: "prompt is too long: 12345 tokens > 12000 maximum",
	})
	model.When(fakellm.Any()).Named("overflow-once").Times(1).ThenError(overflow)
	model.When(fakellm.Any()).ThenRespondText("recovered")

	interceptor := &applyTimeMarkerInterceptor{}
	ag, err := llmagent.New("reactive-agent", "You are a test assistant.", model,
		llmagent.WithMaxTurns(5),
		llmagent.WithCompaction(llmagent.CompactionConfig{}),
		llmagent.WithInterceptors(interceptor),
	)
	require.NoError(t, err)

	// History with prunable results, then the driving question.
	sess := &session.State{ID: "reactive"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("start")))

	for i := range 4 {
		oldTurn(sess, i, 4_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")))

	events, runErr := runOnce(t, ag, sess)
	require.NoError(t, runErr, "the run must survive one provider overflow")

	calls := model.Calls()
	require.Len(t, calls, 2, "exactly one retry")
	assert.Equal(t, 2, interceptor.applicationCount(), "each attempt rebuilds the interceptor chain")

	for _, call := range calls {
		require.NotEmpty(t, call.Request.Messages)
		assert.Contains(t, call.Request.Messages[0].TextContent(), "interceptor marker",
			"each attempt retains request-time interceptor enrichment")
	}

	assert.Less(t, calls[1].InputTokens, calls[0].InputTokens,
		"the retried request must be strictly smaller - even though the estimate already claimed it fit")

	comps := compactionEvents(events)
	require.NotEmpty(t, comps)
	assert.Equal(t, agent.CompactionPhaseReactive, comps[len(comps)-1].Report.Phase)
}

// TestCompaction_OverflowFinishReasonIsTerminalButUnwedges: a mid-generation
// overflow ends the invocation with no retry; the next invocation compacts at
// the top of the turn and proceeds (matrix item 7).
func TestCompaction_OverflowFinishReasonIsTerminalButUnwedges(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(50_000))
	model.When(fakellm.Any()).Named("overflow-once").Times(1).
		ThenRespondWith(func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("partial answer")),
				FinishReason: llm.FinishReasonContextOverflow,
			}, nil
		})
	model.When(fakellm.Any()).ThenRespondText("done")

	ag, err := llmagent.New("overflow-agent", "You are a test assistant.", model,
		llmagent.WithMaxTurns(5),
		llmagent.WithCompaction(llmagent.CompactionConfig{}),
	)
	require.NoError(t, err)

	sess := &session.State{ID: "finish-reason"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("start")))

	for i := range 6 {
		oldTurn(sess, i, 4_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("go")))

	inv := agent.NewInvocationMetadata(sess, agent.Info{})

	var end agent.InvocationEndEvent

	for ev, runErr := range ag.Run(t.Context(), inv) {
		require.NoError(t, runErr)

		if e, ok := ev.(agent.InvocationEndEvent); ok {
			end = e
		}
	}

	assert.Equal(t, agent.FinishReasonContextOverflow, end.FinishReason)
	require.Len(t, model.Calls(), 1, "a mid-generation overflow is never retried")

	// The next invocation compacts at the top of the turn and proceeds.
	// Grow the session past the trigger first so there is something to compact.
	for i := 6; i < 14; i++ {
		oldTurn(sess, i, 4_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("try again")))

	events, runErr := runOnce(t, ag, sess)
	require.NoError(t, runErr, "the session must not be wedged")
	require.NotEmpty(t, compactionEvents(events), "the next invocation compacts before its first model call")
}

// pagedTool returns a fixed-size payload per page and completes in random
// order, to prove burst division is independent of completion order.
type pagedTool struct {
	mu  sync.Mutex
	rng *rand.Rand
}

func (*pagedTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "fetch_records",
		Description: "Fetches a page of records.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"page":{"type":"integer"}}}`),
	}
}

func (p *pagedTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in struct {
		Page int `json:"page"`
	}
	_ = json.Unmarshal(args, &in)

	p.mu.Lock()
	jitter := p.rng.IntN(5)
	p.mu.Unlock()

	time.Sleep(time.Duration(jitter) * time.Millisecond)

	return json.Marshal(map[string]string{
		"data": strings.Repeat(fmt.Sprintf("page%d ", in.Page), (2_000+in.Page*500)*4/7),
	})
}

// TestCompaction_BurstDivisionDeterministic: eight parallel oversized results
// near the limit are capped identically, the request fits, and two runs with
// different completion orders produce identical request sizes.
func TestCompaction_BurstDivisionDeterministic(t *testing.T) {
	t.Parallel()

	const window = 16_000

	run := func(seed uint64) []string {
		registry := tool.NewRegistry(tool.RegistryConfig{})
		require.NoError(t, registry.Register(&pagedTool{rng: rand.New(rand.NewPCG(seed, seed))})) //nolint:gosec // jitter

		model := fakellm.NewFakeModel(fakellm.WithContextWindow(window))
		model.When(fakellm.Any()).
			ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
				if lastMessageHasToolResults(req.Messages) {
					return &llm.Response{
						Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("noted")),
						FinishReason: llm.FinishReasonStop,
					}, nil
				}

				parts := make([]llm.Part, 0, 8)
				for i := range 8 {
					parts = append(parts, llm.NewToolRequestPart(
						fmt.Sprintf("call_%d_%d", cc.TotalCalls, i), "fetch_records",
						fmt.Appendf(nil, `{"page":%d}`, i)))
				}

				return &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant, Content: parts},
					FinishReason: llm.FinishReasonToolCalls,
				}, nil
			})

		ag, err := llmagent.New("burst-agent", "You are a test assistant.", model,
			llmagent.WithTools(registry),
			llmagent.WithMaxTurns(5),
			llmagent.WithToolConcurrency(4),
			llmagent.WithCompaction(llmagent.CompactionConfig{}),
		)
		require.NoError(t, err)

		sess := &session.State{ID: fmt.Sprintf("burst_%d", seed)}
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("fetch all pages")))

		_, runErr := runOnce(t, ag, sess)
		require.NoError(t, runErr)

		calls := make([]string, 0, len(model.Calls()))

		for _, call := range model.Calls() {
			require.LessOrEqual(t, call.InputTokens, window, "burst must never assemble an unfittable frontier")

			raw, mErr := json.Marshal(call.Request.Messages)
			require.NoError(t, mErr)

			calls = append(calls, string(raw))
		}

		return calls
	}

	first := run(1)
	second := run(2) // different tool completion order via different jitter

	assert.Equal(t, first, second, "request content must not depend on tool completion order")
}

// TestCompaction_ReportsContextBreakdown: every pass appends an
// agent.CompactionReport with a per-category before/after footprint, drained
// exactly once by whoever consumes it.
func TestCompaction_ReportsContextBreakdown(t *testing.T) {
	t.Parallel()

	ag, _ := compactionAgent(t, 16_000, llmagent.WithCompaction(llmagent.CompactionConfig{}))

	sess := &session.State{ID: "report"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("start")))

	for i := range 6 {
		oldTurn(sess, i, 3_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("summarise")))

	events, runErr := runOnce(t, ag, sess)
	require.NoError(t, runErr)

	compactions := compactionEvents(events)
	require.Len(t, compactions, 1)

	rep := compactions[0].Report
	assert.Equal(t, agent.CompactionPhaseProactive, rep.Phase)
	assert.Positive(t, rep.PrunedResults)
	assert.Greater(t, rep.Before.Total, rep.After.Total)
	assert.Greater(t, rep.Before.ToolResults, rep.After.ToolResults)
	assert.Positive(t, rep.Before.SystemPrompt)
	assert.Positive(t, rep.Before.Text)
	assert.NotEmpty(t, rep.String())

	for _, u := range []agent.ContextUsage{rep.Before, rep.After} {
		sum := u.SystemPrompt + u.ToolDefinitions + u.Text + u.Reasoning + u.ToolCalls + u.ToolResults + u.Framing
		assert.Equal(t, u.Total, sum, "categories must sum to the total")
	}
}

// hugeTool returns a result far larger than any test window.
type hugeTool struct{}

func (*hugeTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "fetch_huge",
		Description: "Fetches a huge blob.",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func (*hugeTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.Marshal(map[string]string{"data": strings.Repeat("blob ", 60_000)})
}

type mediumTool struct{}

func (*mediumTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "fetch_medium",
		Description: "Fetches a medium blob.",
		Parameters:  json.RawMessage(`{"type":"object"}`),
	}
}

func (*mediumTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.Marshal(map[string]string{"data": strings.Repeat("x", 15_000)})
}

// TestCompaction_RecoveredResultIsCapped: a recovered tool result lands in the
// unread frontier, which compaction can never reduce - so recovery must apply
// the same burst budget as normal execution, or one oversized result makes
// the session permanently unfittable.
func TestCompaction_RecoveredResultIsCapped(t *testing.T) {
	t.Parallel()

	const window = 16_000

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(&hugeTool{}))

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(window))
	model.When(fakellm.Any()).
		ThenRespondWith(func(*llm.Request, *fakellm.CallContext) (*llm.Response, error) {
			return &llm.Response{
				Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("noted")),
				FinishReason: llm.FinishReasonStop,
			}, nil
		})

	ag, err := llmagent.New("recovery-agent", "You are a test assistant.", model,
		llmagent.WithTools(registry),
		llmagent.WithCompaction(llmagent.CompactionConfig{}),
	)
	require.NoError(t, err)

	// The interrupted-session shape recovery repairs: tool call, no result,
	// then a fresh user message.
	sess := &session.State{ID: "recovery_cap"}
	sess.Messages = append(sess.Messages,
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("fetch everything")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call_1", "fetch_huge", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("and summarise it")),
	)

	_, runErr := runOnce(t, ag, sess)
	require.NoError(t, runErr, "an oversized recovered result must be capped, not brick the session")

	require.NotEmpty(t, model.Calls())

	for _, call := range model.Calls() {
		assert.LessOrEqual(t, call.InputTokens, window)
	}
}

// TestCompaction_RecoveryBudgetIncludesFixedCosts verifies recovered results
// reserve room for the system prompt and tool schemas, not just history. The
// medium result fits a history-only calculation but not the complete request.
func TestCompaction_RecoveryBudgetIncludesFixedCosts(t *testing.T) {
	t.Parallel()

	const window = 16_000

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(&mediumTool{}))

	model := fakellm.NewFakeModel(fakellm.WithContextWindow(window))
	model.When(fakellm.Any()).ThenRespondText("noted")

	ag, err := llmagent.New("recovery-budget-agent", strings.Repeat("p", 27_000), model,
		llmagent.WithTools(registry),
		llmagent.WithCompaction(llmagent.CompactionConfig{}),
	)
	require.NoError(t, err)

	sess := &session.State{ID: "recovery_fixed_costs"}
	sess.Messages = append(sess.Messages,
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("fetch it")),
		llm.NewMessage(llm.RoleAssistant,
			llm.NewToolRequestPart("call_1", "fetch_medium", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("summarise it")),
	)

	_, runErr := runOnce(t, ag, sess)
	require.NoError(t, runErr)
	require.NotEmpty(t, model.Calls())

	var recovered *llm.ToolResponsePart
	for _, message := range model.Calls()[0].Request.Messages {
		for _, part := range message.Content {
			if response, ok := part.(*llm.ToolResponsePart); ok {
				recovered = response
			}
		}
	}

	require.NotNil(t, recovered)
	var marker resultMarkerView
	require.NoError(t, json.Unmarshal(recovered.Result, &marker))
	assert.True(t, marker.Truncated, "fixed request costs must force recovery-time capping")
}

type resultMarkerView struct {
	Truncated bool `json:"truncated"`
}

// TestCompaction_ConstructionValidation: each invalid configuration fails
// llmagent.New with a descriptive error (matrix item 14).
func TestCompaction_ConstructionValidation(t *testing.T) {
	t.Parallel()

	valid := fakellm.NewFakeModel(fakellm.WithContextWindow(16_000))
	windowless := fakellm.NewFakeModel(fakellm.WithContextWindow(0))

	tests := []struct {
		name    string
		model   llm.Model
		opts    []llmagent.Option
		wantErr string
	}{
		{
			name:    "trigger fraction above 1",
			model:   valid,
			opts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{TriggerFraction: 1.5})},
			wantErr: "trigger fraction",
		},
		{
			name:    "trigger fraction below the target",
			model:   valid,
			opts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{TriggerFraction: 0.5})},
			wantErr: "trigger fraction",
		},
		{
			name:    "output reserve at the window",
			model:   valid,
			opts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{OutputReserve: 16_000})},
			wantErr: "output reserve",
		},
		{
			name:    "negative tool result limit",
			model:   valid,
			opts:    []llmagent.Option{llmagent.WithToolResultLimit(-1)},
			wantErr: "tool result limit",
		},
		{
			name:    "compaction without a known window",
			model:   windowless,
			opts:    []llmagent.Option{llmagent.WithCompaction(llmagent.CompactionConfig{})},
			wantErr: "known context window",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := llmagent.New("validate-agent", "prompt", tt.model, tt.opts...)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.wantErr)
		})
	}
}

// TestCompaction_ErrorSurfaceUnchanged: with compaction on, a genuinely
// unfittable session still surfaces llm.ErrContextOverflow to the caller.
func TestCompaction_ErrorSurfaceUnchanged(t *testing.T) {
	t.Parallel()

	ag, _ := compactionAgent(t, 2_000, llmagent.WithCompaction(llmagent.CompactionConfig{}))

	sess := &session.State{ID: "surface"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
		llm.NewTextPart(strings.Repeat("x", 60_000))))

	_, err := runOnce(t, ag, sess)
	require.ErrorIs(t, err, llm.ErrContextOverflow)
	assert.ErrorIs(t, err, llm.ErrInvalidInput, "the sentinel keeps wrapping ErrInvalidInput")
}

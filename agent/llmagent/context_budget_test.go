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
	"errors"
	"fmt"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// payloadTool returns a result of a chosen size, to grow a conversation
// predictably. Sizes come from a seeded RNG so a failure is reproducible from
// the subtest name.
type payloadTool struct {
	mu        sync.Mutex
	rng       *rand.Rand
	minTokens int
	spread    int
	calls     int
}

func newPayloadTool(seed uint64, minTokens, maxTokens int) *payloadTool {
	if maxTokens < minTokens {
		maxTokens = minTokens
	}

	return &payloadTool{
		rng:       rand.New(rand.NewPCG(seed, seed^0x9e3779b9)), //nolint:gosec // reproducibility, not secrecy
		minTokens: minTokens,
		spread:    maxTokens - minTokens + 1,
	}
}

func (*payloadTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "fetch_records",
		Description: "Fetches a page of records. Returns a large result payload.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"page":{"type":"integer"}}}`),
	}
}

func (p *payloadTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	p.mu.Lock()
	p.calls++
	tokens := p.minTokens + p.rng.IntN(p.spread)
	p.mu.Unlock()

	// fakellm's tokenizer counts 4 chars per token.
	return json.Marshal(struct {
		Data string `json:"data"`
	}{strings.Repeat("record ", tokens*4/7+1)})
}

func (p *payloadTool) Calls() int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.calls
}

// budgetConfig describes one run of the harness.
type budgetConfig struct {
	// window is the model's enforced context window.
	window int

	// prompts is how many user turns to drive.
	prompts int

	// callsPerTurn is how many tool calls the model requests per turn. More than
	// one breaks a naive trigger: the results all land before the next call.
	callsPerTurn int

	// minTokens and maxTokens bound the randomised result sizes.
	minTokens, maxTokens int

	// seed makes the size sequence reproducible.
	seed uint64

	// agentOpts extends the agent configuration (e.g. compaction).
	agentOpts []llmagent.Option
}

// budgetResult is what the harness observed.
type budgetResult struct {
	runErr error

	// overflow is true when runErr was the window rejection specifically.
	overflow bool

	// inputTokens is the size of each request the model received, in order.
	inputTokens []int

	// overflowedAt is the index of the first request over the window, or -1.
	overflowedAt int

	peakTokens int
	toolCalls  int
}

// runBudget drives a conversation and reports what the model saw. The fake
// enforces the window, so an over-window request is really rejected.
func runBudget(t *testing.T, cfg budgetConfig) budgetResult {
	t.Helper()

	payload := newPayloadTool(cfg.seed, cfg.minTokens, cfg.maxTokens)

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(payload))

	model := fakellm.NewFakeModel(
		fakellm.WithContextWindow(cfg.window),
		fakellm.WithLatency(fakellm.LatencyProfile{}),
	)

	// Ask for tools, then answer once the results arrive — the shape that grows
	// a conversation.
	model.When(fakellm.Any()).
		ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
			if lastMessageHasToolResults(req.Messages) {
				return &llm.Response{
					Message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("noted")),
					FinishReason: llm.FinishReasonStop,
				}, nil
			}

			parts := make([]llm.Part, 0, cfg.callsPerTurn)
			for i := range cfg.callsPerTurn {
				parts = append(parts, llm.NewToolRequestPart(
					fmt.Sprintf("call_%d_%d", cc.TotalCalls, i),
					"fetch_records",
					fmt.Appendf(nil, `{"page":%d}`, i),
				))
			}

			return &llm.Response{
				Message:      llm.Message{Role: llm.RoleAssistant, Content: parts},
				FinishReason: llm.FinishReasonToolCalls,
			}, nil
		})

	opts := append([]llmagent.Option{
		llmagent.WithTools(registry),
		// Generous, so the turn cap is never what stops a run.
		llmagent.WithMaxTurns(50),
	}, cfg.agentOpts...)

	ag, err := llmagent.New("budget-agent", "You are a research assistant.", model, opts...)
	require.NoError(t, err)

	sess := &session.State{ID: "budget-session"}
	result := budgetResult{overflowedAt: -1}

	for i := range cfg.prompts {
		sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
			llm.NewTextPart(fmt.Sprintf("Fetch page set %d and summarise it.", i))))

		inv := agent.NewInvocationMetadata(sess, agent.Info{})

		for _, err := range ag.Run(t.Context(), inv) {
			if err != nil {
				result.runErr = err
				break
			}
		}

		if result.runErr != nil {
			break
		}
	}

	// From the recorded calls, not by recounting: Call.Request aliases the live
	// session, which compaction may rewrite.
	for i, call := range model.Calls() {
		result.inputTokens = append(result.inputTokens, call.InputTokens)

		if call.InputTokens > result.peakTokens {
			result.peakTokens = call.InputTokens
		}

		if result.overflowedAt == -1 && call.InputTokens > cfg.window {
			result.overflowedAt = i
		}
	}

	if result.runErr != nil {
		result.overflow = errors.Is(result.runErr, llm.ErrContextOverflow)
	}

	result.toolCalls = payload.Calls()

	return result
}

func lastMessageHasToolResults(messages []llm.Message) bool {
	if len(messages) == 0 {
		return false
	}

	for _, part := range messages[len(messages)-1].Content {
		if _, ok := part.(*llm.ToolResponsePart); ok {
			return true
		}
	}

	return false
}

// assertNeverOverflows is the property compaction must satisfy: across a long
// run, the provider is never handed a request larger than the window.
//
// It asserts on what the model received, not on whether the run succeeded — a
// compactor that catches rejections and retries is not the same as one that
// never overflows.
func assertNeverOverflows(t *testing.T, cfg budgetConfig) budgetResult {
	t.Helper()

	result := runBudget(t, cfg)

	require.NoError(t, result.runErr)
	require.Equal(t, -1, result.overflowedAt,
		"request %d exceeded the %d-token window", result.overflowedAt, cfg.window)
	assert.LessOrEqual(t, result.peakTokens, cfg.window)

	// A compactor that trims until the agent stops working would pass the above
	// trivially.
	assert.Positive(t, result.toolCalls, "the agent should have done real work")
	assert.GreaterOrEqual(t, len(result.inputTokens), cfg.prompts,
		"every prompt should have reached the model")

	return result
}

// TestContextBudget_OverflowsWithoutCompaction proves assertNeverOverflows can
// fail — otherwise it could pass because the conversation never grew.
func TestContextBudget_OverflowsWithoutCompaction(t *testing.T) {
	t.Parallel()

	for _, seed := range []uint64{1, 7, 42, 1234} {
		t.Run(fmt.Sprintf("seed=%d", seed), func(t *testing.T) {
			t.Parallel()

			result := runBudget(t, budgetConfig{
				window:       20000,
				prompts:      30,
				callsPerTurn: 3,
				minTokens:    200,
				maxTokens:    2500,
				seed:         seed,
			})

			require.Error(t, result.runErr, "an unmitigated conversation must be refused eventually")
			assert.True(t, result.overflow, "expected an overflow, got: %v", result.runErr)

			// It must break partway through, not on the first request.
			assert.Positive(t, result.overflowedAt, "the first request should fit")
			assert.Positive(t, result.toolCalls)

			t.Logf("overflowed at request %d, peak %d tokens, %d tool calls",
				result.overflowedAt, result.peakTokens, result.toolCalls)
		})
	}
}

// TestContextBudget_SingleOversizedResult is the case summarise-the-middle
// cannot fix: a result larger than the window can only be truncated or refused.
func TestContextBudget_SingleOversizedResult(t *testing.T) {
	t.Parallel()

	result := runBudget(t, budgetConfig{
		window:       8000,
		prompts:      3,
		callsPerTurn: 1,
		minTokens:    12000,
		maxTokens:    12000,
		seed:         99,
	})

	require.Error(t, result.runErr)
	assert.True(t, result.overflow, "got: %v", result.runErr)
	assert.Equal(t, 1, result.overflowedAt,
		"the first request fits; the one carrying the oversized result does not")
}

// TestContextBudget_StaysWithinWindowWhenItFits is the positive control: a
// harness that never grew the conversation would look like working compaction.
func TestContextBudget_StaysWithinWindowWhenItFits(t *testing.T) {
	t.Parallel()

	result := assertNeverOverflows(t, budgetConfig{
		window:       400000,
		prompts:      5,
		callsPerTurn: 2,
		minTokens:    100,
		maxTokens:    400,
		seed:         5,
	})

	assert.Positive(t, result.peakTokens)
}

// TestContextBudget_OverflowErrorSurface pins the failure a caller has to
// handle, and that it is identifiable without reading the message.
func TestContextBudget_OverflowErrorSurface(t *testing.T) {
	t.Parallel()

	result := runBudget(t, budgetConfig{
		window:       20000,
		prompts:      30,
		callsPerTurn: 3,
		minTokens:    200,
		maxTokens:    2500,
		seed:         1,
	})

	require.Error(t, result.runErr)
	require.ErrorIs(t, result.runErr, agent.ErrModelGeneration)
	require.ErrorIs(t, result.runErr, llm.ErrContextOverflow)

	// Every overflow is also an invalid input, so callers matching the broader
	// category keep working.
	require.ErrorIs(t, result.runErr, llm.ErrInvalidInput)

	var provErr *llm.ProviderError
	require.ErrorAs(t, result.runErr, &provErr)
	assert.Equal(t, "400", provErr.Code)
	assert.False(t, provErr.Retryable, "an oversized prompt will not fix itself on retry")

	// Work already done and now unrecoverable — what compaction exists to save.
	assert.Positive(t, result.toolCalls)
}

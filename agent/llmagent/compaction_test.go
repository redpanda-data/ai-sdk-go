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
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

// bigResult returns a tool result of roughly the given estimated token size.
func bigResult(id string, tokens int) *llm.ToolResponsePart {
	payload := fmt.Appendf(nil, `{"data":%q}`, strings.Repeat("x", tokens*charsPerToken))
	return llm.NewToolResponsePart(id, "fetch_records", payload, false)
}

// history builds turns of assistant tool calls answered by large results,
// ending with a fresh user question (the typical compaction input).
func history(turns, resultTokens int) []llm.Message {
	msgs := []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("start the research"))}

	for i := range turns {
		id := fmt.Sprintf("call_%d", i)
		msgs = append(msgs,
			llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(id, "fetch_records", json.RawMessage(`{"page":1}`))),
			llm.NewMessage(llm.RoleUser, bigResult(id, resultTokens)),
		)
	}

	return append(msgs, llm.NewMessage(llm.RoleUser, llm.NewTextPart("now summarise everything")))
}

func TestContextBudgetDerivation(t *testing.T) {
	t.Parallel()

	// Worked example: 200k window / 64k output.
	b := newContextBudget(200_000, 64_000, CompactionConfig{})
	assert.Equal(t, 20_000, b.reserve)
	assert.Equal(t, 180_000, b.usable)
	assert.Equal(t, 144_000, b.trigger)
	assert.Equal(t, 108_000, b.target)
	assert.Equal(t, 180_000, b.hardLimit)

	// The ordering must hold across the whole supported range.
	for _, window := range []int{8_192, 16_384, 32_768, 131_072, 200_000, 1_048_576} {
		for _, maxOut := range []int{0, 4_096, 64_000} {
			b := newContextBudget(window, maxOut, CompactionConfig{})
			assert.Positive(t, b.usable, "window %d out %d", window, maxOut)
			assert.Less(t, b.target, b.trigger, "window %d out %d", window, maxOut)
			assert.Less(t, b.trigger, b.hardLimit, "window %d out %d", window, maxOut)
			assert.LessOrEqual(t, b.hardLimit, b.usable, "window %d out %d", window, maxOut)
			assert.Equal(t, window, b.usable+b.reserve, "window %d out %d", window, maxOut)
		}
	}

	// Config overrides take effect.
	b = newContextBudget(200_000, 64_000, CompactionConfig{OutputReserve: 8_192, TriggerFraction: 0.9})
	assert.Equal(t, 8_192, b.reserve)
	assert.Equal(t, 172627, b.trigger)
}

// TestEstimateNeverUndercountsFake pins the estimator against the fake's tokenizer on
// harness-shaped payloads: never lower (counting low costs a dead session),
// and no more than 1.5x (3 vs 4 chars/token plus framing overhead).
func TestEstimateNeverUndercountsFake(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	msgs := history(10, 3_000)
	req := &llm.Request{
		Messages: msgs,
		Tools: []llm.ToolDefinition{{
			Name:        "fetch_records",
			Description: "Fetches a page of records.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"page":{"type":"integer"}}}`),
		}},
	}

	estimate := estimateHistoryTokens(msgs) + estimateToolTokens(req.Tools)
	actual := model.CountRequestTokens(req)

	assert.GreaterOrEqual(t, estimate, actual, "the estimate must never be lower than a provider's count")
	assert.LessOrEqual(t, float64(estimate), 1.5*float64(actual))
}

func TestFrontierDerivation(t *testing.T) {
	t.Parallel()

	msgs := history(2, 100) // ...assistant, user(result), user(question)
	assert.Equal(t, 4, frontierStart(msgs), "frontier begins after the last assistant message")
	assert.Equal(t, 5, newestUserIndex(msgs), "the fresh question, not the tool-result message")

	noAssistant := []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}
	assert.Equal(t, 0, frontierStart(noAssistant), "everything is frontier before the first assistant message")
}

func TestPrune_GatesAndMarker(t *testing.T) {
	t.Parallel()

	msgs := history(8, 4_000)
	before := estimateHistoryTokens(msgs)

	compacted, stats := compactMessages(msgs, 0, before/2, keepRecentResults, minTailTurns)

	require.Positive(t, stats.prunedResults)
	assert.Less(t, stats.afterTokens, stats.beforeTokens)

	// The newest keepRecentResults read results survive verbatim; older ones
	// are markers that keep id, tool name and valid JSON.
	var results []*llm.ToolResponsePart

	for _, msg := range compacted {
		for _, part := range msg.Content {
			if p, ok := part.(*llm.ToolResponsePart); ok {
				results = append(results, p)
			}
		}
	}

	for i, res := range results {
		require.True(t, json.Valid(res.Result), "result %d must stay valid JSON", i)

		var marker resultMarker
		require.NoError(t, json.Unmarshal(res.Result, &marker))

		if i < len(results)-keepRecentResults {
			if marker.Pruned {
				assert.Equal(t, "fetch_records", marker.Tool)
				assert.NotEmpty(t, marker.Preview)
				assert.Positive(t, marker.OriginalTokens)
			}
		} else {
			assert.False(t, marker.Pruned, "result %d is within keepRecentResults and must stay verbatim", i)
		}
	}
}

func TestPrune_ErrorStatusSurvives(t *testing.T) {
	t.Parallel()

	failed := bigResult("call_0", 4_000)
	failed.IsError = true

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("go")),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call_0", "fetch_records", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, failed),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("hm")),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("continue")),
	}

	// A target the prune step alone satisfies, so the message survives to be
	// inspected rather than being dropped outright.
	target := estimateHistoryTokens(msgs) - 3_000

	compacted, stats := compactMessages(msgs, 0, target, 0, 0)
	require.Positive(t, stats.prunedResults)
	require.Zero(t, stats.droppedMessages)

	res, ok := compacted[2].Content[0].(*llm.ToolResponsePart)
	require.True(t, ok)
	assert.True(t, res.IsError, "the part's error flag survives pruning")

	var marker resultMarker
	require.NoError(t, json.Unmarshal(res.Result, &marker))
	assert.Equal(t, "error", marker.Status, "the marker restates the failure")
}

func TestPrune_SmallResultsUntouched(t *testing.T) {
	t.Parallel()

	msgs := history(6, 500) // each result well below pruneAboveTokens

	_, stats := compactMessages(msgs, 0, 1, 0, 0)
	assert.Zero(t, stats.prunedResults, "results under pruneAboveTokens are not worth rewriting")
}

func TestDrop_CutRuleAndFloors(t *testing.T) {
	t.Parallel()

	msgs := history(10, 500) // unprunable, so only dropping can reduce
	total := estimateHistoryTokens(msgs)

	compacted, stats := compactMessages(msgs, 0, total/2, keepRecentResults, minTailTurns)

	require.Positive(t, stats.droppedMessages)
	require.NotEmpty(t, compacted)
	assert.False(t, hasToolResponse(compacted[0]),
		"the first retained message must not carry tool responses (orphaned pair wedges the session)")
	require.NoError(t, fakellm.ValidateConversation(compacted))

	// The tail floor holds: the last minTailTurns turns survive.
	assistants := 0

	for _, msg := range compacted {
		if msg.Role == llm.RoleAssistant {
			assistants++
		}
	}

	assert.GreaterOrEqual(t, assistants, minTailTurns)
}

func TestDrop_HardModeStillProtectsFrontierAndNewestUser(t *testing.T) {
	t.Parallel()

	msgs := history(6, 3_000)
	frontier := frontierStart(msgs)
	frontierMsgs := msgs[frontier:]

	compacted, _ := compactMessages(msgs, 0, 1, 0, 0) // unreachable target, hard floors

	require.NotEmpty(t, compacted)
	// The newest user message and the frontier survive even hard mode.
	last := compacted[len(compacted)-1]
	assert.Equal(t, frontierMsgs[len(frontierMsgs)-1], last)
	require.NoError(t, fakellm.ValidateConversation(compacted))
}

// TestDrop_PlainTextFrontierFreesPrecedingAssistant: when the frontier is
// plain text (a fresh user message after a normal answer), no tool results
// depend on the preceding assistant message, so hard mode may drop it. An
// unconditional frontier-1 bound would turn this into a spurious cannot-fit.
func TestDrop_PlainTextFrontierFreesPrecedingAssistant(t *testing.T) {
	t.Parallel()

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(strings.Repeat("question ", 500))),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart(strings.Repeat("answer ", 6_000))),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("short follow-up")),
	}

	out, stats := compactMessages(msgs, 0, 1_000, 0, 0)

	require.Len(t, out, 1)
	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, 2, stats.droppedMessages)
	require.NoError(t, fakellm.ValidateConversation(out))
}

// TestDrop_AssistantHeadGetsUserPreamble: when the cut lands on an assistant
// message (here a small surviving answer), a synthetic user message is
// prepended - providers reject a conversation that does not start with a
// user message, and the resulting 400 is not an overflow, so it would wedge
// the saved session permanently.
func TestDrop_AssistantHeadGetsUserPreamble(t *testing.T) {
	t.Parallel()

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(strings.Repeat("question ", 4_000))),
		llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("the answer")),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("follow-up")),
	}

	out, stats := compactMessages(msgs, 0, 1_000, 0, 0)

	require.Len(t, out, 3)
	assert.Equal(t, 1, stats.droppedMessages)
	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, droppedTurnsPreamble, out[0].TextContent())
	assert.Equal(t, llm.RoleAssistant, out[1].Role)
	require.NoError(t, fakellm.ValidateConversation(out))

	// The preamble counts toward the reported footprint.
	assert.Equal(t, stats.afterTokens,
		estimateHistoryTokens(out), "afterTokens must include the preamble")

	// A second pass never stacks a second preamble.
	again, statsAgain := compactMessages(out, 0, 1_000, 0, 0)
	assert.False(t, statsAgain.changed())
	assert.Equal(t, out, again)
}

// TestDrop_ToolTurnWindowStillYieldsUserHead: with tool turns pinning the
// window, the only droppable prefix ends at an assistant issuer - the cut
// must still produce a user-role head via the preamble instead of landing
// illegally or refusing to move.
func TestDrop_ToolTurnWindowStillYieldsUserHead(t *testing.T) {
	t.Parallel()

	msgs := []llm.Message{
		llm.NewMessage(llm.RoleUser, llm.NewTextPart(strings.Repeat("research ", 4_000))),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c1", "fetch_records", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", "fetch_records", json.RawMessage(`{"n":1}`), false)),
		llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("c2", "fetch_records", json.RawMessage(`{}`))),
		llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c2", "fetch_records", json.RawMessage(`{"n":1}`), false)),
		llm.NewMessage(llm.RoleUser, llm.NewTextPart("now summarise")),
	}

	out, stats := compactMessages(msgs, 0, 1_000, 0, 0)

	require.Positive(t, stats.droppedMessages)
	assert.Equal(t, llm.RoleUser, out[0].Role)
	assert.Equal(t, droppedTurnsPreamble, out[0].TextContent())
	require.NoError(t, fakellm.ValidateConversation(out))
}

// TestCompact_DoesNotMutateInput: the caller's history (and any snapshot
// aliasing its backing arrays, such as recorded fake calls) must be
// byte-identical after compaction; the rewrite happens on clones only.
func TestCompact_DoesNotMutateInput(t *testing.T) {
	t.Parallel()

	msgs := history(10, 3_000)
	before, err := json.Marshal(msgs)
	require.NoError(t, err)

	out, stats := compactMessages(msgs, 0, 5_000, 0, 0)
	require.True(t, stats.changed())
	require.Positive(t, stats.prunedResults)
	require.NoError(t, fakellm.ValidateConversation(out))

	after, err := json.Marshal(msgs)
	require.NoError(t, err)
	assert.JSONEq(t, string(before), string(after), "compaction must not mutate the caller's history")
}

func TestCompact_NoOpWhenFitting(t *testing.T) {
	t.Parallel()

	msgs := history(3, 500)
	before := estimateHistoryTokens(msgs)

	compacted, stats := compactMessages(msgs, 0, before+1, keepRecentResults, minTailTurns)
	assert.False(t, stats.changed())
	assert.Equal(t, msgs, compacted)
}

func TestCompact_Idempotent(t *testing.T) {
	t.Parallel()

	msgs := history(10, 4_000)
	target := estimateHistoryTokens(msgs) / 3

	once, statsOnce := compactMessages(msgs, 0, target, keepRecentResults, minTailTurns)
	twice, statsTwice := compactMessages(once, 0, target, keepRecentResults, minTailTurns)

	if statsOnce.afterTokens <= target {
		assert.False(t, statsTwice.changed(), "compacting twice must equal compacting once")
	}

	assert.Equal(t, once, twice)
}

// TestCompact_Property exercises randomized histories: every rewrite passes
// the provider-shape validator, never grows, and never touches the frontier.
func TestCompact_Property(t *testing.T) {
	t.Parallel()

	for _, seed := range []uint64{1, 7, 42, 99, 1234} {
		t.Run(fmt.Sprintf("seed_%d", seed), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef)) //nolint:gosec // reproducibility, not secrecy
			msgs := []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("start"))}

			turns := 3 + rng.IntN(10)
			for i := range turns {
				burst := 1 + rng.IntN(4)
				reqParts := make([]llm.Part, 0, burst+1)

				if rng.IntN(3) == 0 {
					reqParts = append(reqParts, &llm.ReasoningPart{Text: strings.Repeat("think ", 50)})
				}

				respParts := make([]llm.Part, 0, burst)

				for j := range burst {
					id := fmt.Sprintf("c_%d_%d", i, j)
					reqParts = append(reqParts, llm.NewToolRequestPart(id, "fetch_records", json.RawMessage(`{"page":1}`)))

					res := bigResult(id, 200+rng.IntN(6_000))
					res.IsError = rng.IntN(5) == 0
					respParts = append(respParts, res)
				}

				msgs = append(msgs,
					llm.Message{Role: llm.RoleAssistant, Content: reqParts},
					llm.Message{Role: llm.RoleUser, Content: respParts},
				)

				if rng.IntN(3) == 0 {
					msgs = append(msgs, llm.NewMessage(llm.RoleUser, llm.NewTextPart("interim question")))
				}
			}

			require.NoError(t, fakellm.ValidateConversation(msgs), "generated history must be valid")

			frontier := frontierStart(msgs)
			frontierCopy := make([]llm.Message, len(msgs[frontier:]))
			copy(frontierCopy, msgs[frontier:])

			before := estimateHistoryTokens(msgs)
			target := before / (2 + rng.IntN(3))

			compacted, stats := compactMessages(msgs, 0, target, keepRecentResults, minTailTurns)

			require.NoError(t, fakellm.ValidateConversation(compacted), "rewrite must stay provider-valid")
			assert.LessOrEqual(t, stats.afterTokens, stats.beforeTokens, "output never larger than input")
			assert.Equal(t, frontierCopy, compacted[len(compacted)-len(frontierCopy):], "frontier untouched")
			assert.Equal(t, llm.RoleUser, compacted[0].Role, "providers require a user-role first message")

			hard, _ := compactMessages(compacted, 0, target, 0, 0)
			assert.Equal(t, llm.RoleUser, hard[0].Role, "hard mode must also leave a user-role head")

			for _, msg := range compacted {
				require.NotEmpty(t, msg.Content, "no step may produce an empty message")

				for _, part := range msg.Content {
					if p, ok := part.(*llm.ToolResponsePart); ok {
						assert.True(t, json.Valid(p.Result), "every rewritten Result stays valid JSON")
					}
				}
			}
		})
	}
}

func TestCapToolResult(t *testing.T) {
	t.Parallel()

	small := bigResult("c1", 100)
	assert.Same(t, small, capToolResult(small, 2_000), "under the cap passes through")
	assert.Same(t, small, capToolResult(small, 0), "zero cap means uncapped")

	failed := bigResult("c2", 5_000)
	failed.IsError = true
	capped := capToolResult(failed, 1_000)

	require.NotSame(t, failed, capped)
	assert.True(t, capped.IsError, "error flag survives capping")
	assert.Equal(t, "c2", capped.ID)
	assert.Less(t, estimatePartTokens(capped), 1_000)

	var marker resultMarker
	require.NoError(t, json.Unmarshal(capped.Result, &marker))
	assert.True(t, marker.Truncated)
	assert.Equal(t, "error", marker.Status)

	tight := capToolResult(failed, 1)
	require.True(t, json.Valid(tight.Result))
	assert.JSONEq(t, `{"truncated":true}`, string(tight.Result))
	assert.Less(t, estimatePartTokens(tight), estimatePartTokens(capped),
		"tight budgets use the minimal marker")
}

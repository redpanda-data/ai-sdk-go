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
	"slices"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Compaction internals. When the counted request crosses the trigger line,
// already-read tool results are pruned to markers first, then the oldest
// turns are dropped, reducing toward target. Messages after the last
// assistant message (the unread frontier) are never touched. Every invariant
// is enforced by a test in compaction_test.go.
const (
	// keepRecentResults is how many of the newest already-read tool results
	// stay verbatim: recently read is probably still in play.
	keepRecentResults = 5

	// pruneAboveTokens gates pruning by size: results smaller than this are
	// not worth rewriting. Also what makes pruning idempotent - a marker is
	// far below this line.
	pruneAboveTokens = 2000

	// droppedTurnsPreamble stands in as the first message when the drop step
	// leaves an assistant message at the head of the history: providers
	// (Anthropic, Bedrock; Gemini for model-role turns) reject a
	// conversation that does not start with a user message.
	droppedTurnsPreamble = "(earlier turns were removed to fit the context window)"

	// minTailTurns is how many recent turns the drop step retains verbatim in
	// proactive mode. A turn is an assistant message plus what follows it.
	minTailTurns = 3

	// previewChars is how much of a pruned result's head survives in the
	// marker. A bare mask measures strictly worse than an addressable one.
	previewChars = 200
)

// resultMarker is what replaces a pruned or capped tool result. Always
// marshalled - never spliced bytes - so Result stays valid JSON. The status
// field restates the part's error flag: a failed tool pruned without its
// failure marker becomes a fabricated success in later reasoning.
type resultMarker struct {
	Pruned         bool   `json:"pruned,omitempty"`
	Truncated      bool   `json:"truncated,omitempty"`
	Tool           string `json:"tool"`
	Status         string `json:"status"`
	OriginalTokens int    `json:"original_tokens"`
	Preview        string `json:"preview"`
	Note           string `json:"note"`
}

// markerKind selects which marker shape marshalMarker renders.
type markerKind int

const (
	markerPruned markerKind = iota
	markerTruncated
)

// marshalMarker renders a marker for a result. kind selects the pruned or
// truncated shape.
func marshalMarker(part *llm.ToolResponsePart, kind markerKind) json.RawMessage {
	status := "ok"
	if part.IsError {
		status = "error"
	}

	preview := string(part.Result)
	if len(preview) > previewChars {
		preview = preview[:previewChars]
	}

	m := resultMarker{
		Tool:           part.Name,
		Status:         status,
		OriginalTokens: estimateTextTokens(string(part.Result)),
		Preview:        preview,
	}

	switch kind {
	case markerPruned:
		m.Pruned = true
		m.Note = "output removed from context to free space; re-run the tool if needed"
	case markerTruncated:
		m.Truncated = true
		m.Note = "output truncated at collection to fit the context window; re-run the tool with a narrower query if needed"
	}

	raw, err := json.Marshal(m)
	if err != nil {
		// A marker is plain strings and ints; marshalling cannot realistically
		// fail. Keep a valid-JSON fallback anyway.
		return json.RawMessage(`{"pruned":true,"note":"output removed from context to free space"}`)
	}

	return raw
}

// frontierStart returns the index of the first message of the unread
// frontier: everything after the last assistant message. The model has read
// nothing there, so nothing at or past this index may be pruned or dropped.
// Derived structurally at call time, never a tracked index -
// recoverIncompleteToolCalls inserts messages mid-history.
func frontierStart(msgs []llm.Message) int {
	for i, msg := range slices.Backward(msgs) {
		if msg.Role == llm.RoleAssistant {
			return i + 1
		}
	}

	return 0
}

// newestUserIndex returns the index of the newest user message that carries
// no tool responses - the message driving the current task - or len(msgs)
// when there is none. It is never dropped, in any mode.
func newestUserIndex(msgs []llm.Message) int {
	for i, msg := range slices.Backward(msgs) {
		if msg.Role == llm.RoleUser && !hasToolResponse(msg) {
			return i
		}
	}

	return len(msgs)
}

// tailFloorStart returns the index where the last n structural turns begin:
// the nth-from-last assistant message. Messages at or past it are protected
// from dropping in proactive mode.
func tailFloorStart(msgs []llm.Message, n int) int {
	if n <= 0 {
		return len(msgs)
	}

	seen := 0

	for i, msg := range slices.Backward(msgs) {
		if msg.Role == llm.RoleAssistant {
			seen++
			if seen == n {
				return i
			}
		}
	}

	return 0
}

func hasToolResponse(msg llm.Message) bool {
	for _, part := range msg.Content {
		if _, ok := part.(*llm.ToolResponsePart); ok {
			return true
		}
	}

	return false
}

// compactionStats reports what a compaction pass did.
type compactionStats struct {
	prunedResults   int
	droppedMessages int
	beforeTokens    int
	afterTokens     int
}

func (s compactionStats) changed() bool {
	return s.prunedResults > 0 || s.droppedMessages > 0
}

// compactMessages reduces msgs toward target and returns the rewritten slice.
// fixedTokens is the per-request cost outside the history (system prompt and
// tool schemas). keepRecent/minTail are the retention floors; hard mode
// passes zero for both so only the frontier (and the newest user message)
// stays protected.
//
// Prune first: replace already-read, non-recent, large tool results with
// markers, oldest first, stopping once under target. Then drop whole
// messages from the front using the cut rule. Compaction on a history that
// already fits is a no-op.
func compactMessages(msgs []llm.Message, fixedTokens, target, keepRecent, minTail int) ([]llm.Message, compactionStats) {
	stats := compactionStats{
		beforeTokens: fixedTokens + estimateHistoryTokens(msgs),
	}
	stats.afterTokens = stats.beforeTokens

	if stats.beforeTokens <= target {
		return msgs, stats
	}

	// Copy-on-write: the caller's slice and each message's Content may be
	// aliased by recorded calls, emitted events and retained requests, so
	// mutate only clones (Content is cloned per pruned message below).
	msgs = slices.Clone(msgs)

	counted := stats.beforeTokens
	frontier := frontierStart(msgs)

	// Prune: collect read results oldest-first, protect the newest keepRecent.
	type resultRef struct{ msg, part int }

	var read []resultRef

	for i := range frontier {
		for j, part := range msgs[i].Content {
			if _, ok := part.(*llm.ToolResponsePart); ok {
				read = append(read, resultRef{msg: i, part: j})
			}
		}
	}

	prunable := read
	if keepRecent > 0 && len(read) > keepRecent {
		prunable = read[:len(read)-keepRecent]
	} else if keepRecent > 0 {
		prunable = nil
	}

	for _, ref := range prunable {
		if counted <= target {
			break
		}

		part, ok := msgs[ref.msg].Content[ref.part].(*llm.ToolResponsePart)
		if !ok {
			continue
		}

		size := estimatePartTokens(part)
		if size <= pruneAboveTokens {
			continue
		}

		// Replace the part in a cloned Content slice, never in place: the
		// original backing array is shared with whoever else holds this
		// history (Call records, snapshots).
		replacement := &llm.ToolResponsePart{
			ID:      part.ID,
			Name:    part.Name,
			Result:  marshalMarker(part, markerPruned),
			IsError: part.IsError,
		}
		content := slices.Clone(msgs[ref.msg].Content)
		content[ref.part] = replacement
		msgs[ref.msg].Content = content

		counted -= size - estimatePartTokens(replacement)
		stats.prunedResults++
	}

	// Drop: advance the cut forward until under target or nothing more is
	// droppable. The retained head must be a plain user message: providers
	// reject a history that does not start with a user message, and a
	// tool_result whose tool_use was dropped wedges the session permanently.
	// So the cut skips tool-result messages, and an assistant head gets a
	// synthetic user preamble below - constraining the cut instead would
	// deadlock when tool turns leave only assistant candidates.
	//
	// frontier-1 bounds the cut only when the frontier carries tool results:
	// dropping the assistant that issued them would orphan them. A frontier
	// of plain text protects nothing before it - the assistant answer
	// preceding a fresh user message is droppable.
	limit := min(newestUserIndex(msgs), tailFloorStart(msgs, minTail))
	if frontier == 0 {
		limit = 0
	} else if frontierNeedsIssuer(msgs, frontier) {
		limit = min(limit, frontier-1)
	}

	preamble := llm.NewMessage(llm.RoleUser, llm.NewTextPart(droppedTurnsPreamble))
	preambleTokens := estimateMessageTokens(preamble)

	cut := 0
	droppedTokens := 0
	droppedMessages := 0

	for counted-droppedTokens > target && cut < limit {
		next := cut + 1
		for next < limit && hasToolResponse(msgs[next]) {
			next++
		}

		for i := cut; i < next; i++ {
			droppedTokens += estimateMessageTokens(msgs[i])
			droppedMessages++
		}

		cut = next
	}

	// A cut landing on an assistant message needs the preamble; when the
	// prefix it removes is smaller than the preamble itself, dropping would
	// grow the history - keep the prefix instead.
	if cut > 0 && msgs[cut].Role == llm.RoleAssistant && droppedTokens <= preambleTokens {
		cut, droppedTokens, droppedMessages = 0, 0, 0
	}

	counted -= droppedTokens
	stats.droppedMessages += droppedMessages

	if cut > 0 {
		// Clone rather than reslice so the dropped messages' payloads are
		// released instead of pinned by the shared backing array.
		msgs = slices.Clone(msgs[cut:])

		if msgs[0].Role == llm.RoleAssistant {
			msgs = append([]llm.Message{preamble}, msgs...)
			counted += preambleTokens
		}
	}

	stats.afterTokens = counted

	return msgs, stats
}

// frontierNeedsIssuer reports whether the frontier carries any tool response.
// If it does, the assistant message directly before the frontier issued those
// calls and must survive every drop, or the results would be orphaned.
func frontierNeedsIssuer(msgs []llm.Message, frontier int) bool {
	return slices.ContainsFunc(msgs[frontier:], hasToolResponse)
}

// deriveContextBudget derives the context budget. It is only called with
// compaction enabled; construction validated that the window is known.
func (a *LLMAgent) deriveContextBudget() contextBudget {
	c := a.config.model.Constraints()

	return newContextBudget(c.MaxInputTokens, c.MaxOutputTokens, *a.config.compaction)
}

// ensureFits is the top-of-turn check: when the counted request crosses the
// trigger, reduce sess.Messages toward target - proactive floors first, hard
// floors if the result still exceeds hardLimit. A request that cannot reach
// target but fits under hardLimit is sent, not rejected; the typed error is
// returned only when the request exceeds hardLimit after every safe
// reduction. fixedTokens is the system prompt plus tool schemas.
func (a *LLMAgent) ensureFits(sess *session.State, fixedTokens int) (compactionStats, error) {
	b := a.deriveContextBudget()

	stats := compactionStats{beforeTokens: fixedTokens + estimateHistoryTokens(sess.Messages)}
	stats.afterTokens = stats.beforeTokens

	if stats.beforeTokens <= b.trigger {
		return stats, nil
	}

	msgs, stats := compactMessages(sess.Messages, fixedTokens, b.target, keepRecentResults, minTailTurns)

	if stats.afterTokens > b.hardLimit {
		var hard compactionStats

		msgs, hard = compactMessages(msgs, fixedTokens, b.target, 0, 0)
		stats.prunedResults += hard.prunedResults
		stats.droppedMessages += hard.droppedMessages
		stats.afterTokens = hard.afterTokens
	}

	sess.Messages = msgs

	if stats.afterTokens > b.hardLimit {
		return stats, cannotFitError(stats.afterTokens, b)
	}

	return stats, nil
}

// reduceAfterOverflow is the reactive path: the provider just proved the
// estimate wrong with a pre-flight overflow, so the reduction is forced -
// at least 25% below the counted size at failure, hard floors, frontier
// still inviolable. Reports whether the history strictly shrank; a retry
// with an unreduced request is never acceptable.
func (a *LLMAgent) reduceAfterOverflow(sess *session.State, fixedTokens int) (compactionStats, bool) {
	b := a.deriveContextBudget()
	countedAtFailure := fixedTokens + estimateHistoryTokens(sess.Messages)
	hardTarget := min(b.target, countedAtFailure*3/4)

	msgs, stats := compactMessages(sess.Messages, fixedTokens, hardTarget, 0, 0)
	sess.Messages = msgs

	return stats, stats.afterTokens < countedAtFailure
}

// cannotFitError is the typed failure for a request whose irreducible parts
// exceed the usable window, wrapping ErrContextOverflow.
func cannotFitError(counted int, b contextBudget) error {
	return fmt.Errorf("llmagent: cannot fit request: minimum %d tokens exceeds usable window %d "+
		"(window %d, output reserve %d) - reduce attached content, lower WithToolResultLimit, "+
		"or use a model with a larger context window: %w",
		counted, b.hardLimit, b.window, b.reserve, llm.ErrContextOverflow)
}

// compactionReport assembles the observability report for one pass.
func compactionReport(phase agent.CompactionPhase, stats compactionStats, before, after agent.ContextUsage) agent.CompactionReport {
	return agent.CompactionReport{
		At:              time.Now().UTC(),
		Phase:           phase,
		PrunedResults:   stats.prunedResults,
		DroppedMessages: stats.droppedMessages,
		Before:          before,
		After:           after,
	}
}

# Context compaction — V1

Status: accepted design, V1 implemented (2026-08-21) on branch
`martin/compaction-v1`.

This is the single normative document for V1. It deliberately describes a
small, deterministic mechanism — no summarisation, no archive, no persisted
state — chosen so each later capability can be added against evidence from a
shipped baseline. Deferred features and the reason each was deferred are in §12.

Existing harness in the tree: `llm/fakellm/context_window.go` (window
enforcement) and `agent/llmagent/context_budget_test.go`
(`assertNeverOverflows`, `runBudget`). Field data came from a local
context-overflow demo replaying a real 163k-token recorded session against a
200k window.

---

## 1. The problem

`llmagent` replays the whole session on every turn. Every tool result is
appended to `session.State.Messages` and re-sent forever, so the prompt grows
monotonically (`TestContextBudget_OverflowsWithoutCompaction`). Eventually the
provider rejects the request (HTTP 400, today indistinguishable from a
malformed request: `llm.ErrInvalidInput`) or reports
`FinishReasonContextOverflow`. Either way the invocation dies mid-task and the
work already paid for is wasted.

## 2. The goal

**Prevent avoidable context overflow, and return a typed, actionable error when
the minimum valid request cannot fit.**

Not "a conversation never dies": a request whose irreducible parts (system
prompt + tool schemas + the unread frontier) exceed the usable window cannot be
sent, and the correct behaviour is a typed error naming the numbers — not
silent truncation of content the model has never seen.

## 3. Shape

- **Native in `llmagent`, not an interceptor.** Interceptors cannot yield
  events, and compaction must persist its result into the session, which is the
  agent's job. Unexported files inside `agent/llmagent` (`compact.go`,
  `budget.go`, `count.go`); no new public package. Extract a `compaction`
  package only when there is a second consumer.
- **Opt-in.** `llmagent.WithCompaction(...)`, off by default. Zero cost when
  disabled (nil check).
- **Deterministic only.** V1 makes no model calls. Every step is a pure
  rewrite, so the guarantee never depends on a summariser being reachable, and
  idempotence and testability are structural.
- **One check, before every model call.** At the top of `executeSingleTurn`,
  after the system prompt is resolved and before the request is built. That
  single position sees an oversized fresh user message (the runner appends it
  before the loop runs) and every burst of tool results (appended before the
  next call).

## 4. Budget arithmetic

Everything derives from the catalogue: `Constraints().MaxInputTokens` (the
context window) and `Constraints().MaxOutputTokens`.

```
window    = MaxInputTokens
reserve   = min(MaxOutputTokens, max(4096, window/10))   // room for the answer
usable    = window − reserve
trigger   = 0.8 × usable    // WHEN to compact: counted request crosses this
target    = 0.6 × usable    // HOW FAR to reduce once compacting
hardLimit = usable          // safety boundary: never knowingly exceed
```

200k window / 64k output → reserve 20k, usable 180k, trigger 144k, target 108k.
32k window / 4k output → reserve 4k, usable 28k, trigger 22.4k, target 16.8k.

The three lines have distinct jobs and must not be conflated:

- **`trigger`** decides *when* to act.
- **`target`** decides *how far* to reduce. The trigger−target gap (20% of
  usable) is what makes compaction rare and big-step: after each compaction
  there is ~20% of usable headroom to consume before the next one fires, so the
  prompt-prefix cache is invalidated occasionally, not per-turn. This gap
  replaces any separate "minimum reclaim" gate.
- **`hardLimit`** is the safety boundary, not a compaction goal. A request that
  cannot be reduced to `target` but still fits under `hardLimit` — for example
  an irreducible 150k frontier against a 144k trigger on a 200k window — is
  **sent, not rejected**. The typed error (§8) is returned only when the
  request exceeds `hardLimit` after every safe reduction.

`reserve` is what turns "prompt fits but there is no room for the answer" from
a mystery into arithmetic. Callers who set a provider-specific `max_tokens` in
`Request.Options` (unreadable generically — `Options` is `any`) override it via
config.

**Counting.** Heuristic, counted high on purpose: 3.0 chars/token over every
part (text, reasoning, tool arguments, tool results), a flat cost per image
part, per-message framing overhead, plus the tool schemas and the resolved
system prompt. An estimate that is low costs a dead session; one that is high
costs a slightly early compaction. The reactive path (§5.4) is the backstop for
when the estimate is still wrong. No calibration in V1.

**Internal constants** (named constants in code, not options — promote to
options only when a real caller asks):

| Constant | Value | Purpose |
|---|---|---|
| `charsPerToken` | 3.0 | count high |
| `triggerFraction` | 0.8 | of usable — when to compact |
| `targetFraction` | 0.6 | of usable — how far to reduce |
| `keepRecentResults` | 5 | newest already-read tool results kept verbatim |
| `pruneAboveTokens` | 2,000 | results smaller than this are not worth rewriting |
| `minTailTurns` | 3 | verbatim recent turns the drop step retains (proactive mode) |

**A "turn" is defined structurally:** an assistant message together with the
messages that follow it up to the next assistant message. `minTailTurns = 3`
keeps the last three such turns intact. The frontier — everything after the
*last* assistant message — is protected separately and absolutely (§5.0).

## 5. The algorithm

When the counted request crosses `trigger`: prune toward `target`, re-count,
drop if still over, re-count. Send if the result is under `hardLimit` (even
when `target` was unreachable); return the typed error only when it is not.

### 5.0 The unread frontier (inviolable)

**The frontier is every message after the last assistant message** — the tool
responses answering that assistant's calls, and any user message the runner
appended. The model has not read any of it. Pruning or dropping it means the
agent paid for work whose answer it never learns, silently. Nothing in V1 may
touch the frontier; if the frontier alone cannot fit under `hardLimit`, that is
the typed error (§8), not a truncation.

The frontier is **derived structurally at check time**, never a tracked index:
`recoverIncompleteToolCalls` inserts messages mid-history, so any stored index
can shift. The "after the last assistant message" rule is immune to that.

The frontier is kept affordable *by construction* through the per-call result
cap (§6): the budget a turn's results may occupy is divided across the calls
before any tool runs, so a parallel burst cannot assemble an unfittable
frontier in the first place.

### 5.1 Prune already-read tool results (primary)

Tool observations dominate an agent's context (~84% of turn tokens in measured
trajectories; see appendix) and most are re-derivable — the agent can call the
tool again. For every `ToolResponsePart` that is

- **not** in the frontier (the model has read it), and
- **not** among the newest `keepRecentResults` read results (recently read is
  probably still in play), and
- larger than `pruneAboveTokens`,

replace `Result` with a **marshalled marker object** — never spliced bytes;
`Result` is `json.RawMessage` and must remain valid JSON — preserving the part,
its `ID` and its pairing:

```json
{"pruned": true, "tool": "fetch_records", "status": "error",
 "original_tokens": 12943,
 "preview": "region=eu-central-1, 412 rows, first id 90c1…",
 "note": "output removed from context to free space; re-run the tool if needed"}
```

- The part's error flag survives and `status` restates it: a failed tool pruned
  without its failure marker becomes a fabricated success in later reasoning.
- `preview` is the head of the original payload (~200 chars). A bare mask with
  no preview measured strictly worse than an addressable one; the preview is
  the cheap part of that result.
- The word is **pruned**, not archived: the bytes are gone. No id promising a
  recall that does not exist. (An archive + `recall` tool is the V2 upgrade.)
- Pruning is idempotent by size: a marker is far below `pruneAboveTokens`.

Pruning operates on **parts, not messages**, which is what makes the parallel
burst tractable: `executeTools` packs all of a turn's results into one
`RoleUser` message, so five 25k results are a single 125k message that no
message-level rule can trim.

### 5.2 Drop oldest messages (fallback)

If still over `target` after pruning, drop whole messages from the front using
the cut rule, until under `target` or nothing more is droppable. Never drop
into the frontier, the newest user message, or (in proactive mode) the last
`minTailTurns` turns.

**The cut rule:** advance the cut index forward until `messages[cut]` contains
no `ToolResponsePart`. A retained `tool_result` whose `tool_use` was dropped is
a hard 400 on every subsequent request — a permanently wedged session. An
assistant message carrying `tool_use` is a legal first retained message: its
results follow it. Forward, not backward, so the retained tail can never grow
past its budget.

Dropped user turns are lost — V1 has no summary to represent them. This is the
deliberate, documented cost of a deterministic V1, and the first thing V2
addresses (§12). In practice pruning does the bulk of the work and dropping is
rare; the drop step is why the guarantee holds when it isn't.

### 5.3 Rewrite in place

`sess.Messages` is rewritten directly — pruned parts and dropped messages are
gone from the session. **What is in the session is what gets sent**: no derived
view, no second prompt-assembly path, no stored offsets.

The SDK forwards events; it does not persist them. **Applications that need an
audit transcript must persist the event stream themselves, before enabling
compaction.** Events carry content as the runtime recorded it — in particular,
`ToolResponseEvent` carries the *capped* result (§6), the same bytes the model
will see; an application that needs full tool outputs must capture them at the
tool layer. One truth: session, events and prompt never diverge.

### 5.4 Reactive path

Our count is an estimate; providers count differently. The reactive retry
applies to exactly one signal: **a model call that fails pre-flight with
`llm.ErrContextOverflow`** (WP0) — an error, delivered before any
content, so nothing has been emitted to the consumer.

1. Re-run prune + drop in **hard mode**:
   - target `min(target, 0.75 × counted size at failure)` — the provider just
     proved the estimate wrong, so the reduction must be *forced*: even if the
     estimate claims the request already fits, shrink it by at least 25%.
     Retrying an identical request is never acceptable.
   - `keepRecentResults` reduced to 0 and `minTailTurns` relaxed to 0 — only
     the frontier stays protected, so the minimum request is actually minimal.
2. Retry the call **once**.
3. If it overflows again, fail with the typed error.

`FinishReasonContextOverflow` is deliberately **not** retried in V1: the
response may already carry delivered content (`llm/types.go` is explicit that
content produced before the limit is still on the response), streaming emits
deltas immediately, and `executeSingleTurn` appends and emits the
`MessageEvent` before examining the finish reason — a retry would duplicate
output. The invocation ends with that finish reason as it does today; the
session is *not* wedged, because the next invocation's top-of-turn check
compacts it before the first model call.

### Observability

Each pass yields a typed `agent.CompactionEvent` **after the rewrite**,
carrying a `CompactionReport`: phase (proactive or reactive), pruned and
dropped counts, and a before/after context breakdown by category in
conservative heuristic estimates. `Report.String()` renders the one-line summary
(`"pruned 14 results, dropped 6 messages, 158k -> 96k tokens"`). (Compaction
is a deterministic rewrite measured in microseconds; a "starting…" event has
nothing to report and is omitted.)

Interceptors can implement the observe-only `agent.EventObserver` interface
to see every event on the yield path. The otel plugin uses it to render each
compaction as a zero-duration `redpanda.compaction` span and stamps
`gen_ai.conversation.compacted` on chat spans thereafter.

## 6. Public API

The entire new public surface:

```go
// llm
var ErrContextOverflow error   // wraps ErrInvalidInput; mapped per provider

// agent
type CompactionPhase string
const (
    CompactionPhaseProactive CompactionPhase = "proactive"
    CompactionPhaseReactive  CompactionPhase = "reactive"
)
type ContextUsage struct { ... }     // per-category token footprint
type CompactionReport struct { ... } // phase, counts, Before/After ContextUsage
type CompactionEvent struct { ... }  // Envelope + Report; on the event stream

type EventObserver interface {       // optional interceptor interface
    ObserveEvent(ctx context.Context, inv *InvocationMetadata, event Event)
}
func ApplyEventObservers(...)        // yield-wrapping chain helper

// llmagent
func WithCompaction(cfg CompactionConfig) Option

type CompactionConfig struct {
    // OutputReserve overrides the derived answer-room reservation.
    // 0 = min(MaxOutputTokens, max(4096, window/10)).
    OutputReserve int
    // TriggerFraction of the usable window at which compaction runs.
    // 0 = 0.8.
    TriggerFraction float64
}

func WithToolResultLimit(tokens int) Option  // per-result cap at collection time
```

A config struct rather than functional options so fields can be added without
breaking callers, and so the signature survives a later extraction of the
implementation into its own package.

**Per-result cap.** `WithToolResultLimit` is prevention upstream, applied when
`executeTools` collects results. A capped result is replaced by a marshalled
marker object of the same shape as §5.1 (with `"truncated": true` and the
preview) — never spliced bytes, so `Result` stays valid JSON, and the error
flag survives (I8). Useful with or without compaction.

**Deterministic burst division.** When compaction is enabled, the effective
per-result cap for a turn is computed *before any tool runs* — the requested
calls are all known at that point:

```
availableForResults = hardLimit − countedRequestSize
effectiveCap        = max(markerFloor, min(configuredCap, availableForResults / numberOfCalls))
```

The same cap applies to every result in the turn **independently of completion
order**, so runs are reproducible even though tools execute concurrently.
`markerFloor` guarantees room for at least the marker object. This is what
keeps a parallel burst — eight 25k results is ~200k before schemas — from
assembling an unfittable frontier that compaction is (correctly) forbidden to
touch. It is internal behaviour, not an option.

**Configuration validation.** `llmagent.New` fails fast, at construction:
`TriggerFraction` outside `(0.6, 1)`; `OutputReserve` ≥ the model's window;
`WithToolResultLimit` < 0; and compaction enabled on a model whose catalogue
reports no context window (`MaxInputTokens` zero or unknown) — compaction
cannot budget against a window it does not know.

## 7. Invariants

Each is a test, not a comment.

| | Invariant |
|---|---|
| I1 | **Frontier.** No message after the last assistant message is pruned or dropped. |
| I2 | **Pairing.** The first retained message never contains a `ToolResponsePart`; no `tool_use` is retained without its result except as the final message. Every rewrite passes the fakellm conversation validator. |
| I3 | **No empty content, no invalid JSON.** No step produces a message with zero parts. Pruning and capping replace `Result` with a marshalled marker object, never remove parts, never splice bytes. |
| I4 | **Progress.** Compaction reduces toward `target`. If `target` is unreachable, a request ≤ `hardLimit` is sent anyway. The typed error is returned only when the request exceeds `hardLimit` after every safe reduction (hard mode: tail floor and recency protection relaxed, frontier never). Hard mode strictly reduces the counted size — an identical retry is impossible. Never an unbounded loop. |
| I5 | **Session equals prompt.** `sess.Messages` is what gets sent. No hidden index, no second assembly path. |
| I6 | **Idempotence.** Compaction on a history that already fits is a no-op. Compacting twice equals compacting once. |
| I7 | **Cache stability.** Two consecutive turns without compaction produce byte-identical prompt prefixes; the trigger−target gap keeps compactions rare and big-step. |
| I8 | **Error status survives.** A pruned or capped result that was an error is still identifiable as an error. |
| I9 | **Determinism.** Given the same history, configuration and tool results, compaction and burst division produce identical output regardless of tool completion order. |

## 8. Failure behaviour

When system prompt + tool schemas + the frontier exceed `hardLimit` even after
hard-mode compaction (everything already-read pruned and dropped):

```
llmagent: cannot fit request: minimum 34120 tokens exceeds usable window 28000
(window 32000, output reserve 4000) — reduce attached content, lower
WithToolResultLimit, or use a model with a larger context window
```

A typed error (wrapping `llm.ErrContextOverflow`), with the numbers and
the available fixes. Silently truncating unread content is worse than saying it
does not fit. Note the boundary is `hardLimit`, not `trigger`: a request that
merely cannot reach the compaction target still gets sent (§4).

## 9. Session contract change

`store/session/store.go` documents `State.Messages` as "should be treated as
append-only". The runtime already violates this (`recoverIncompleteToolCalls`
inserts a repair message mid-history; the empty-content guard skips appends),
and compaction rewrites it by design. V1 amends the doc comment: **`Messages`
is the model's working context, owned and maintained by the runtime — not an
audit transcript.** Applications that need a transcript must persist the event
stream themselves (§5.3); the SDK forwards events but does not store them.

## 10. Work packages

Each independently reviewable; WP0–WP2 independently mergeable.

- **WP0 — typed overflow error. DONE (2026-08-21).** `llm.ErrContextOverflow`
  (named for symmetry with `FinishReasonContextOverflow`) wrapping
  `ErrInvalidInput`; mapped in
  `providers/{anthropic,openai,openaicompat,bedrock,google}/errors.go`
  (Anthropic `prompt is too long`, OpenAI `context_length_exceeded`, Bedrock
  `ValidationException` on input length, Google `input token count exceeds`).
  Rate-limit payloads do not match; oversized max_tokens stays plain
  `ErrInvalidInput`. Every mapping validated against live provider APIs and
  covered by `providers/conformance` `TestContextOverflow` (integration) plus
  per-provider unit tests with recorded payloads.
- **WP1 — conversation validator. DONE.** `llm/fakellm`: validate every request's
  shape as a provider would (orphaned `tool_result`, `tool_use` without a
  following result, empty content, non-JSON `Result` payloads). Prerequisite
  for the pairing property tests; does not validate role alternation
  (provider-specific).
- **WP2 — counting + budget. DONE.** Internal to `llmagent`. Parity with
  `fakellm.CountRequestTokens` within 15% on the harness payloads; the budget
  table (§4) asserted over windows 8k–1M, including
  `target < trigger < hardLimit ≤ usable`.
- **WP3 — prune step. DONE.** Part-level marker rewrite, frontier/recency/size
  gates, idempotence, marker validity.
- **WP4 — drop step. DONE.** Cut rule + retention floors (and their hard-mode
  relaxation); property-tested against the WP1 validator on randomized
  histories.
- **WP5 — wiring. DONE.** Top-of-turn check, reactive retry, burst division in
  `executeTools`, `CompactionEvent`, `WithCompaction`,
  `WithToolResultLimit`, construction-time validation. The only
  behaviour-changing PR, off by default.
- **WP6 — docs. DONE.** Session-contract comment (§9), package docs, this
  design doc. The context-overflow demo used for field data stays local.

## 11. Test matrix

1. `assertNeverOverflows` with compaction on: windows 16k/32k/200k × several
   seeds × {1, 3} tool calls per turn — the provider is never handed a request
   over the window.
2. `TestContextBudget_SingleOversizedResult` passes with
   `WithToolResultLimit` + compaction.
3. Fresh user message larger than remaining space → compaction runs before the
   first model call; turn proceeds.
4. Irreducible-but-fitting: a request between `trigger` and `hardLimit` that
   cannot reach `target` is **sent**, not rejected.
5. Burst division: a turn requesting 8 parallel oversized results near the
   limit → every result capped identically, request fits, byte-identical
   output across shuffled completion orders (I9).
6. Reactive: fakellm window smaller than advertised → hard mode, one retry,
   run survives. Assert the retried request is strictly smaller than the
   failed one — including when the failed estimate was already under
   `hardLimit`.
7. Overflow finish reason mid-generation → invocation ends terminally with no
   retry and no duplicated events; the *next* invocation compacts at the top
   of the turn and proceeds (session not wedged).
8. Minimum request cannot fit → typed error with numbers; no retry loop.
9. Property test: for randomized histories (parallel bursts, reasoning parts,
   interleaved user turns), every rewrite passes the WP1 validator; output
   never larger than input; frontier untouched; every rewritten `Result` is
   valid JSON.
10. Idempotence: compacting a fitting history is a no-op; compact twice ==
    compact once.
11. Error status: a pruned or capped error-result still reads as an error.
12. Cache stability: consecutive under-trigger turns produce byte-identical
    prefixes.
13. Save/load round-trip mid-session: pruned stays pruned, no re-compaction,
    no state to corrupt.
14. Construction-time validation: each invalid configuration from §6 fails
    `llmagent.New` with a descriptive error.

## 12. Deferred, with reasons

Ordered by expected V2 priority:

- **Summarisation + pinned context.** The V1 drop step loses old user turns
  outright; a summary (separate request, own system prompt, no tools, output
  capped in code) plus `WithPinnedContext` for standing rules is the first
  upgrade. Deferred because it introduces a model call into the guarantee
  path, a prompt to tune, and persisted summary state — and the measurements
  say pruning does most of the work anyway.
- **Archive + `recall` tool.** Turns pruning from lossy to reversible.
  Deferred: needs a storage interface with session/tenant namespacing and a
  retention story; must not live in session metadata (the runner re-saves the
  whole record every assistant message).
- **Identifier ledger.** Mechanical harvest of tool calls/ids from dropped
  regions. Deferred: free-text identifier scraping is heuristic and can pin
  secrets into every subsequent prompt; revisit scoped to structured tool
  arguments.
- **Calibration (on provider-billed tokens). Built, then removed.** Field
  data (real webfetch content, ~2.34 chars/token) showed the count-high
  heuristic undercounting on token-dense content, so a billed/estimated
  ratio briefly scaled the budget lines. It was removed: the moving scale
  made every budget decision history-dependent and hard to reason about,
  and the reactive overflow path already backstops a wrong estimate.
  Revisit only with field data showing the reactive path firing often.
- **Thrash detector + circuit breaker.** Deferred until compaction can fire
  often enough to thrash — the trigger−target gap makes back-to-back
  compaction rare by construction; revisit with field data.
- **Deferral inside tool chains** (compact at sub-task boundaries between
  trigger and hardLimit). Real measured win, small mechanism — first candidate
  once V1 has soak time.
- **Reactive retry for `FinishReasonContextOverflow`.** Requires proving no
  delta or message was emitted for the call (a code reorder in
  `executeSingleTurn` plus a streaming guard); V1 keeps it terminal.
- **Package extraction, provider-side compaction, async compaction, retrieval
  over archives, external memory.** Each waits on a concrete consumer.

---

## Appendix: evidence (condensed, 2026-08)

From the 2025–2026 agent context-management literature and a source-level
survey of Claude Code v2.1.237, Codex, OpenCode, Cline, Gemini CLI and Crush.
The numbers that shaped V1:

- **Tool observations are ~84% of turn tokens** in measured SWE-agent
  trajectories; mechanically masking old ones matched or beat LLM
  summarisation on solve rate in 4 of 5 configurations at roughly half the
  cost. LLM condensers increased total token cost 24–94% in a second study and
  summarised runs were 13–15% longer. → prune-first, no summariser in V1.
- **Every shipping agent reserves output room** (Claude Code
  `window − min(max_out, 20k) − 13k`; Codex 5%; OpenCode `min(20k, max_out)`)
  and none uses a cheap model when they do summarise. → §4 arithmetic.
- **Fire-high, reduce-low is the shipping pattern**: Cline fires at ~81% of
  the window and reduces to ~63%. A trigger that doubles as the target
  compacts every turn once the session hovers at the line. → separate
  `trigger` and `target`.
- **Splitting a tool pair wedges the session permanently** (400 on every
  subsequent request). Cline enforces the safe-cut predicate three separate
  ways; Gemini CLI's single safeguard is the same predicate. → the cut rule
  and the WP1 validator.
- **Crush evaluates its trigger after the step completes**, so a turn that
  trips it has its tool results computed, charged and discarded. → the unread
  frontier rule and the pre-call check position.
- **Count high**: Cline divides characters by 3, not 4, explicitly so
  thresholds fire before provider rejection. Codex and OpenCode omit tool
  schemas from the driving count and both carry overflow bugs. → 3.0
  chars/token, schemas and system prompt included.
- **Rewrites invalidate the prompt-prefix cache** (cached input ≈ 10× cheaper);
  Cline gates pruning on 64KB reclaimable for this reason; Anthropic's
  context-editing API exposes `clear_at_least`. → rare, big-step compaction
  via the trigger−target gap.
- **88–100% of compaction-induced errors first manifest within 2–3 steps.**
  → `minTailTurns = 3` and `keepRecentResults = 5` (Claude Code keeps 5;
  Anthropic's context-editing default is 3).
- **A bare mask is strictly worse than an addressable one** (99.4% vs 88.1%
  needle-in-trajectory recall for archive+recall vs the best baseline). V1
  keeps the cheap half (a preview in the marker); the archive is the named V2
  upgrade.
- **Reactive recovery should not trust the estimator that just failed**: Cline
  forces its deterministic strategy on overflow recovery; Claude Code allows
  one reactive attempt per turn. → deterministic-only hard mode with a forced
  minimum reduction, one retry.

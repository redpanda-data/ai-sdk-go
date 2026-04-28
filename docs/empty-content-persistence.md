# Empty-content responses: open design question

Status: parked. PR #123 ships the minimal fix — agent-loop guard at the
session-store boundary. The deeper question of where the invariant
should live is not resolved.

## The bug

After PR #116 landed (drop partial `tool_use` blocks at stream
finalisation), an unrelated wedge surfaced in production: agent
sessions started failing every call with

```
messages.15.content: Field required
```

from Anthropic. The cause: a previous turn finalised with `Content =
[]` (max_tokens hit before any non-partial block was emitted, or the
only block was a partial tool_use that #116 correctly dropped, or a
refusal with no text). The agent loop's `sess.Messages = append(...)`
ran *before* the FinishReason terminal-handling check, so the empty
`Message` landed in session state. Every subsequent replay then sent
that empty array to Anthropic, which 400'd. Pod restarts didn't help —
session was the persistent thing.

## What we shipped (PR #123)

One-line guard at `agent/llmagent/llmagent.go:273`:

```go
if len(resp.Message.Content) > 0 {
    sess.Messages = append(sess.Messages, resp.Message)
}
```

The MessageEvent still fires; the FinishReason still propagates; the
terminal-reason handling still terminates the loop cleanly. Only
persistence is skipped.

## What we considered and rejected

### A. Provider returns error on empty content

Move the check into each `response_mapper`: if `len(content) == 0`,
return `ErrResponseMapping`. The agent loop's existing terminal-error
path then fires (`generate()` returns err, loop exits cleanly).

Rejected because it swallows the FinishReason signal. `Length + empty`
is a real, valid state (the model ran out of budget before producing
anything) — surfacing it as a generic mapping error loses the
information the caller needs to decide whether to retry with higher
`max_tokens`. Same for `ContentFilter + empty` (refusal). The agent
loop's existing handling for these reasons is correct and should keep
firing.

### B. Provider error only on `Stop + empty`

Conditional version of (A): only error when the stop reason is `Stop`
(end_turn / clean stop) with empty content, since *that* is the
genuinely anomalous case. Length and ContentFilter with empty content
flow through normally.

Rejected as speculative paranoia: `Stop + empty` is theoretical and
hasn't been observed in the wild. Adding code for a phantom case adds
complexity without value. If it ever happens we can revisit.

### C. Add a `Partial bool` field to `llm.Response`

Mirror adk-go's `Event.Partial` flag. Providers set it when the
response is a non-result; the agent loop respects the flag at the
persistence boundary.

Rejected as redundant. adk-go needs Partial because they expose
streaming chunks at the API surface — they have to distinguish
"chunk-with-content" from "complete." This SDK's agent loop sees only
the final aggregated `Response`, so `Partial == (len(Content) == 0)`
in our architecture. The flag adds ceremony without distinguishing
power.

### D. Key persistence on FinishReason instead of content length

"Skip persistence when FinishReason == Length."

Rejected because Length with non-empty content is valid and worth
persisting — e.g., the model completed 3 of 4 parallel tool calls
before hitting `max_tokens` (PR #116's central scenario). The caller
wants those 3 completed tool_use blocks to inspect. Length alone
isn't synonymous with "unusable result"; `len(Content) == 0` is.

## The open question

The session-store boundary is the *right place* to honour "non-result
responses don't get persisted." That part is settled. What's not
settled:

1. **Should providers also be involved?** Right now, the invariant is
   enforced only in `llmagent.go`. A non-llmagent caller that drives
   `model.Generate()` directly and persists the result has to know to
   apply the same guard. That's not great encapsulation. Options:

   - Document the invariant: "callers must skip persistence when
     `len(resp.Message.Content) == 0`." Cheapest. Relies on docs.
   - Add a helper: `resp.IsPersistable() bool` returning
     `len(Content) > 0`. Self-documenting at call sites.
   - Move the guard into the session store itself (any
     `sess.Append(msg)` helper rejects empty Content). Requires
     introducing a session abstraction the SDK doesn't currently
     own at this layer.

2. **Should we have telemetry?** Right now empty-content responses
   silently disappear into the void. A counter/log would tell us how
   often this fires and whether some provider/scenario is leaking
   them at an unexpected rate. Cheap to add, useful for spotting
   regressions.

3. **Is there a class of "incomplete" responses that don't have
   `len(Content) == 0` but still shouldn't be persisted?** PR #116's
   reproducer (truncated tool args yielding `unexpected end of JSON
   input`) was such a case before the fix. Are there other shapes we
   haven't seen yet? Hard to know without more production exposure.

## Move on

For now: PR #123 fixes the production bug. The architectural
improvements above are not blocking. Revisit when:

- Telemetry shows empty-content responses are happening at a
  non-trivial rate, or
- A non-llmagent caller hits the same wedge, or
- A new "incomplete" response shape surfaces that the simple
  `len(Content) == 0` check doesn't catch.

## References

- PR #116 — surface truncation instead of corrupting state when
  tool_use is cut off (the sibling fix)
- PR #123 — the agent-loop guard shipped here
- adk-go `session/inmemory.go::AppendEvent` — prior art for
  guarding the persistence boundary on a "not-final" signal

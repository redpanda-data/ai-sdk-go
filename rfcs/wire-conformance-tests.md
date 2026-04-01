# RFC: Wire Conformance Tests

## Problem

The existing conformance test suite validates behavior: "given this `llm.Request`, does the
provider return a sane `llm.Response`?" This catches high-level breakage but is blind to
silent wire-level regressions. If a request mapper drops a field, sends a wrong type, or
omits a header, the model usually still returns *something* -- the test passes, the bug
ships.

Examples of bugs this misses:

- `cache_control` markers silently dropped -- requests work, but caching never activates
  and the user pays full price.
- `tool_choice` mapped as `"auto"` when the user asked for `"required"` -- model still
  calls tools most of the time, behavioral test passes by luck.
- Provider-specific headers missing (e.g., `anthropic-beta`) -- API falls back to default
  behavior, which happens to work today but may not tomorrow.
- New fields added to native SDK (e.g., a new `metadata` or `reasoning` parameter) that
  our mapper never learned about -- no error, just silent feature loss.

## Solution

Add a second conformance axis: **wire conformance**. For each provider, run the same
logical operation two ways:

1. **Native SDK** -- use `anthropic-sdk-go` / `openai-go` / `google.golang.org/genai`
   directly.
2. **ai-sdk-go provider** -- use our `providers/anthropic` / `providers/openai` / etc.

Both go through a **recording HTTP transport** that captures the outgoing request. Then
**structurally diff the two request bodies** and fail with exact JSON paths if they
diverge.

## Design principles

- **No API calls.** The recording transport returns canned responses. Wire tests are free,
  fast, and deterministic. They run in CI without API keys.
- **Native SDK is the oracle.** We don't maintain hand-written expected JSON. The native
  SDK *is* the spec. When it changes serialization, we track it automatically.
- **Agent-fixable output.** Test failures print the exact JSON path, the native value, the
  ai-sdk value, and which file to fix. A coding agent can resolve the diff without reading
  API documentation.

## Architecture

```
providers/
  wireconformance/
    recorder.go        -- recording http.RoundTripper (MITM transport)
    differ.go          -- structural JSON diff engine, path-based output
    normalize.go       -- normalization rules (ignore, sort, strip)
    suite.go           -- generic wire conformance suite (testify suite)
    fixture.go         -- WireFixture interface
    report.go          -- human/agent-readable failure formatter
    canned/            -- minimal valid response bodies per provider
```

Each provider adds a wire test file:

```
providers/anthropic/anthropic_wire_test.go
providers/openai/openai_wire_test.go
providers/google/google_wire_test.go
```

## Key types

### Recording transport

```go
// RecordingTransport is an http.RoundTripper that captures request bodies
// and returns canned responses. No real network calls.
type RecordingTransport struct {
    // CannedResponse is returned for every request.
    CannedStatusCode int
    CannedBody       []byte
    CannedHeaders    http.Header

    // Captured holds all recorded exchanges after the test runs.
    Captured []CapturedExchange
}

type CapturedExchange struct {
    Method      string
    URL         string
    RequestBody json.RawMessage
    Headers     http.Header
}
```

The transport reads and buffers the request body, records it, and returns the canned
response. No network involved.

### Wire fixture

```go
type WireFixture interface {
    Name() string
    Scenarios() []WireScenario
}

type WireScenario struct {
    Name string

    // NativeCall makes the API call using the native provider SDK,
    // configured with the recording transport.
    NativeCall func(t *testing.T, transport *RecordingTransport) error

    // SDKCall makes the equivalent call using ai-sdk-go,
    // configured with the recording transport.
    SDKCall func(t *testing.T, transport *RecordingTransport) error

    // NormalizeRules are provider/scenario-specific rules applied
    // before diffing. Merged with default rules.
    NormalizeRules []NormalizeRule
}
```

### Differ

```go
type FieldDiff struct {
    Path     string          // e.g. "messages[0].content[1].cache_control"
    Expected json.RawMessage // from native SDK call
    Actual   json.RawMessage // from ai-sdk-go call
    Kind     DiffKind        // Missing, Extra, Changed, TypeMismatch
}

type DiffKind int

const (
    DiffMissing      DiffKind = iota // field in native, absent in ai-sdk
    DiffExtra                        // field in ai-sdk, absent in native
    DiffChanged                      // both present, values differ
    DiffTypeMismatch                 // both present, JSON types differ
)

func DiffJSON(native, aisdk json.RawMessage, rules []NormalizeRule) []FieldDiff
```

### Normalization

```go
type NormalizeAction int

const (
    Ignore         NormalizeAction = iota // drop field before comparison
    SortArray                             // sort array elements before comparison
    StripWhitespace                       // normalize whitespace in strings
)

type NormalizeRule struct {
    Path   string          // JSON path pattern, supports wildcards (e.g. "*.metadata")
    Action NormalizeAction
}
```

Default rules applied to all providers:

```go
var defaultRules = []NormalizeRule{
    {Path: "stream", Action: Ignore},            // streaming flag expectedly differs
    {Path: "stream_options", Action: Ignore},     // streaming config
    {Path: "metadata", Action: Ignore},           // SDK-injected metadata
}
```

Provider-specific rules handle legitimate differences, e.g.:

- Anthropic: ignore `anthropic-version` header value (SDK pins its own version)
- OpenAI: ignore `stream_options.include_usage` (our SDK may set this unconditionally)
- Google: ignore request URL path differences (SDK versions may vary)

## Test execution flow

For each `WireScenario`:

1. Create two `RecordingTransport` instances with the same canned response.
2. Run `NativeCall(t, nativeTransport)` -- captures what the native SDK sends.
3. Run `SDKCall(t, sdkTransport)` -- captures what ai-sdk-go sends.
4. Assert both captured exactly one exchange (or the same number).
5. Apply normalization rules to both request bodies.
6. Run `DiffJSON(native.RequestBody, sdk.RequestBody, rules)`.
7. Optionally diff headers (with separate header normalization).
8. If any diffs remain, fail with the structured report.

## Failure output format

Designed for consumption by both humans and coding agents:

```
=== WIRE CONFORMANCE: anthropic ===

FAIL: tool_call_with_cache_control (2 diffs)

  [REQUEST BODY]

  Path: tools[0].cache_control
    native:  {"type":"ephemeral"}
    ai-sdk:  <missing>
    kind:    MISSING - field present in native SDK request but absent in ai-sdk-go
    fix-in:  providers/anthropic/request_mapper.go (ToProvider method)

  Path: messages[1].content[0].cache_control
    native:  {"type":"ephemeral"}
    ai-sdk:  <missing>
    kind:    MISSING
    fix-in:  providers/anthropic/request_mapper.go (ToProvider method)

  [HEADERS]

  Path: anthropic-beta
    native:  "prompt-caching-2024-07-31"
    ai-sdk:  <missing>
    kind:    MISSING
    fix-in:  providers/anthropic/provider.go (NewProvider)
```

The `fix-in` hint is derived from a per-provider mapping: request body diffs point to
`request_mapper.go`, header diffs point to `provider.go`, response diffs (if added later)
point to `response_mapper.go`.

## Example scenario (Anthropic)

```go
{
    Name: "simple_text_generation",
    NativeCall: func(t *testing.T, rt *RecordingTransport) error {
        client := anthropic.NewClient(
            option.WithAPIKey("test-key"),
            option.WithHTTPClient(&http.Client{Transport: rt}),
        )
        _, _ = client.Beta.Messages.New(ctx, anthropic.BetaMessageNewParams{
            Model:     "claude-sonnet-4-5",
            MaxTokens: 1024,
            Messages: []anthropic.BetaMessageParam{
                anthropic.NewBetaUserMessage(
                    anthropic.NewBetaTextBlock("Say hello"),
                ),
            },
        })
        return nil
    },
    SDKCall: func(t *testing.T, rt *RecordingTransport) error {
        provider, _ := NewProvider("test-key",
            WithHTTPClient(&http.Client{Transport: rt}),
        )
        model, _ := provider.NewModel("claude-sonnet-4-5")
        _, _ = model.Generate(ctx, &llm.Request{
            Messages: []llm.Message{
                llm.NewMessage(llm.RoleUser, llm.NewTextPart("Say hello")),
            },
            Config: &llm.Config{MaxTokens: intPtr(1024)},
        })
        return nil
    },
}
```

## Scenarios to cover per provider

Each provider should have wire scenarios for:

- Simple text generation (single user message)
- System message handling
- Multi-turn conversation (user/assistant/user)
- Tool definitions + tool_choice variants (auto, required, specific)
- Tool call response (assistant tool_use + user tool_result)
- Structured output / response format
- Temperature, top_p, stop sequences
- Max tokens configuration
- Reasoning / thinking mode (where supported)
- Image input (where supported)
- Prompt caching (where supported)

## Maintenance concern and mitigation

Each scenario requires writing the same call twice (native + ai-sdk-go). With ~12
scenarios per provider and 4 providers, that's ~96 function pairs. This is real
maintenance cost.

Mitigation options (not for v1, but worth noting):

1. **Derive native calls from `llm.Request`**: Build a per-provider helper that takes an
   `llm.Request` and constructs the expected native SDK call programmatically. This halves
   the scenario code but requires per-provider reflection/construction logic.

2. **Golden file mode**: Instead of running the native SDK every time, capture its output
   once and store as golden files. Regenerate when the native SDK is upgraded. Trades
   always-live comparison for lower maintenance, at the cost of staleness.

3. **Auto-generate from behavioral conformance**: Each behavioral conformance test already
   defines an `llm.Request`. A wire test could be auto-derived from it if the native SDK
   call can be constructed programmatically.

For v1, explicit dual-call scenarios are fine. The verbosity is acceptable because each
scenario is simple, self-contained, and easy to review.

## Phasing

### Phase 1: Foundation + Anthropic

Build the infrastructure and prove it on one provider.

### Phase 2: OpenAI + Google

Port to remaining providers with real native SDKs.

### Phase 3: Response mapping

Extend to also compare response parsing: feed the same canned response body through the
native SDK's response types and through our `ResponseMapper`, diff the resulting
structures.

### Phase 4: Streaming

Compare SSE event streams. The recording transport returns a canned SSE body; diff the
parsed event sequence between native SDK streaming and our `GenerateEvents`.

## TODO

### Infrastructure (Phase 1)

- [ ] Create `providers/wireconformance/` package
- [ ] Implement `RecordingTransport` (captures request body/headers, returns canned response)
- [ ] Implement `DiffJSON` engine (recursive structural diff with JSON path tracking)
- [ ] Implement `NormalizeRule` application (ignore, sort, strip on path patterns)
- [ ] Implement failure report formatter (structured output with fix-in hints)
- [ ] Define `WireFixture` and `WireScenario` types
- [ ] Implement `WireSuite` (testify suite runner that executes scenarios and diffs)
- [ ] Add canned response bodies for Anthropic (minimal valid `/v1/messages` response)

### Anthropic scenarios (Phase 1)

- [ ] Scenario: simple text generation
- [ ] Scenario: system message
- [ ] Scenario: multi-turn conversation
- [ ] Scenario: tool definitions with tool_choice (auto, required, specific)
- [ ] Scenario: tool call round-trip (tool_use + tool_result)
- [ ] Scenario: structured output / response format
- [ ] Scenario: generation config (temperature, top_p, stop_sequences, max_tokens)
- [ ] Scenario: reasoning / thinking mode
- [ ] Scenario: prompt caching (cache_control markers)
- [ ] Scenario: header comparison (anthropic-beta, anthropic-version, content-type)

### OpenAI scenarios (Phase 2)

- [ ] Add canned response bodies for OpenAI
- [ ] Port all applicable scenarios to OpenAI native SDK
- [ ] Add OpenAI-specific scenarios (response_format, function calling variants)

### Google scenarios (Phase 2)

- [ ] Add canned response bodies for Google/Gemini
- [ ] Port all applicable scenarios to Google genai SDK
- [ ] Add Google-specific scenarios (safety settings, thinking config)

### Response mapping (Phase 3)

- [ ] Extend `RecordingTransport` to also capture/inject response bodies
- [ ] Build response comparison: canned response -> native SDK parse vs. ResponseMapper parse
- [ ] Add response diff scenarios per provider

### Streaming (Phase 4)

- [ ] Build SSE canned response support in `RecordingTransport`
- [ ] Compare parsed event sequences (native streaming vs. GenerateEvents)
- [ ] Add streaming scenarios per provider

### Maintenance / DX

- [ ] Add `go generate` or helper script to regenerate canned responses from native SDKs
- [ ] Document how to add new wire scenarios (CONTRIBUTING or package doc)
- [ ] Investigate auto-deriving scenarios from behavioral conformance test requests

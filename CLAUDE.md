# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses [Task](https://taskfile.dev) as its build tool. Key commands:

```bash
task ci                     # Run all CI checks (license:check, lint, unit tests)
task test                   # Run all tests (unit + integration, 30m timeout)
task test:unit              # Unit tests only (-short flag)
task test:integration       # Integration tests only (requires API keys)
task lint                   # Run golangci-lint with --fix (all issues)
task lint:new               # Run golangci-lint with --fix (new issues only)
task license                # Add Apache 2.0 license headers to Go files
task license:check          # Verify license headers
task security               # Run govulncheck + osv-scanner
```

## Testing

- Use `t.Parallel()` in all tests
- Use testify's `assert` and `require` for assertions
- Prefer table-driven tests
- Integration tests require provider API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, CONTEXT7_API_KEY) — tests are skipped if keys are missing
- Integration tests against LLM providers can be flaky due to provider behavior

## Hooks

Stop hooks in `.claude/settings.json` automatically run `task license` (adds Apache 2.0 headers) and `task lint:new` (golangci-lint with --fix, new issues only) when Claude finishes a turn. Code style (import grouping, gofumpt formatting, forbidden functions, snake_case tags) is enforced by the linter — no need to remember these rules manually.

## Adding New Models

Each provider authors `catalog.Entry` values in its `models.go` (`entries()`), frozen into a
shared `catalog.Catalog` at init — `catalog.MustNew` validates every entry, so a broken entry
fails every test run with a path-qualified error. Bedrock authors one `family` declaration per
logical model in `models.go` and `families.go` expands the geo-profile variants.

Steps for a new model:
1. Register its host-independent facts once in `catalog/facts_data.go` (canonical `ModelID`
   like `"anthropic/claude-opus-5"`, display name, **Series** — the non-branching succession
   line — release date, knowledge cutoff). Two providers offering the same model reference the
   same `ModelID`; never author facts twice.
2. Add the provider entry: exported ID constant (keep it greppable), capabilities, constraints,
   modalities, `Reasoning` (efforts/adaptive/budget), `Pricing`, and `Life` (lifecycle dates —
   see below). Adding a model must never require editing an existing entry: generation ordering,
   "superseded by", and price tiers are derived at read time.
3. Regenerate the committed snapshot: `task catalog:snapshot` (CI fails on a stale
   `catalog/snapshot.json`). The snapshot diff is the review surface — check it shows exactly
   what you meant to change.

Pricing rules:
- `pricing.NewRates` / `pricing.FlatInfo` take **US dollars per million tokens** — `$2.50/M`
  is written `2.50`. (Internally they convert to microcents; never pass microcent literals —
  `FlatInfo(250_000_000, …)` compiles and overprices by 10⁸.)
  ```go
  Pricing: pricing.FlatInfo(2.50, 10.00, 1.25), // $2.50/M in, $10.00/M out, $1.25/M cached
  ```
- Sources: OpenAI https://developers.openai.com/api/docs/pricing · Anthropic
  https://docs.anthropic.com/en/docs/about-claude/pricing · Google
  https://ai.google.dev/gemini-api/docs/pricing · Bedrock https://aws.amazon.com/bedrock/pricing/
- Context-tiered pricing (Gemini Pro, Anthropic >200K surcharge) uses `pricing.TieredInfo(...)`;
  service-tier/speed/region variation uses `Pricing.WithOverride(...)` — never provider-specific
  pricing fields. An input/output rate of `0` means *unpriced* and is rejected; a published $0
  rate is `pricing.RateFree`.

Lifecycle rules:
- `Life.Retires` is an **exact announced shutdown date only** (inclusive: retired *on* that
  date). A published "not sooner than" floor goes in `RetirementNotBefore` and never derives
  retirement. `Life.ReplacedBy` must name an offering in the same catalog; skip it when the
  provider's recommendation isn't one we carry.
- **The catalog is append-only.** When a provider retires a model, do NOT delete its entry:
  set `Life.Retires`, add a `// Deprecated:` doc comment on its ID constant naming the
  replacement, and leave the entry so historical usage stays priceable. Lifecycle sources:
  Anthropic platform.claude.com/docs/en/about-claude/model-deprecations · OpenAI
  developers.openai.com/api/docs/deprecations · Google ai.google.dev/gemini-api/docs/deprecations
  · Bedrock `ListFoundationModels` (`modelLifecycle`).

## Project Structure

- `llm/` — Core types and interfaces (Request, Response, Message, Part, Event)
- `catalog/` — Shared model-metadata read model (facts registry, offerings, lifecycle views,
  snapshot encoder); `catalog/snapshot.json` is the committed, CI-checked artifact
- `providers/` — LLM provider implementations (anthropic, openai, google, bedrock, openaicompat)
- `agent/` — Agent framework; `llmagent/` has the LLM-powered agent with tool calling
- `tool/` — Tool registry, MCP integration, built-in tools, agent-as-tool
- `adapter/a2a/` — Agent-to-Agent protocol adapter
- `runner/` — Agent execution runner with session management
- `plugins/` — Interceptor plugins (retry, OpenTelemetry)
- `examples/` — Example applications (also in go.work workspace)

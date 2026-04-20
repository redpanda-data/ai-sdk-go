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

When adding a model to a provider's `supportedModels` map, you **must** set the `Pricing` field. The per-provider `TestAllModelsHavePricing` test enforces this.

Prices are in **microcents per million tokens** (see `pricing/pricing.go` for why). To convert from a provider's pricing page:

```
dollars × 100_000_000 = microcents
```

Steps:
1. Find the model's pricing on the provider's page:
   - OpenAI: https://openai.com/api/pricing/
   - Anthropic: https://docs.anthropic.com/en/docs/about-claude/pricing
   - Google: https://ai.google.dev/gemini-api/docs/pricing
   - Bedrock: https://aws.amazon.com/bedrock/pricing/
2. Convert each dollar price: `$2.50/M` → `250_000_000`
3. Add a dollar comment on every value for readability:
   ```go
   Pricing: pricing.Info{
       InputPerMillion:       250_000_000, // $2.50/M
       OutputPerMillion:      1_000_000_000, // $10.00/M
       CachedInputPerMillion: 125_000_000, // $1.25/M
   },
   ```
4. If the model has tiered pricing (like Gemini Pro), use the `Tiers` field instead.

## Project Structure

- `llm/` — Core types and interfaces (Request, Response, Message, Part, Event)
- `providers/` — LLM provider implementations (anthropic, openai, google, bedrock, openaicompat)
- `agent/` — Agent framework; `llmagent/` has the LLM-powered agent with tool calling
- `tool/` — Tool registry, MCP integration, built-in tools, agent-as-tool
- `adapter/a2a/` — Agent-to-Agent protocol adapter
- `runner/` — Agent execution runner with session management
- `plugins/` — Interceptor plugins (retry, OpenTelemetry)
- `examples/` — Example applications (also in go.work workspace)

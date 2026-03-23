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

## Project Structure

- `llm/` — Core types and interfaces (Request, Response, Message, Part, Event)
- `providers/` — LLM provider implementations (anthropic, openai, google, bedrock, openaicompat)
- `agent/` — Agent framework; `llmagent/` has the LLM-powered agent with tool calling
- `tool/` — Tool registry, MCP integration, built-in tools, agent-as-tool
- `adapter/a2a/` — Agent-to-Agent protocol adapter
- `runner/` — Agent execution runner with session management
- `plugins/` — Interceptor plugins (retry, OpenTelemetry)
- `examples/` — Example applications (also in go.work workspace)

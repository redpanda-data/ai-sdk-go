# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses [Task](https://taskfile.dev) as its build tool. Key commands:

```bash
task ci                     # Run all CI checks (license:check, lint, unit tests)
task test                   # Run all tests (unit + integration, 30m timeout)
task test:unit              # Unit tests only (-short flag)
task test:integration       # Integration tests only (requires API keys)
task lint                   # Run golangci-lint with --fix
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
- New files must have Apache 2.0 license headers (`task license` to add them)

## Code Style

- **Imports**: Group as stdlib, third-party, then local (`github.com/redpanda-data/ai-sdk-go`)
- **JSON/YAML struct tags**: Use snake_case (enforced by tagliatelle linter)
- **Forbidden**: `fmt.Print*`, `log.*`, `print()`, `println()`, `panic()` (enforced by forbidigo linter)
- **Errors**: Use wrapped static errors, avoid dynamic error creation
- Formatting is handled by gofumpt/goimports via `task lint`

## Project Structure

- `llm/` — Core types and interfaces (Request, Response, Message, Part, Event)
- `providers/` — LLM provider implementations (anthropic, openai, google, bedrock, openaicompat)
- `agent/` — Agent framework; `llmagent/` has the LLM-powered agent with tool calling
- `tool/` — Tool registry, MCP integration, built-in tools, agent-as-tool
- `adapter/a2a/` — Agent-to-Agent protocol adapter
- `runner/` — Agent execution runner with session management
- `plugins/` — Interceptor plugins (retry, OpenTelemetry)
- `examples/` — Example applications (also in go.work workspace)

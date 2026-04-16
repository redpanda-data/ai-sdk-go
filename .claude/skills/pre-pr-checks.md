---
name: pre-pr-checks
description: "Pre-PR checks for ai-sdk-go. Runs license headers, lint, and unit tests matching the CI pipeline (task ci)."
---

# Pre-PR Checks — ai-sdk-go

Run these before pushing changes or creating a PR.

## Quick Check (all CI steps)

```bash
task ci
```

This runs: `license:check` → `lint` → `test:unit` (matches CI exactly).

## Individual Steps

### 1. License Headers

```bash
# Check
task license:check

# Auto-fix (add missing headers)
task license
```

### 2. Lint

```bash
# Full lint with auto-fix
task lint

# Only new issues (faster, what the Stop hook runs)
task lint:new
```

Common lint rules that catch people:
- **wsl_v5** — requires blank lines before `if`, `return`, and between declarations and logic
- **funcorder** — constructors must appear before methods on the same struct
- **gocritic** — catches duplicate branch bodies, unnecessary conversions, etc.

### 3. Tests

```bash
# Unit tests only (fast, no API keys needed) — this is what CI runs
task test:unit

# Single package
go test ./providers/openai/ -short -count=1 -v

# Single test
go test ./providers/openai/ -run TestResolveModelFamily -v -count=1

# All tests including integration (requires API keys)
task test
```

### 4. Security (optional, not in CI gate)

```bash
task security
```

## CI → Local Mapping

| CI Step | Local Command |
|---------|--------------|
| License check | `task license:check` |
| Lint (all issues, with fix) | `task lint` |
| Unit tests | `task test:unit` |
| All of the above | `task ci` |

## Common Failures

1. **Missing license header** — Fix: `task license`
2. **wsl_v5 whitespace** — Add blank lines before `if`/`return` after multi-line blocks
3. **funcorder** — Move constructor `NewFoo()` above `Foo.Method()`
4. **Test flakes on integration** — Integration tests hit live APIs; re-run or use `task test:unit`
5. **Import grouping** — gofumpt enforced: stdlib, then external, then internal

# Web Research Agent

An interactive research assistant that decomposes a question into sub-topics,
runs the searches **in parallel**, reads the promising results, and synthesizes a
cited answer.

It exists to demonstrate three things the other examples don't:

- **Parallel tool calls** — the system prompt pushes the model to emit several
  `web_search` / `fetch_url` calls in a *single* turn, so the registry executes
  them concurrently.
- **Multi-turn sessions** — the REPL reuses one session, so follow-up questions
  keep their context.
- **Tracing** — the `plugins/otel` interceptor instruments the whole run:
  `invoke_agent` → `chat` → parallel `execute_tool` spans.

## Running

```bash
export OPENAI_API_KEY=sk-...
export TAVILY_API_KEY=tvly-...   # free tier at https://app.tavily.com

go run .
```

Type a question; `exit` or `quit` to leave.

## Tracing (optional)

Tracing is off unless you configure it, and it's configured entirely through the
[standard OpenTelemetry environment variables][otel-env] — the example itself
hardcodes no endpoint or auth scheme. Any OTLP/HTTP collector works:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_SERVICE_NAME=web-research-agent          # optional
go run .
```

Add auth headers if your collector needs them:

```bash
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer $TOKEN"
```

Every exported batch is logged (`otlp: exported N span(s) to …`), and export
failures are logged in red, which makes this a convenient collector smoke test.

[otel-env]: https://opentelemetry.io/docs/specs/otel/configuration/sdk-environment-variables/

## Note on `fetch_url`

`fetch_url` issues a plain HTTP GET against whatever URL the model picks, with no
SSRF allowlist — it's a local dev example. Don't run it against untrusted input
on a networked host.

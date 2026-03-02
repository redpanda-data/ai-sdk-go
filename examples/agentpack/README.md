# Quick Start with Ollama

This example runs a local assistant with two MCP servers:
- **Filesystem** — read, write, and search local files
- **Atlassian** — Jira, Confluence, and Compass via API token

## Prerequisites

- [Ollama](https://ollama.ai) installed
- Node.js / npx (for MCP servers)
- An Atlassian API token (your org admin must enable API token auth for the MCP server)
  - Create one at: https://id.atlassian.com/manage-profile/security/api-tokens

## Run

1. Pull a model:
   ```bash
   ollama pull llama3.3
   ```
2. Run the agent (CLI mode):
   ```bash
   ATLASSIAN_API_TOKEN=your-token-here \
     AI_PROVIDER=ollama AI_MODEL=llama3.3 AGENT_MODE=cli \
     go run ../../cmd/agent-runner
   ```
   Try asking: "List the files in the current directory" or "Search Jira for recent bugs".

3. Run as HTTP server (cloud-ready, no browser needed):
   ```bash
   ATLASSIAN_API_TOKEN=your-token-here \
     AI_PROVIDER=ollama AI_MODEL=llama3.3 \
     go run ../../cmd/agent-runner
   # In another terminal:
   curl http://localhost:8080/healthz
   ```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AI_PROVIDER` | yes | - | `anthropic`, `openai`, `google`, `openai-compat`, `ollama` |
| `AI_MODEL` | yes | - | Model name (e.g., `claude-opus-4-6`, `llama3.3`) |
| `AI_API_KEY` | yes* | - | API key (*not required for `ollama`) |
| `AI_BASE_URL` | no | provider default | Custom endpoint |
| `SESSION_STORE` | no | `memory` | `memory` or `redpanda` |
| `AGENT_MODE` | no | `http` | `http` or `cli` |
| `AGENT_PORT` | no | `8080` | HTTP server port |
| `LOG_LEVEL` | no | `info` | `debug`, `info`, `warn`, `error` |
| `OTEL_ENABLED` | no | `false` | Enable OpenTelemetry tracing |
| `ATLASSIAN_API_TOKEN` | no | - | Atlassian API token for MCP server |

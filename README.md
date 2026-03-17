# ai-sdk-go

[![Go Reference](https://pkg.go.dev/badge/github.com/redpanda-data/ai-sdk-go.svg)](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Go SDK for building AI-powered applications with a unified interface across multiple LLM providers. Supports OpenAI, Anthropic, Google Gemini, AWS Bedrock, and any OpenAI-compatible API.

## Install

```bash
go get github.com/redpanda-data/ai-sdk-go
```

## Quick Start

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func main() {
	provider, err := openai.NewProvider(os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		log.Fatal(err)
	}
	model, err := provider.NewModel(openai.ModelGPT5_2)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Explain Go interfaces in two sentences.")),
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp.TextContent())
}
```

## Providers

### Anthropic

```go
import "github.com/redpanda-data/ai-sdk-go/providers/anthropic"

provider, err := anthropic.NewProvider(os.Getenv("ANTHROPIC_API_KEY"))
model, err := provider.NewModel(anthropic.ModelClaudeOpus46)
```

### Google Gemini

```go
import "github.com/redpanda-data/ai-sdk-go/providers/google"

provider, err := google.NewProvider(ctx, os.Getenv("GOOGLE_API_KEY"))
model, err := provider.NewModel(google.ModelGemini25Pro)
```

### AWS Bedrock

```go
import "github.com/redpanda-data/ai-sdk-go/providers/bedrock"

provider, err := bedrock.NewProvider(ctx) // uses AWS credential chain
model, err := provider.NewModel(bedrock.ModelClaudeOpus46)
```

### OpenAI-Compatible

Works with DeepSeek, local models, or any OpenAI-compatible API.

```go
import "github.com/redpanda-data/ai-sdk-go/providers/openaicompat"

provider, err := openaicompat.NewProvider(apiKey, openaicompat.WithBaseURL("https://api.deepseek.com"))
model, err := provider.NewModel("deepseek-chat")
```

## Streaming

Use `GenerateEvents` with Go's range-over-func for streaming responses:

```go
for event, err := range model.GenerateEvents(ctx, req) {
	if err != nil {
		log.Fatal(err)
	}
	switch e := event.(type) {
	case llm.ContentPartEvent:
		fmt.Print(e.Part.Text())
	case llm.StreamEndEvent:
		if e.Error != nil {
			log.Fatal(e.Error)
		}
	}
}
```

## Agents & Tools

Build agentic workflows with tool registries and the LLM agent runner. Agents execute in a loop, calling tools and reasoning until a task is complete.

```go
import (
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

registry := tool.NewRegistry(tool.RegistryConfig{})
registry.Register(myTool)

agent, err := llmagent.New("my-agent", "You are a helpful assistant.", model,
	llmagent.WithTools(registry),
)
```

See [`examples/`](examples/) for full working demos.

## Key Packages

- [`llm`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/llm) — Core types: `Model`, `Request`, `Response`, `Event`
- [`agent`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/agent) — Agent framework and interceptor interfaces
- [`agent/llmagent`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/agent/llmagent) — LLM-powered agent implementation
- [`runner`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/runner) — Agent execution runner with session management
- [`tool`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/tool) — Tool registry and execution
- [`tool/mcp`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/tool/mcp) — Model Context Protocol integration
- [`adapter/a2a`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/adapter/a2a) — Agent-to-Agent protocol adapter
- [`llm/fakellm`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/llm/fakellm) — Test doubles for LLM models

## Examples

The [`examples/`](examples/) directory contains runnable demos:

- **[agent_as_tool](examples/agent_as_tool)** — Delegate subtasks to a nested agent for context isolation
- **[agent_interceptors](examples/agent_interceptors)** — Observability and approval hooks for agent execution
- **[grafana-cloud-agent](examples/grafana-cloud-agent)** — Full-featured agent with Grafana Cloud integration

## License

Apache 2.0 — see [LICENSE](LICENSE).

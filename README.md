# Redpanda AI SDK for Go

[![Go Reference](https://pkg.go.dev/badge/github.com/redpanda-data/ai-sdk-go.svg)](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Go SDK for building AI-powered applications with a unified interface across multiple LLM providers. Supports OpenAI, Anthropic, Google Gemini, Google Vertex AI, Meta, AWS Bedrock, and any OpenAI-compatible API.

<p align="center">
  <img src="assets/gopher.png" alt="Redpanda AI SDK Gopher" width="400">
</p>

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
	model, err := provider.NewModel(openai.ModelGPT5_4)
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
model, err := provider.NewModel(google.ModelGemini31ProPreview)
```

### Meta

```go
import "github.com/redpanda-data/ai-sdk-go/providers/meta"

provider, err := meta.NewProvider(os.Getenv("MODEL_API_KEY"))
if err != nil {
    return err
}
model, err := provider.NewModel(meta.ModelMuseSpark13)
if err != nil {
    return err
}
```

See [Meta provider documentation](providers/meta/README.md) for shared options and known limits.

### AWS Bedrock

```go
import "github.com/redpanda-data/ai-sdk-go/providers/bedrock"

provider, err := bedrock.NewProvider(ctx) // uses AWS credential chain
model, err := provider.NewModel(bedrock.ModelClaudeOpus46)
```

### Google Vertex AI

Serves Gemini and Claude models from one Google Cloud project and location. The HTTP client must add the credential itself.

```go
import "github.com/redpanda-data/ai-sdk-go/providers/vertex"

provider, err := vertex.NewProvider(ctx,
    vertex.WithProject("my-project"),
    vertex.WithLocation("us-east5"),
    vertex.WithHTTPClient(authClient), // adds a bearer token to every request
)
model, err := provider.NewModel(vertex.ModelClaudeHaiku45)
```

### OpenAI-Compatible

Works with DeepSeek, local models, or any OpenAI-compatible API.

```go
import "github.com/redpanda-data/ai-sdk-go/providers/openaicompat"

provider, err := openaicompat.NewProvider(apiKey, openaicompat.WithBaseURL("https://api.deepseek.com"))
model, err := provider.NewModel("deepseek-v4-flash", openaicompat.WithThinking(false))
```

Use `deepseek-v4-pro` with `WithReasoning()` and `WithThinking(true)` for maximum capability.

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

### Contextual registries

`tool.Registry` resolves tools with `List(ctx)` and `Get(ctx, name)`; both return errors.
`tool.SnapshotRegistry` is an optional capability that adds `Snapshot(ctx)` to capture definitions,
membership and execution policies for one model step. Native registries and the contextual MCP
registry implement it. `WithTools` captures sources before schema planning, lazy discovery or
context budgeting. A context-overflow retry reuses that step's capture; the next model step
resolves afresh.

For caller-dependent HTTP MCP tools, use the on-demand registry:

```go
remote, err := mcp.NewRegistry("github", "https://example.com/mcp",
	mcp.WithCredentials(func(ctx context.Context) (mcp.Credentials, error) {
		caller, err := authenticatedCaller(ctx) // Application-owned identity and credential lookup.
		if err != nil {
			return mcp.Credentials{}, err
		}
		return mcp.NewCredentials(
			mcp.WithPartitionKey(caller.AuthScope),
			mcp.WithTokenSource(caller.TokenSource),
		)
	}),
)
if err != nil {
	return err
}
defer remote.Close()

assistant, err := llmagent.New("my-agent", "Help the caller.", model,
	llmagent.WithTools(remote),
)
```

The application defines `PartitionKey`: equal keys share client-side connection/protocol
state and the bounded tool-catalog cache. Keys are not MCP wire session IDs. Credentials
are resolved again for execution and HTTP requests, but captured tools cannot execute under
a different partition. The SDK does not append a header hash or automatically invalidate
catalogs when headers change under the same key. Include an authorization version or header
hash in the key when that should select fresh state. Server change notifications invalidate
that partition's catalog; server TTLs are capped by `WithRegistryMaxToolCacheTTL` (default
30 seconds, zero disables caching). A server that provides no positive TTL is rediscovered
on each snapshot, including each model step; notifications alone do not make its catalog cacheable.
There is no periodic discovery timer.
See [`tool/mcp`](tool/mcp/README.md) for authentication options and legacy client migration.

### Combining registries

Pass multiple registries to `WithTools`; repeated options append sources. Nil sources are ignored:

```go
assistant, err := llmagent.New("my-agent", "You are a helpful assistant.", model,
	llmagent.WithTools(staticTools, remote),
)
```

Each source is captured once before composition. Removed tools cannot be restored by session
load metadata. Duplicate names, conflicting group metadata, or an error resolving any source stop
the entire step before the model call; valid sources are not used as a partial fallback.
`tool.Snapshot(ctx, registries...)` captures and combines sources directly. For a `SnapshotRegistry`,
it uses the returned snapshot for dispatch. For a plain `Registry`, it copies the advertised
definitions but dispatches through the original registry, preserving its middleware and live
execution lookup. That legacy fallback does not promise frozen executable selection or policies.
Snapshots do not clone executable implementations or close their connections. Callers own those
lifetimes and must make tools and credential callbacks concurrency-safe.

### Registry wrappers

Embedding `tool.Registry` does not inherit snapshot capability. An existing wrapper that overrides
`Execute` for authorization or logging therefore keeps its execution path; capturing tools does not
silently strip the wrapper. A wrapper can explicitly implement `SnapshotRegistry` by capturing its
inner registry and returning a new wrapper around the capture:

```go
func (r *authorizedRegistry) Snapshot(ctx context.Context) (tool.Registry, error) {
    captured, err := tool.Snapshot(ctx, r.Registry)
    if err != nil {
        return nil, err
    }
    return &authorizedRegistry{Registry: captured, authorize: r.authorize}, nil
}
```

Keep the wrapper's `Execute` checks active on each call; do not cache an authorization decision
at discovery time. Every layer in a snapshot-capable decorator chain must preserve its own behavior
when it returns the wrapped capture. Wrappers around caller-scoped contextual registries should
explicitly forward this capability to preserve captured routes and partition checks; the plain-registry
fallback is not a substitute for contextual snapshot guarantees.

### Lazy tool loading

For larger registries, defer tools whose schemas are only needed occasionally:

```go
support := tool.NewGroup(llm.ToolGroup{
	Name:         "support",
	Description:  "Customer tickets and account lookups",
	Instructions: "Look up the customer before creating a ticket.",
})
support.Add(searchTickets).AddDeferred(createTicket, closeTicket) // Add: always loaded; AddDeferred: on demand

if err := support.Register(registry); err != nil {
	return err
}

agent, err := llmagent.New("support", "Help with support requests.", model,
	llmagent.WithTools(registry),
)
```

Nothing else to switch on. The agent chooses discovery based on the model's capabilities:

- Supported Anthropic models use hosted tool search and native deferred schemas.
- OpenAI Responses models with tool search support use native deferred functions; tool groups
  become namespaces. For an unchanged registry, both native paths keep the tool catalog stable
  as tools are discovered.
- Gemini, OpenAI-compatible endpoints, Bedrock Converse, and models without native search
  support use the local `tool_search` tool. Discovered schemas join the next request's tools
  array; prefix caching can be invalidated when the set changes.

Loaded names persist across turns, compaction and session restarts; a tool is available only
while present in the current step's registry. If compaction removes native search references,
previously loaded tools still in the registry become eager in the next request. Native mode puts
a group directory and all group instructions in the system prompt up front to keep it stable
while the registry is unchanged; the local fallback includes a name/summary manifest and activates
group instructions as tools load.

`mcp.WithDeferredTools()`, `mcp.WithAlwaysLoad("search_tickets")`, and `mcp.WithToolGroup(...)`
express the same policy per server. `tool.WithGroup` and `tool.WithDeferred` are the per-tool
registration options underneath. `llmagent.WithToolLoadingConfig` tunes local search limits;
hosted search controls its own selection and does not use `MaxLoadTokens`. To use local search
on a model that supports native search, pass
`llmagent.WithToolLoadingConfig(llmagent.ToolLoadingConfig{ForceLocal: true})`.
[`examples/lazy_tools`](examples/lazy_tools/) runs against public MCP servers. Its keyless dry-run
prints the local fallback request.

The session remembers loaded names, not schemas or authorization. Schemas come from the current
step's registry. If later loads exhaust its tool capacity, a fresh session may have room for
those tools. Searches return flat lists of loaded names; explicit `select:` queries use the
requested order when applying the load budget.

Tool interceptors may edit arguments, deny execution, retry, or transform results. The request
and response name and ID must remain those of the original call; identity changes return a tool error.

## Key Packages

- [`llm`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/llm) — Core types: `Model`, `Request`, `Response`, `Event`
- [`agent`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/agent) — Agent framework and interceptor interfaces
- [`agent/llmagent`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/agent/llmagent) — LLM-powered agent implementation
- [`runner`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/runner) — Agent execution runner with session management
- [`tool`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/tool) — Tool registry and execution
- [`tool/mcp`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/tool/mcp) — Model Context Protocol integration
- [`adapter/a2a`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/adapter/a2a) — Agent-to-Agent protocol adapter
- [`adapter/a2ui`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/adapter/a2ui) — Opt-in, read-only A2UI 1.0 snapshots for self-managed A2A 1.0 endpoints
- [`llm/fakellm`](https://pkg.go.dev/github.com/redpanda-data/ai-sdk-go/llm/fakellm) — Test doubles for LLM models

## Examples

The [`examples/`](examples/) directory contains runnable demos:

- **[agent_as_tool](examples/agent_as_tool)** — Delegate subtasks to a nested agent for context isolation
- **[agent_interceptors](examples/agent_interceptors)** — Observability and approval hooks for agent execution

## Read-only A2UI snapshots

A self-managed A2A 1.0 executor can opt in by adding `a2ui.AgentExtension()`
to its AgentCard and calling `a2ui.SnapshotPart(request.Message, snapshot)`.
Use `a2ui.SupportsSnapshot(request.Message)` before generating optional UI.
Keep a text part in the reply. If validation fails, omit the snapshot and
return the text reply.
Without a compatible version and catalog offer on **that message**, the helper
returns no part. Extension activation alone does not opt a turn in.

Build the `a2ui.Snapshot` from application data with a new surface ID and
Basic Catalog `Text`, `Row`, and `Column` nodes. The helper validates the tree
and emits the [A2UI-over-A2A v1.0 data part](https://a2ui.org/specification/v1.0-a2ui-extension-specification/).
Basic Catalog `Text` may render simple Markdown. The SDK validates the
snapshot structure, not the text markup. Sanitize untrusted strings for your
renderer before using them as `Text`.
This does not modify the existing A2A 0.3 adapter, prompt the model, or enable UI actions.

## License

Apache 2.0 — see [LICENSE](LICENSE).

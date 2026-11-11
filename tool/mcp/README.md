# MCP Client Package

Tools for connecting to [Model Context Protocol (MCP)](https://modelcontextprotocol.io) servers:

- `NewRegistry`: contextual, on-demand HTTP discovery with application-defined authorization partitions.
- `NewClient`: the existing stdio/HTTP/SSE client, optionally synchronizing a separate native registry.

Both can feed `llmagent.WithTools`. The consumer can select either implementation behind a feature flag;
there is no automatic fallback from a failed contextual credential lookup to shared credentials.

## Features

- **Multiple Transport Types**: stdio (subprocess), HTTP streaming, SSE
- **Automatic Tool Management**: Discover and register tools from MCP servers
- **OAuth 2.1 Support**: Built-in support for OAuth authentication with PKCE and Resource Indicators
- **Contextual HTTP registry**: Discover and execute under caller-resolved credentials without a polling timer
- **Optional legacy auto-sync**: Keep an existing client-managed registry refreshed
- **Connection Management**: Streamable HTTP transport includes automatic reconnection with exponential backoff
- **Tool Filtering**: Selectively register tools based on custom criteria

## Contextual HTTP registry

The registry constructor performs no network I/O. Its credential provider receives the application's
trusted request context and must explicitly return an authorization partition, even for public servers.
The agent captures a registry snapshot once before each model step.

```go
import (
    "context"

    "github.com/redpanda-data/ai-sdk-go/agent/llmagent"
    "github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

registry, err := mcp.NewRegistry("github", "https://example.com/mcp",
    mcp.WithCredentials(func(ctx context.Context) (mcp.Credentials, error) {
        key, tokenSource, err := credentialsForCaller(ctx) // Application-owned lookup or token exchange.
        if err != nil {
            return mcp.Credentials{}, err
        }
        return mcp.NewCredentials(
            mcp.WithPartitionKey(key),
            mcp.WithTokenSource(tokenSource), // Any concurrency-safe oauth2.TokenSource.
        )
    }),
)
if err != nil {
    return err
}
defer registry.Close()

assistant, err := llmagent.New("assistant", "Help the caller.", model,
    llmagent.WithTools(registry),
)
```

### Credential options

`NewCredentials` takes functional options. `WithPartitionKey` is required; the other options support
static bearer tokens, API keys, HTTP Basic authentication, OAuth token sources and extra headers:

```go
mcp.NewCredentials(mcp.WithPartitionKey(key), mcp.WithBearerToken(token))
mcp.NewCredentials(mcp.WithPartitionKey(key), mcp.WithAPIKey("X-API-Key", apiKey))
mcp.NewCredentials(mcp.WithPartitionKey(key), mcp.WithBasicAuth(username, password))
mcp.NewCredentials(mcp.WithPartitionKey(key), mcp.WithTokenSource(source), mcp.WithHeaders(extraHeaders))
mcp.NewCredentials(mcp.WithPartitionKey("public")) // Explicit anonymous access.
```

The last Authorization-setting option wins. A token source supplies Authorization on every HTTP request;
standard `golang.org/x/oauth2` sources can cache and refresh tokens. Callers own the source and its lifecycle
context. Credential errors stop the operation. Header maps are copied before use and are never installed
as mutable global transport state.

### Partitioning and freshness

`PartitionKey` is the application's authorization boundary, not necessarily a username or a wire MCP
session ID. It may include tenant, delegation or permission scope. Equal keys reuse client-side protocol
state and the bounded catalog cache. The SDK does **not** add a header hash or compare changed
credentials to invalidate a catalog. If a credential or permission change must select fresh state,
include a hash or version in the key. Legacy servers are supported by the same registry through
protocol negotiation.

Execution re-resolves current credentials and requires the partition captured during discovery. An
Alice snapshot cannot execute as Bob. A stable partition does not freeze credentials: newly supplied
headers or refreshed OAuth tokens are used on outgoing requests. Client-side state survives independently
of whether the negotiated protocol uses a wire MCP session.

Server `tools/list_changed` notifications invalidate the corresponding partition, including all pages.
The registry subscribes to modern pushes and receives legacy notifications. Server TTLs cannot exceed
`WithRegistryMaxToolCacheTTL` (default 30 seconds); zero disables catalog caching, and a nonpositive
server TTL never becomes cacheable. Expired entries refresh on demand, not on a background timer.
A notification received during discovery prevents an older response from restoring a stale cache.
The current model step keeps its captured definitions; the next step resolves the changed catalog.

The pool is bounded, evicts only idle/unleased entries and expires idle entries on access. Configure it
with `WithRegistryMaxConnections` and `WithRegistryIdleTimeout`. `Close`/`Shutdown` cancel and drain
operations. `WithRegistryHTTPClient` supports custom transports, TLS and timeouts; shared cookie jars
are rejected. Signing and client-certificate identities are transport configuration, not header options.
`WithRegistryToolOptions` applies native tool registration options such as timeout, result limit, group
or deferral to every discovered tool.

### Migration and result payloads

`tool.Registry.List(ctx)` now returns `(definitions, error)`, `Get(ctx, name)` receives context, and every
implementation provides `Snapshot(ctx) (tool.Registry, error)`. Consumers must make this compile-time
migration even when a runtime flag selects the legacy client. Use `WithTools(staticTools, registry)`
to compose multiple contextual sources; no separate toolset callback is required.

The new registry returns the complete MCP `CallToolResult` JSON object, including `content`,
`structuredContent` and metadata; MCP `isError` is also reflected in `ToolResponsePart.IsError`.
The existing `Client.ExecuteTool` still returns only the content array. Consumers rolling out the new
registry behind a flag must account for this payload difference. Transport/JSON-RPC failures remain
errors; no fake batch tools are used for scheduling.

## Legacy client quick start

### Basic usage with stdio transport

```go
import (
    "context"
    "github.com/redpanda-data/ai-sdk-go/tool"
    "github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

// Create tool registry
registry := tool.NewRegistry(tool.RegistryConfig{})

// Create stdio transport for local MCP server
transport := mcp.NewStdioTransport(
    "npx",
    []string{"-y", "@modelcontextprotocol/server-everything"},
    nil,
)

// Create and start client
client, err := mcp.NewClient(
    "mcp-server",
    transport,
    mcp.WithRegistry(registry),
    mcp.WithAutoSync(5*time.Minute),
)
if err != nil {
    log.Fatal(err)
}

ctx := context.Background()
if err := client.Start(ctx); err != nil {
    log.Fatal(err)
}
defer client.Close()

// Tools are now automatically registered in the registry
```

### Using HTTP/SSE Transports

```go
// HTTP streaming transport (2025-03-26 spec)
transport := mcp.NewStreamableTransport("https://api.example.com/mcp")

// SSE transport (2024-11-05 spec)
transport := mcp.NewSSETransport("https://api.example.com/sse")
```

## OAuth Authentication

The MCP specification requires OAuth 2.1 with Resource Indicators (RFC 8707). PKCE is required for Authorization Code flows but does not apply to Client Credentials flows.

### Authorization Code Grant (User Authentication)

For applications that need user authorization with PKCE:

```go
import (
    "context"
    "net/url"
    "golang.org/x/oauth2"
    "github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

// Step 1: Configure OAuth
oauthConfig := &oauth2.Config{
    ClientID:     "your-client-id",
    ClientSecret: "your-client-secret", // Optional for public clients
    RedirectURL:  "http://localhost:8080/callback",
    Scopes:       []string{"mcp.tools"},
    Endpoint: oauth2.Endpoint{
        AuthURL:  "https://auth.example.com/authorize",
        TokenURL: "https://auth.example.com/token",
    },
}

// Step 2: Generate PKCE verifier (required by MCP spec)
verifier := oauth2.GenerateVerifier()

// Step 3: Build authorization URL with PKCE challenge
authURL := oauthConfig.AuthCodeURL("state",
    oauth2.S256ChallengeOption(verifier))

// Redirect user to authURL...

// Step 4: After user authorizes, exchange code for token
token, err := oauthConfig.Exchange(ctx, code,
    oauth2.VerifierOption(verifier),
    mcp.WithResourceIndicator("https://api.example.com/mcp"))

// Step 5: Create transport with OAuth
factory := mcp.NewStreamableTransport("https://api.example.com/mcp",
    mcp.WithOAuth(ctx, oauthConfig))
```

### Client Credentials Grant (Machine-to-Machine)

For server-to-server authentication (PKCE not applicable):

```go
import (
    "context"
    "net/url"
    "golang.org/x/oauth2/clientcredentials"
    "github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

// Step 1: Configure client credentials with resource indicator
credConfig := &clientcredentials.Config{
    ClientID:     "your-client-id",
    ClientSecret: "your-client-secret",
    TokenURL:     "https://auth.example.com/token",
    Scopes:       []string{"mcp.tools"},
    EndpointParams: url.Values{
        "resource": []string{"https://api.example.com/mcp"}, // Required by MCP spec
    },
}

// Step 2: Create HTTP client (automatically handles token fetching and refresh)
httpClient := credConfig.Client(ctx)

// Step 3: Create transport with the configured client
factory := mcp.NewStreamableTransport("https://api.example.com/mcp",
    mcp.WithHTTPClient(httpClient))
```

### Combining OAuth with Custom HTTP Client

You can customize the HTTP client while using OAuth:

```go
// Create custom HTTP client
customClient := &http.Client{
    Timeout: 30 * time.Second,
    Transport: &http.Transport{
        TLSClientConfig: &tls.Config{
            MinVersion: tls.VersionTLS12,
        },
    },
}

// Use with OAuth - custom client settings are preserved
factory := mcp.NewStreamableTransport("https://api.example.com/mcp",
    mcp.WithHTTPClient(customClient),
    mcp.WithOAuth(ctx, oauthConfig))
```

## Client Configuration

### Tool Registry Integration

Automatically register tools with a tool registry:

```go
registry := tool.NewRegistry(tool.RegistryConfig{})

client, err := mcp.NewClient(
    "github-mcp",
    transport,
    mcp.WithRegistry(registry),
)
```

### Auto-sync

Automatically refresh tools when the server updates:

```go
client, err := mcp.NewClient(
    "github-mcp",
    transport,
    mcp.WithRegistry(registry),
    mcp.WithAutoSync(5*time.Minute), // Sync every 5 minutes
)
```

### Tool Filtering

Selectively register tools based on custom criteria:

```go
filter := func(name, description string) bool {
    return strings.HasPrefix(name, "search_")
}

client, err := mcp.NewClient(
    "github-mcp",
    transport,
    mcp.WithRegistry(registry),
    mcp.WithToolFilter(filter),
)
```

### Custom Logger

Provide a custom logger for debugging:

```go
import "log/slog"

logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
    Level: slog.LevelDebug,
}))

client, err := mcp.NewClient(
    "github-mcp",
    transport,
    mcp.WithLogger(logger),
)
```

## Manual Tool Access

Access tools without a registry:

```go
// List available tools
tools, err := client.ListTools(ctx)
for _, tool := range tools {
    fmt.Printf("Tool: %s\n", tool.NamespacedName)
}

// Execute a tool
args := json.RawMessage(`{"query": "example"}`)
result, err := client.ExecuteTool(ctx, "search", args)
```

## Transport Options

### stdio Transport

For local MCP servers running as subprocesses:

```go
transport := mcp.NewStdioTransport(
    "npx",
    []string{"-y", "@modelcontextprotocol/server-everything"},
    []string{"ENV_VAR=value"}, // Optional environment variables
)
```

### HTTP Streaming Transport

For bidirectional HTTP streaming (2025-03-26 spec):

```go
// Basic
transport := mcp.NewStreamableTransport("https://api.example.com/mcp")

// With custom HTTP client
transport := mcp.NewStreamableTransport(
    "https://api.example.com/mcp",
    mcp.WithHTTPClient(customClient),
)

// With OAuth
transport := mcp.NewStreamableTransport(
    "https://api.example.com/mcp",
    mcp.WithOAuth(ctx, oauthConfig),
)
```

### SSE Transport

For Server-Sent Events streaming (2024-11-05 spec):

```go
transport := mcp.NewSSETransport("https://api.example.com/sse")

// Options are the same as NewStreamableTransport
transport := mcp.NewSSETransport(
    "https://api.example.com/sse",
    mcp.WithHTTPClient(customClient),
    mcp.WithOAuth(ctx, oauthConfig),
)
```

## Complete Example

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"

    "github.com/redpanda-data/ai-sdk-go/tool"
    "github.com/redpanda-data/ai-sdk-go/tool/mcp"
)

func main() {
    ctx := context.Background()

    // Create tool registry
    registry := tool.NewRegistry(tool.RegistryConfig{})

    // Create stdio transport
    transport := mcp.NewStdioTransport(
        "npx",
        []string{"-y", "@modelcontextprotocol/server-everything"},
        nil,
    )

    // Create client with registry and auto-sync
    client, err := mcp.NewClient(
        "mcp-server",
        transport,
        mcp.WithRegistry(registry),
        mcp.WithAutoSync(5*time.Minute),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Start client
    if err := client.Start(ctx); err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // List tools
    tools, err := client.ListTools(ctx)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Discovered %d tools:\n", len(tools))
    for _, tool := range tools {
        fmt.Printf("- %s: %s\n", tool.NamespacedName, tool.Description)
    }

    // Execute a tool
    if len(tools) > 0 {
        args := json.RawMessage(`{}`)
        result, err := client.ExecuteTool(ctx, tools[0].ServerToolName, args)
        if err != nil {
            log.Printf("Error executing tool: %v", err)
        } else {
            fmt.Printf("Result: %s\n", string(result))
        }
    }
}
```

## MCP Specification Compliance

This client implements the MCP specification with the following OAuth 2.1 requirements:

- **Authorization Code Grant**: PKCE (RFC 7636) with S256 challenge method
- **Client Credentials Grant**: Standard OAuth 2.1 M2M flow (PKCE not applicable)
- **Resource Indicators**: RFC 8707 for token scoping to specific MCP servers
- **Token Management**: Automatic token refresh and secure storage via `golang.org/x/oauth2`

For more information, see the [MCP Authorization Specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/authorization).

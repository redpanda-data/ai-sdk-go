// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// ErrPartitionMismatch prevents a captured catalog from executing as another identity.
var ErrPartitionMismatch = errors.New("mcp: credential partition changed")

// RegistryOption configures a contextual MCP registry.
type RegistryOption func(*registryConfig)

type registryConfig struct {
	credentials     CredentialsProvider
	httpClient      *http.Client
	maxConnections  int
	idleTimeout     time.Duration
	maxToolCacheTTL time.Duration
	now             func() time.Time
	toolOptions     []tool.Option
}

// WithCredentials requires the application to explicitly resolve an authorization
// partition, including for public servers. PartitionKey alone selects shared state.
func WithCredentials(provider CredentialsProvider) RegistryOption {
	return func(c *registryConfig) { c.credentials = provider }
}

// WithRegistryHTTPClient preserves a custom client's transport, TLS configuration
// and timeouts. A shared cookie Jar is rejected; credentials belong in the provider.
func WithRegistryHTTPClient(client *http.Client) RegistryOption {
	return func(c *registryConfig) { c.httpClient = client }
}

// WithRegistryMaxConnections bounds resident authorization partitions. Operations
// wait for capacity when all entries are leased, respecting cancellation.
func WithRegistryMaxConnections(n int) RegistryOption {
	return func(c *registryConfig) { c.maxConnections = n }
}

// WithRegistryIdleTimeout expires unleased entries on pool access. Zero disables
// idle expiration. Capacity eviction still removes the least recently used entry.
func WithRegistryIdleTimeout(timeout time.Duration) RegistryOption {
	return func(c *registryConfig) { c.idleTimeout = timeout }
}

// WithRegistryMaxToolCacheTTL caps positive server catalog TTLs. The default is
// 30 seconds. Zero disables catalog caching; negative durations are rejected.
// Server TTLs of zero or less are never extended. Server change notifications
// invalidate cached tools immediately; discovery itself remains on demand.
func WithRegistryMaxToolCacheTTL(ttl time.Duration) RegistryOption {
	return func(c *registryConfig) { c.maxToolCacheTTL = ttl }
}

// WithRegistryToolOptions applies native registration policies to every discovered
// tool, including timeouts, grouping, deferred loading and response-size limits.
// The default group name is the server ID; a supplied group overrides it.
func WithRegistryToolOptions(opts ...tool.Option) RegistryOption {
	return func(c *registryConfig) { c.toolOptions = append(c.toolOptions, opts...) }
}

// Registry discovers tools on demand using caller-context credentials. SDK clients
// and negotiated protocol state are isolated by the application-owned partition
// key. Catalog caching honors server TTLs up to the configured maximum and
// invalidates on server change notifications. No periodic discovery is performed.
// Tool results contain the complete MCP CallToolResult object (content,
// structuredContent and metadata), unlike legacy Client.ExecuteTool's content
// array. MCP isError results retain that object and propagate tool.ResultError.
// Call Close or Shutdown to release resident connections.
type Registry struct {
	serverID string
	endpoint *url.URL
	config   registryConfig

	mu      sync.Mutex
	entries map[string]*registryEntry
	// closing counts sessions detached from entries whose Close has not
	// returned, including retired entries still leased (draining). They hold
	// capacity: len(entries)+closing never exceeds maxConnections.
	closing  int
	draining int
	waiting  int
	changed  chan struct{}
	closed   bool
	ctx      context.Context //nolint:containedctx // Owns the registry lifetime, not an individual caller.
	cancel   context.CancelFunc
	wg       sync.WaitGroup
	cleanup  sync.WaitGroup // detached session closes started by closeDetached
	// cleanupErrs holds detached close failures, reported by Shutdown.
	cleanupErrs registryCleanupErrors
	done        chan struct{}
	closeErr    error
}

var _ tool.Registry = (*Registry)(nil)

// NewRegistry constructs a lazy HTTP MCP registry. It performs no network I/O.
// The pinned SDK negotiates modern server/discover and legacy initialize protocols.
func NewRegistry(serverID, endpoint string, opts ...RegistryOption) (*Registry, error) {
	if serverID == "" {
		return nil, errors.New("mcp: serverID cannot be empty")
	}

	parsed, err := url.Parse(endpoint)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil || parsed.Fragment != "" {
		return nil, errors.New("mcp: endpoint must be an absolute HTTP(S) URL without userinfo or fragment")
	}

	cfg := registryConfig{maxConnections: 64, idleTimeout: 5 * time.Minute, maxToolCacheTTL: 30 * time.Second, now: time.Now}

	for i, opt := range opts {
		if opt == nil {
			return nil, fmt.Errorf("mcp: registry option %d is nil", i)
		}

		opt(&cfg)
	}

	if cfg.credentials == nil {
		return nil, errors.New("mcp: WithCredentials is required")
	}

	if cfg.maxConnections <= 0 || cfg.idleTimeout < 0 || cfg.maxToolCacheTTL < 0 {
		return nil, errors.New("mcp: invalid registry connection limits")
	}

	if cfg.httpClient != nil && cfg.httpClient.Jar != nil {
		return nil, errors.New("mcp: shared HTTP cookie jars are not supported")
	}

	if cfg.httpClient != nil {
		copied := *cfg.httpClient
		cfg.httpClient = &copied
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &Registry{
		serverID: serverID, endpoint: parsed, config: cfg,
		entries: make(map[string]*registryEntry), changed: make(chan struct{}),
		ctx: ctx, cancel: cancel, done: make(chan struct{}),
	}, nil
}

// Register is unsupported: the remote server owns membership.
func (*Registry) Register(tool.Tool, ...tool.Option) error { return tool.ErrRegistryReadOnly }

// Unregister is unsupported: the remote server owns membership.
func (*Registry) Unregister(string) error { return tool.ErrRegistryReadOnly }

// Snapshot captures definitions, raw server routes and native execution policies.
// Execution resolves current credentials but requires the captured partition key.
func (r *Registry) Snapshot(ctx context.Context) (tool.Registry, error) {
	ctx, end, err := r.begin(ctx)
	if err != nil {
		return nil, err
	}
	defer end()

	credentials, err := r.credentials(ctx, "")
	if err != nil {
		return nil, err
	}

	entry, release, err := r.acquire(ctx, credentials.PartitionKey)
	if err != nil {
		return nil, err
	}
	defer release()

	catalog, err := r.catalog(ctx, entry)
	if err != nil {
		return nil, fmt.Errorf("mcp: list tools: %w", err)
	}

	native := tool.NewRegistry(tool.RegistryConfig{})

	for _, remote := range catalog {
		if !validMCPToolName.MatchString(remote.Name) {
			continue
		}

		parameters, err := json.Marshal(remote.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q tool %q schema: %w", r.serverID, remote.Name, err)
		}

		bindings, err := registryInputSchema(parameters)
		if err != nil {
			return nil, fmt.Errorf("mcp: server %q tool %q: %w", r.serverID, remote.Name, err)
		}

		wrapped := &registryTool{
			registry: r, partition: credentials.PartitionKey, rawName: remote.Name, parameterBindings: bindings,
			definition: llm.ToolDefinition{
				Name:        mangleHeadIfTooLong(r.serverID+"__"+remote.Name, maxToolNameLen),
				Description: remote.Description, Parameters: parameters, Type: llm.ToolTypeExtension,
			},
		}

		opts := append([]tool.Option{tool.WithGroup(llm.ToolGroup{Name: r.serverID})}, r.config.toolOptions...)
		if err := native.Register(wrapped, opts...); err != nil {
			return nil, fmt.Errorf("mcp: register discovered tool: %w", err)
		}
	}

	return native.Snapshot(ctx)
}

// List discovers the calling context's catalog.
func (r *Registry) List(ctx context.Context) ([]llm.ToolDefinition, error) {
	captured, err := r.Snapshot(ctx)
	if err != nil {
		return nil, err
	}

	return captured.List(ctx)
}

// Get captures the calling context's tool and authorization partition.
func (r *Registry) Get(ctx context.Context, name string) (tool.Tool, error) {
	captured, err := r.Snapshot(ctx)
	if err != nil {
		return nil, err
	}

	return captured.Get(ctx, name)
}

// Execute discovers and executes under the same calling context. Agents should
// use Snapshot once per model step to keep discovery and execution routes fixed.
func (r *Registry) Execute(ctx context.Context, req *llm.ToolRequestPart) (*llm.ToolResponsePart, error) {
	if req == nil {
		return nil, tool.ErrToolRequestNil
	}

	captured, err := r.Snapshot(ctx)
	if err != nil {
		return nil, err
	}

	return captured.Execute(ctx, req)
}

// ExecuteAll captures once and delegates scheduling and policies to the native registry.
func (r *Registry) ExecuteAll(ctx context.Context, reqs []*llm.ToolRequestPart, opts ...tool.BatchOption) []*llm.ToolResponsePart {
	if len(reqs) == 0 {
		return []*llm.ToolResponsePart{}
	}

	captured, err := r.Snapshot(ctx)
	if err == nil {
		return captured.ExecuteAll(ctx, reqs, opts...)
	}

	responses := make([]*llm.ToolResponsePart, len(reqs))
	for i, req := range reqs {
		if req == nil {
			req = &llm.ToolRequestPart{}
		}

		payload, _ := json.Marshal(map[string]string{"error": err.Error()}) //nolint:errchkjson // Strings always marshal.
		responses[i] = &llm.ToolResponsePart{ID: req.ID, Name: req.Name, Result: payload, IsError: true}
	}

	return responses
}

func (r *Registry) credentials(ctx context.Context, partition string) (Credentials, error) {
	if err := ctx.Err(); err != nil {
		return Credentials{}, err
	}

	ctx, err := registryCredentialContext(ctx)
	if err != nil {
		return Credentials{}, err
	}

	credentials, err := r.config.credentials(ctx)
	if err != nil {
		return Credentials{}, fmt.Errorf("mcp: resolve credentials: %w", err)
	}

	if err := credentials.Validate(); err != nil {
		return Credentials{}, err
	}

	if partition != "" && credentials.PartitionKey != partition {
		return Credentials{}, ErrPartitionMismatch
	}

	credentials.Headers = credentials.Headers.Clone()

	return credentials, nil
}

type registryTool struct {
	registry          *Registry
	partition         string
	rawName           string
	definition        llm.ToolDefinition
	parameterBindings []registryParameterBinding
}

func (t *registryTool) Definition() llm.ToolDefinition {
	def := t.definition
	def.Parameters = append(json.RawMessage(nil), def.Parameters...)

	return def
}

func (t *registryTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	ctx, end, err := t.registry.begin(ctx)
	if err != nil {
		return nil, err
	}
	defer end()

	if _, err := t.registry.credentials(ctx, t.partition); err != nil {
		return nil, err
	}

	var arguments map[string]any
	if len(args) != 0 {
		if err := json.Unmarshal(args, &arguments); err != nil {
			return nil, fmt.Errorf("mcp: tool arguments must be a JSON object: %w", err)
		}
	}

	if arguments == nil {
		arguments = make(map[string]any)
	}

	headers, err := registryParameterHeaders(t.parameterBindings, arguments)
	if err != nil {
		return nil, err
	}

	entry, release, err := t.registry.acquire(ctx, t.partition)
	if err != nil {
		return nil, err
	}
	defer release()

	ctx = context.WithValue(ctx, registryParameterHeadersKey{}, headers)

	result, err := entry.session.CallTool(ctx, &sdkmcp.CallToolParams{Name: t.rawName, Arguments: arguments})
	if err != nil {
		t.registry.retireClosed(entry, err)

		return nil, fmt.Errorf("mcp: call tool: %w", err)
	}

	payload, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("mcp: encode tool result: %w", err)
	}

	if result.IsError {
		return payload, &tool.ResultError{Result: payload}
	}

	return payload, nil
}

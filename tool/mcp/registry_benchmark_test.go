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
	"iter"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const registryBenchmarkListMethod = "tools/list"

const registryBenchmarkParameters = `{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"limit":{"type":"integer"}},"required":["query"],"additionalProperties":false}`

// Loopback transport, one-page catalogs, static fake credentials and no pushes.
// Uncached excludes connection setup; rotating eight scopes through four pool
// entries intentionally includes steady-state eviction and connection setup.
func BenchmarkMCPRegistrySnapshot(b *testing.B) {
	for _, size := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("tools=%d", size), func(b *testing.B) {
			b.Run("Native", func(b *testing.B) {
				registry := newRegistryBenchmarkNative(b, size)
				ctx := b.Context()
				checkRegistryBenchmarkSnapshot(ctx, b, registry, size)
				b.ReportAllocs()

				for b.Loop() {
					if _, err := registry.Snapshot(ctx); err != nil {
						b.Fatal(err)
					}
				}

				b.ReportMetric(0, "tools/list/op")
				b.ReportMetric(0, "server/discover/op")
			})

			for _, cache := range []bool{true, false} {
				name := "WarmCached"
				if !cache {
					name = "Uncached"
				}

				b.Run(name, func(b *testing.B) {
					var opts []RegistryOption
					if !cache {
						opts = append(opts, WithRegistryMaxToolCacheTTL(0))
					}

					fixture := newRegistryBenchmarkMCP(b, size, opts...)
					ctx := registryBenchmarkCaller(b.Context(), 0)
					checkRegistryBenchmarkSnapshot(ctx, b, fixture.registry, size)

					if cache {
						fixture.checkCacheHit(ctx, b, size)
					}

					lists, discoveries := fixture.counts()

					b.ReportAllocs()

					for b.Loop() {
						if _, err := fixture.registry.Snapshot(ctx); err != nil {
							b.Fatal(err)
						}
					}

					fixture.report(b, lists, discoveries)
				})
			}

			b.Run("ParallelSamePartition", func(b *testing.B) {
				fixture := newRegistryBenchmarkMCP(b, size)
				ctx := registryBenchmarkCaller(b.Context(), 0)
				checkRegistryBenchmarkSnapshot(ctx, b, fixture.registry, size)
				fixture.checkCacheHit(ctx, b, size)
				lists, discoveries := fixture.counts()

				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						if _, err := fixture.registry.Snapshot(ctx); err != nil {
							b.Error(err)

							return
						}
					}
				})
				b.StopTimer()
				fixture.report(b, lists, discoveries)
			})

			b.Run("Rotating8PartitionsMax4", func(b *testing.B) {
				fixture := newRegistryBenchmarkMCP(b, size, WithRegistryMaxConnections(4))

				contexts := make([]context.Context, 0, 8)
				parent := b.Context()

				for i := range 8 {
					ctx := registryBenchmarkCaller(parent, i)
					contexts = append(contexts, ctx)
					checkRegistryBenchmarkSnapshot(ctx, b, fixture.registry, size)
				}

				fixture.checkCacheHit(contexts[len(contexts)-1], b, size)
				lists, discoveries := fixture.counts()

				b.ReportAllocs()

				i := 0

				for b.Loop() {
					if _, err := fixture.registry.Snapshot(contexts[i%len(contexts)]); err != nil {
						b.Fatal(err)
					}

					i++
				}

				fixture.report(b, lists, discoveries)
			})
		})
	}
}

// A complete one-turn agent invocation with fresh session metadata and a
// stateless zero-latency fake model. This does not measure LLM or tool execution.
func BenchmarkMCPRegistryAgentStep(b *testing.B) {
	for _, size := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("tools=%d", size), func(b *testing.B) {
			for _, remote := range []bool{false, true} {
				name := "Native"
				if remote {
					name = "WarmMCP"
				}

				b.Run(name, func(b *testing.B) {
					ctx := registryBenchmarkCaller(b.Context(), 0)
					var registry tool.Registry
					var fixture *registryBenchmarkFixture

					if remote {
						fixture = newRegistryBenchmarkMCP(b, size)
						registry = fixture.registry
						checkRegistryBenchmarkSnapshot(ctx, b, registry, size)
						fixture.checkCacheHit(ctx, b, size)
					} else {
						registry = newRegistryBenchmarkNative(b, size)
					}

					ag, err := llmagent.New("benchmark", "Use the available tools.", registryBenchmarkModel{tools: size}, llmagent.WithTools(registry), llmagent.WithMaxTurns(1))
					require.NoError(b, err)
					runRegistryBenchmarkAgent(ctx, b, ag)

					var lists, discoveries int64

					if fixture != nil {
						lists, discoveries = fixture.counts()
					}

					b.ReportAllocs()

					for b.Loop() {
						runRegistryBenchmarkAgent(ctx, b, ag)
					}

					if fixture != nil {
						fixture.report(b, lists, discoveries)
					} else {
						b.ReportMetric(0, "tools/list/op")
						b.ReportMetric(0, "server/discover/op")
					}
				})
			}
		})
	}
}

type registryBenchmarkFixture struct {
	registry    *Registry
	lists       atomic.Int64
	discoveries atomic.Int64
}

func newRegistryBenchmarkMCP(b *testing.B, size int, opts ...RegistryOption) *registryBenchmarkFixture {
	b.Helper()

	fixture := new(registryBenchmarkFixture)
	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "benchmark", Version: "1"}, &sdkmcp.ServerOptions{
		PageSize: size, Capabilities: &sdkmcp.ServerCapabilities{Tools: &sdkmcp.ToolCapabilities{}},
	})

	for i := range size {
		server.AddTool(&sdkmcp.Tool{Name: fmt.Sprintf("tool_%04d", i), Description: "Search caller records", InputSchema: json.RawMessage(registryBenchmarkParameters)},
			func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
				return &sdkmcp.CallToolResult{}, nil
			})
	}

	server.AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			switch method {
			case registryBenchmarkListMethod:
				fixture.lists.Add(1)
			case "server/discover":
				fixture.discoveries.Add(1)
			}

			result, err := next(ctx, method, req)
			if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil {
				listed.TTLMs = int(time.Hour.Milliseconds())
			}

			return result, err
		}
	})

	handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
	peer := httptest.NewServer(handler)
	b.Cleanup(peer.Close)

	transport := new(http.Transport)
	b.Cleanup(transport.CloseIdleConnections)
	options := append([]RegistryOption{
		WithCredentials(registryBenchmarkCredentials), WithRegistryMaxToolCacheTTL(time.Hour),
		WithRegistryIdleTimeout(0), WithRegistryHTTPClient(&http.Client{Transport: transport}),
	}, opts...)
	registry, err := NewRegistry("bench", peer.URL, options...)
	require.NoError(b, err)
	b.Cleanup(func() { require.NoError(b, registry.Close()) })

	fixture.registry = registry

	return fixture
}

func (f *registryBenchmarkFixture) counts() (int64, int64) {
	return f.lists.Load(), f.discoveries.Load()
}

func (f *registryBenchmarkFixture) checkCacheHit(ctx context.Context, b *testing.B, size int) {
	b.Helper()

	lists, discoveries := f.counts()
	checkRegistryBenchmarkSnapshot(ctx, b, f.registry, size)
	require.Equal(b, lists, f.lists.Load(), "the advertised positive server TTL must produce a real cache hit")
	require.Equal(b, discoveries, f.discoveries.Load())
}

func (f *registryBenchmarkFixture) report(b *testing.B, lists, discoveries int64) {
	b.Helper()
	b.StopTimer()
	b.ReportMetric(float64(f.lists.Load()-lists)/float64(b.N), "tools/list/op")
	b.ReportMetric(float64(f.discoveries.Load()-discoveries)/float64(b.N), "server/discover/op")
}

func checkRegistryBenchmarkSnapshot(ctx context.Context, b *testing.B, registry tool.Registry, size int) {
	b.Helper()

	captured, err := registry.Snapshot(ctx)
	require.NoError(b, err)
	definitions, err := captured.List(ctx)
	require.NoError(b, err)
	require.Len(b, definitions, size)
}

type registryBenchmarkScopeKey struct{}

func registryBenchmarkCaller(ctx context.Context, scope int) context.Context {
	return context.WithValue(ctx, registryBenchmarkScopeKey{}, fmt.Sprintf("scope-%d", scope))
}

func registryBenchmarkCredentials(ctx context.Context) (Credentials, error) {
	scope, ok := ctx.Value(registryBenchmarkScopeKey{}).(string)
	if !ok {
		return Credentials{}, errors.New("missing benchmark caller")
	}

	return Credentials{PartitionKey: scope, Headers: http.Header{"Authorization": {"Bearer " + scope}}}, nil
}

type registryBenchmarkTool struct{ definition llm.ToolDefinition }

func (t registryBenchmarkTool) Definition() llm.ToolDefinition { return t.definition }

func (registryBenchmarkTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

func newRegistryBenchmarkNative(b *testing.B, size int) tool.Registry {
	b.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	for i := range size {
		require.NoError(b, registry.Register(registryBenchmarkTool{definition: llm.ToolDefinition{
			Name: fmt.Sprintf("bench__tool_%04d", i), Description: "Search caller records",
			Parameters: json.RawMessage(registryBenchmarkParameters), Type: llm.ToolTypeExtension,
		}}, tool.WithGroup(llm.ToolGroup{Name: "bench"})))
	}

	return registry
}

type registryBenchmarkModel struct{ tools int }

func (registryBenchmarkModel) Name() string { return "benchmark" }

func (registryBenchmarkModel) Provider() llm.ProviderID { return "benchmark" }

func (registryBenchmarkModel) Capabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{Tools: true, Streaming: true, SystemPrompts: true}
}

func (registryBenchmarkModel) Constraints() llm.ModelConstraints {
	return llm.ModelConstraints{MaxInputTokens: 1 << 20, MaxOutputTokens: 4096}
}

func (m registryBenchmarkModel) Generate(_ context.Context, req *llm.Request) (*llm.Response, error) {
	if len(req.Tools) != m.tools {
		return nil, fmt.Errorf("benchmark model received %d tools, want %d", len(req.Tools), m.tools)
	}

	return &llm.Response{Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("done")), FinishReason: llm.FinishReasonStop}, nil
}

func (m registryBenchmarkModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		response, err := m.Generate(ctx, req)
		if err != nil {
			yield(nil, err)

			return
		}

		yield(llm.StreamEndEvent{Response: response}, nil)
	}
}

func runRegistryBenchmarkAgent(ctx context.Context, b *testing.B, ag *llmagent.LLMAgent) {
	b.Helper()

	state := &session.State{ID: "benchmark", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Answer briefly."))}}
	invocation := agent.NewInvocationMetadata(state, agent.Info{})

	for _, err := range ag.Run(ctx, invocation) {
		if err != nil {
			b.Fatal(err)
		}
	}
}

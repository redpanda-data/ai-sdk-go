package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"sync"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

// mockMCPServer implements a simple MCP server for testing.
type mockMCPServer struct {
	server          *sdkmcp.Server
	clientTransport sdkmcp.Transport
	serverTransport sdkmcp.Transport
	tools           []*sdkmcp.Tool
	toolExecutor    func(name string, args map[string]any) ([]sdkmcp.Content, error)
	mu              sync.Mutex
	started         bool
	serverSession   *sdkmcp.ServerSession
}

// newMockMCPServer creates a mock MCP server with test tools.
func newMockMCPServer() *mockMCPServer {
	ms := &mockMCPServer{
		tools: []*sdkmcp.Tool{
			{
				Name:        "echo",
				Description: "Echoes back the input",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"message": map[string]any{
							"type":        "string",
							"description": "Message to echo",
						},
					},
					"required": []string{"message"},
				},
			},
			{
				Name:        "add",
				Description: "Adds two numbers",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"a": map[string]any{"type": "number"},
						"b": map[string]any{"type": "number"},
					},
					"required": []string{"a", "b"},
				},
			},
		},
	}

	ms.server = sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "mock-test-server",
		Title:   "Mock Test Server",
		Version: "1.0.0",
	}, &sdkmcp.ServerOptions{
		HasTools: true,
	})

	// Register tools with handlers
	for _, t := range ms.tools {
		ms.server.AddTool(t, func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			// Parse arguments from raw JSON
			var args map[string]any
			if len(req.Params.Arguments) > 0 {
				err := json.Unmarshal(req.Params.Arguments, &args)
				if err != nil {
					//nolint:nilerr // Not a protocol error, we want to let the LLM know about this error
					return &sdkmcp.CallToolResult{
						Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "invalid arguments"}},
						IsError: true,
					}, nil
				}
			}

			if ms.toolExecutor != nil {
				content, err := ms.toolExecutor(req.Params.Name, args)
				if err != nil {
					//nolint:nilerr // Not a protocol error, we want to let the LLM know about this error
					return &sdkmcp.CallToolResult{
						Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: err.Error()}},
						IsError: true,
					}, nil
				}

				return &sdkmcp.CallToolResult{Content: content}, nil
			}

			// Default executor
			switch req.Params.Name {
			case "echo":
				msg, ok := args["message"].(string)
				if !ok {
					return &sdkmcp.CallToolResult{
						Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "message must be a string"}},
						IsError: true,
					}, nil
				}

				return &sdkmcp.CallToolResult{
					Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: msg}},
				}, nil
			case "add":
				a, ok := args["a"].(float64)
				if !ok {
					return &sdkmcp.CallToolResult{
						Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "a must be a number"}},
						IsError: true,
					}, nil
				}

				b, ok := args["b"].(float64)
				if !ok {
					return &sdkmcp.CallToolResult{
						Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "b must be a number"}},
						IsError: true,
					}, nil
				}

				result := a + b

				return &sdkmcp.CallToolResult{
					Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: jsonFloat(result)}},
				}, nil
			default:
				return &sdkmcp.CallToolResult{
					Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "unknown tool"}},
					IsError: true,
				}, nil
			}
		})
	}

	return ms
}

func jsonFloat(f float64) string {
	b, _ := json.Marshal(f) //nolint:errchkjson // test helper - marshaling float64 never fails
	return string(b)
}

func (ms *mockMCPServer) start(ctx context.Context) (sdkmcp.Transport, error) {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if ms.started {
		return nil, errors.New("server already started")
	}

	// Use SDK's built-in in-memory transports
	ct, st := sdkmcp.NewInMemoryTransports()
	ms.clientTransport = ct
	ms.serverTransport = st

	// Start server in background
	go func() {
		ss, err := ms.server.Connect(ctx, st, nil)
		if err != nil {
			return
		}

		ms.mu.Lock()
		ms.serverSession = ss
		ms.mu.Unlock()

		_ = ss.Wait() // Wait for server to finish
	}()

	ms.started = true

	return ct, nil
}

func (ms *mockMCPServer) stop() error {
	ms.mu.Lock()
	defer ms.mu.Unlock()

	if !ms.started {
		return nil
	}

	if ms.serverSession != nil {
		_ = ms.serverSession.Close()
	}

	ms.started = false

	return nil
}

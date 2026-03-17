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
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// TestTraceHTTPPropagation verifies that OTEL trace context is propagated
// through HTTP headers from the MCP client to the MCP server.
func TestTraceHTTPPropagation(t *testing.T) { //nolint:paralleltest // spawns HTTP server with shared state
	// Setup OTEL tracer
	tp := sdktrace.NewTracerProvider()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(
		propagation.NewCompositeTextMapPropagator(
			propagation.TraceContext{},
			propagation.Baggage{},
		),
	)

	defer func() { _ = tp.Shutdown(context.Background()) }()

	tracer := tp.Tracer("test-tracer")

	// Variables to capture trace information
	var clientTraceID trace.TraceID
	var mu sync.Mutex
	var capturedHeaders http.Header

	// Create MCP server
	mcpServer := sdkmcp.NewServer(&sdkmcp.Implementation{
		Name:    "test-http-server",
		Title:   "Test HTTP Server",
		Version: "1.0.0",
	}, &sdkmcp.ServerOptions{
		HasTools: true,
	})

	// Add a simple echo tool
	echoTool := &sdkmcp.Tool{
		Name:        "echo",
		Description: "Echoes back the message",
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
	}

	mcpServer.AddTool(echoTool, func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		// Try to extract trace from server context
		serverSpanCtx := trace.SpanContextFromContext(ctx)
		t.Logf("Server-side span context - TraceID: %s, Valid: %v",
			serverSpanCtx.TraceID(), serverSpanCtx.IsValid())

		var args map[string]any
		if len(req.Params.Arguments) > 0 {
			if err := json.Unmarshal(req.Params.Arguments, &args); err != nil {
				//nolint:nilerr // Tool errors are returned in result, not as Go errors
				return &sdkmcp.CallToolResult{
					Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "invalid args"}},
					IsError: true,
				}, nil
			}
		}

		msg, _ := args["message"].(string)

		return &sdkmcp.CallToolResult{
			Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "echo: " + msg}},
		}, nil
	})

	// Create HTTP handler with middleware to capture headers
	mcpHandler := sdkmcp.NewStreamableHTTPHandler(
		func(_ *http.Request) *sdkmcp.Server { return mcpServer },
		&sdkmcp.StreamableHTTPOptions{},
	)

	// Wrap with middleware to capture incoming HTTP headers
	headerCaptureHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()

		capturedHeaders = r.Header.Clone()

		mu.Unlock()

		t.Logf("Incoming HTTP request headers:")

		for k, v := range r.Header {
			t.Logf("  %s: %v", k, v)
		}

		mcpHandler.ServeHTTP(w, r)
	})

	// Start test HTTP server
	httpServer := httptest.NewServer(headerCaptureHandler)
	defer httpServer.Close()

	t.Logf("Test HTTP server started at: %s", httpServer.URL)

	// Create MCP client with instrumented HTTP client
	httpClient := &http.Client{
		Transport: otelhttp.NewTransport(http.DefaultTransport),
	}

	factory := NewStreamableTransport(
		httpServer.URL,
		WithHTTPClient(httpClient),
	)

	client, err := NewClient("test-http-server", factory)
	require.NoError(t, err)

	ctx := context.Background()
	require.NoError(t, client.Start(ctx))

	defer func() { _ = client.Close() }()

	// Create a parent span to simulate agent tool execution
	ctx, parentSpan := tracer.Start(ctx, "parent-tool-execution")
	defer parentSpan.End()

	// Capture client-side trace ID
	clientSpanCtx := trace.SpanContextFromContext(ctx)
	clientTraceID = clientSpanCtx.TraceID()
	require.True(t, clientTraceID.IsValid(), "client trace ID should be valid")

	t.Logf("Client-side span context - TraceID: %s, SpanID: %s",
		clientTraceID, clientSpanCtx.SpanID())

	// Execute tool with traced context
	args := json.RawMessage(`{"message":"test"}`)
	result, err := client.ExecuteTool(ctx, "test-http-server__echo", args)
	require.NoError(t, err)
	require.NotNil(t, result)
	t.Logf("Tool execution result: %s", string(result))

	// Check captured HTTP headers
	mu.Lock()

	headers := capturedHeaders

	mu.Unlock()

	require.NotNil(t, headers, "should have captured HTTP headers")

	// Check for traceparent header (W3C Trace Context)
	traceparent := headers.Get("Traceparent")
	t.Logf("Captured traceparent header: %q", traceparent)

	// Assert that traceparent header is present
	assert.NotEmpty(t, traceparent, "traceparent header should be present in HTTP request")

	// Parse traceparent header to extract trace ID
	// Format: 00-{trace-id}-{span-id}-{flags}
	if traceparent != "" {
		// Simple parsing - just check if our trace ID is in the header
		traceIDStr := clientTraceID.String()
		assert.Contains(t, traceparent, traceIDStr,
			"traceparent header should contain client trace ID")
		t.Logf("✓ Trace ID propagated correctly in traceparent header")
	} else {
		t.Error("✗ traceparent header is missing - trace context not propagated")
	}

	// Check tracestate header (optional but good to verify)
	tracestate := headers.Get("Tracestate")
	t.Logf("Captured tracestate header: %q", tracestate)
}

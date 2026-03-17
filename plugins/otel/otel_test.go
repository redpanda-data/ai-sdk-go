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

package otel_test

import (
	"context"
	"encoding/json"
	"errors"
	"iter"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	pluginotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// mockModelInfo implements llm.ModelInfo for testing.
type mockModelInfo struct {
	name     string
	provider string
}

func (m *mockModelInfo) Name() string     { return m.name }
func (m *mockModelInfo) Provider() string { return m.provider }
func (m *mockModelInfo) Capabilities() llm.ModelCapabilities {
	return llm.ModelCapabilities{}
}

func (m *mockModelInfo) Constraints() llm.ModelConstraints {
	return llm.ModelConstraints{}
}

// mockModelHandler implements agent.ModelCallHandler for testing.
type mockModelHandler struct {
	generateFn       func(ctx context.Context, req *llm.Request) (*llm.Response, error)
	generateEventsFn func(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error]
}

func (m *mockModelHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if m.generateFn != nil {
		return m.generateFn(ctx, req)
	}

	return &llm.Response{
		Message:      llm.Message{Role: llm.RoleAssistant},
		FinishReason: llm.FinishReasonStop,
		Usage: &llm.TokenUsage{
			InputTokens:  10,
			OutputTokens: 20,
		},
		ID: "resp-123",
	}, nil
}

func (m *mockModelHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	if m.generateEventsFn != nil {
		return m.generateEventsFn(ctx, req)
	}

	return func(yield func(llm.Event, error) bool) {
		yield(llm.StreamEndEvent{
			Response: &llm.Response{
				Message:      llm.Message{Role: llm.RoleAssistant},
				FinishReason: llm.FinishReasonStop,
				Usage: &llm.TokenUsage{
					InputTokens:  10,
					OutputTokens: 20,
				},
				ID: "resp-456",
			},
		}, nil)
	}
}

// setupTracer creates a test tracer provider with span recording.
func setupTracer() (*tracetest.InMemoryExporter, *sdktrace.TracerProvider) {
	exporter := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSyncer(exporter),
	)

	return exporter, tp
}

func TestTracingInterceptor_InterceptTurn_CreatesInvocationSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	// Simulate a single turn that completes
	reason, err := interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	require.NoError(t, err)
	assert.Equal(t, agent.FinishReasonStop, reason)

	// Check spans - should only have invocation span (no per-turn spans)
	spans := exporter.GetSpans()
	require.Len(t, spans, 1, "Expected 1 span: invocation only")

	invocationSpan := spans[0]

	// Span name includes agent name per OTel convention
	assert.Equal(t, "invoke_agent test-agent", invocationSpan.Name)

	// Check invocation attributes
	assertHasAttribute(t, invocationSpan.Attributes, "gen_ai.operation.name", "invoke_agent")
	assertHasAttribute(t, invocationSpan.Attributes, "gen_ai.conversation.id", "sess-123")
	assertHasAttribute(t, invocationSpan.Attributes, "gen_ai.agent.name", "test-agent")
}

func TestTracingInterceptor_InterceptTurn_MultipleTurns(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	// Turn 0 - continues
	reason, err := interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return "", nil // No finish reason = continue
	})
	require.NoError(t, err)
	assert.Empty(t, reason)

	// Increment turn (normally done by agent framework)
	agent.IncrementTurn(inv)

	// Turn 1 - completes
	reason, err = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})
	require.NoError(t, err)
	assert.Equal(t, agent.FinishReasonStop, reason)

	// Check spans - should only have invocation span (created on turn 0, ended on turn 1)
	spans := exporter.GetSpans()
	require.Len(t, spans, 1, "Expected 1 invocation span across multiple turns")

	invocationSpan := spans[0]
	assertHasAttribute(t, invocationSpan.Attributes, "gen_ai.operation.name", "invoke_agent")
}

func TestTracingInterceptor_InterceptTurn_ErrorRecording(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()
	testErr := errors.New("turn failed")

	reason, err := interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return "", testErr
	})

	require.ErrorIs(t, err, testErr)
	assert.Empty(t, reason)

	spans := exporter.GetSpans()
	require.Len(t, spans, 1, "Expected only invocation span")

	// Invocation span should have error status
	invocationSpan := spans[0]
	assert.Equal(t, codes.Error, invocationSpan.Status.Code)
}

func TestTracingInterceptor_InterceptModel_Generate(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	// First create a turn to establish parent span
	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Within turn, intercept model call
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		resp, err := handler.Generate(ctx, &llm.Request{})
		require.NoError(t, err)
		assert.NotNil(t, resp)

		return agent.FinishReasonStop, nil
	})

	// Find the chat span (name format: "chat {model_name}")
	spans := exporter.GetSpans()
	var chatSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, chatSpan, "Expected chat span")

	// Check attributes
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.operation.name", "chat")
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.request.model", "gpt-4")
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.provider.name", "openai")
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.response.id", "resp-123")
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.usage.input_tokens", int64(10))
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.usage.output_tokens", int64(20))

	// Verify SpanKind is Client for model calls
	assert.Equal(t, trace.SpanKindClient, chatSpan.SpanKind)
}

func TestTracingInterceptor_InterceptModel_GenerateEvents(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "test-model", provider: "test"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})

		// Consume all events
		for evt, err := range handler.GenerateEvents(ctx, &llm.Request{}) {
			require.NoError(t, err)
			require.NotNil(t, evt)
		}

		return agent.FinishReasonStop, nil
	})

	// Find the chat span (name format: "chat {model_name}")
	spans := exporter.GetSpans()
	var chatSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, chatSpan, "Expected chat span")

	// Streaming should also capture response ID from StreamEndEvent
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.response.id", "resp-456")
}

func TestTracingInterceptor_InterceptToolExecution(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "get_weather",
			ID:        "tool-call-123",
			Arguments: json.RawMessage(`{"city": "Seattle"}`),
		}

		toolInfo := &agent.ToolCallInfo{Inv: inv, Req: req}
		resp, err := interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`"Sunny, 72F"`)}, nil
			})
		require.NoError(t, err)
		assert.NotNil(t, resp)

		return agent.FinishReasonStop, nil
	})

	// Find the tool span (name format: "execute_tool {tool_name}")
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Check attributes use gen_ai.tool.* namespace
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.operation.name", "execute_tool")
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.name", "get_weather")
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.call.id", "tool-call-123")

	// Verify SpanKind is Internal for tool calls
	assert.Equal(t, trace.SpanKindInternal, toolSpan.SpanKind)
}

//nolint:dupl // Similar to WithRecordOutputs test but tests different functionality
func TestTracingInterceptor_InterceptToolExecution_WithRecordInputs(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordInputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "get_weather",
			ID:        "tool-call-123",
			Arguments: json.RawMessage(`{"city": "Seattle"}`),
		}

		toolInfo := &agent.ToolCallInfo{Inv: inv, Req: req}
		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`{"temp": "72F"}`)}, nil
			})

		return agent.FinishReasonStop, nil
	})

	// Find the tool span (name format: "execute_tool {tool_name}")
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan)

	// With recordInputs=true, arguments should be recorded as span attribute (not event)
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.call.arguments", `{"city": "Seattle"}`)
}

//nolint:dupl // Similar to WithRecordInputs test but tests different functionality
func TestTracingInterceptor_InterceptToolExecution_WithRecordOutputs(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordOutputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "get_weather",
			ID:        "tool-call-123",
			Arguments: json.RawMessage(`{"city": "Seattle"}`),
		}

		toolInfo := &agent.ToolCallInfo{Inv: inv, Req: req}
		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`{"temperature":"72F","conditions":"sunny"}`)}, nil
			})

		return agent.FinishReasonStop, nil
	})

	// Find the tool span (name format: "execute_tool {tool_name}")
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan)

	// With recordOutputs=true, result should be recorded as span attribute (not event)
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.call.result", `{"temperature":"72F","conditions":"sunny"}`)
}

func TestTracingInterceptor_InterceptToolExecution_Error(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()
	toolErr := errors.New("tool execution failed")

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{Name: "failing_tool", ID: "tool-123"}

		toolInfo := &agent.ToolCallInfo{Inv: inv, Req: req}
		_, err := interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return nil, toolErr
			})
		require.ErrorIs(t, err, toolErr)

		return agent.FinishReasonStop, nil
	})

	// Find the tool span (name format: "execute_tool {tool_name}")
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan)

	// Check error status
	assert.Equal(t, codes.Error, toolSpan.Status.Code)
}

func TestTracingInterceptor_InterceptToolExecution_ToolErrorResponse(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordOutputs(true), // Ensure result is NOT recorded for tool errors
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{Name: "query_logs", ID: "tool-call-abc", Arguments: json.RawMessage(`{"query":"errors"}`)}

		toolInfo := &agent.ToolCallInfo{Inv: inv, Req: req}
		resp, err := interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				// Tool returns error content (not a Go error) — like a 502 from an upstream API
				return &llm.ToolResponse{
					ID:    "tool-call-abc",
					Name:  "query_logs",
					Error: "query failed: upstream returned status 502",
				}, nil
			})
		require.NoError(t, err) // No Go error
		assert.NotNil(t, resp)

		return agent.FinishReasonStop, nil
	})

	// Find the tool span
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Span status SHOULD be Error (per MCP semconv: isError=true → span error)
	assert.Equal(t, codes.Error, toolSpan.Status.Code)
	assert.Contains(t, toolSpan.Status.Description, "query failed")

	// error.type SHOULD be "tool_error" (per MCP semconv well-known value)
	assertHasAttribute(t, toolSpan.Attributes, "error.type", "tool_error")

	// gen_ai.tool.call.result SHOULD NOT be recorded (spec: "if execution was successful")
	assertMissingAttribute(t, toolSpan.Attributes, "gen_ai.tool.call.result")

	// redpanda.tool.result.available should be false
	for _, attr := range toolSpan.Attributes {
		if string(attr.Key) == "redpanda.tool.result.available" {
			assert.False(t, attr.Value.AsBool())
		}
	}
}

func TestTracingInterceptor_SpanHierarchy(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Model call
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "test-model", provider: "test"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		_, _ = handler.Generate(ctx, &llm.Request{})

		// Tool calls
		req1 := &llm.ToolRequest{Name: "tool1", ID: "t1"}
		req2 := &llm.ToolRequest{Name: "tool2", ID: "t2"}

		toolInfo1 := &agent.ToolCallInfo{Inv: inv, Req: req1}
		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo1,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{}, nil
			})

		toolInfo2 := &agent.ToolCallInfo{Inv: inv, Req: req2}
		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo2,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{}, nil
			})

		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()

	// Count span types
	var invocationSpan *tracetest.SpanStub
	var chatSpans, toolSpans []*tracetest.SpanStub

	for i := range spans {
		// Check if it's an invocation span by looking at the name prefix
		//nolint:gocritic // if-else chain is clearer than switch for prefix matching
		if strings.HasPrefix(spans[i].Name, "invoke_agent") {
			invocationSpan = &spans[i]
		} else if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpans = append(chatSpans, &spans[i])
		} else if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpans = append(toolSpans, &spans[i])
		}
	}

	require.NotNil(t, invocationSpan, "Missing invocation span")
	require.Len(t, chatSpans, 1, "Expected 1 chat span")
	require.Len(t, toolSpans, 2, "Expected 2 tool spans")

	// Verify hierarchy: chat and tools are direct children of invocation (no turn span)
	assert.Equal(t, invocationSpan.SpanContext.SpanID(), chatSpans[0].Parent.SpanID(),
		"Chat span should be child of invocation span")
	assert.Equal(t, invocationSpan.SpanContext.SpanID(), toolSpans[0].Parent.SpanID(),
		"Tool span should be child of invocation span")
	assert.Equal(t, invocationSpan.SpanContext.SpanID(), toolSpans[1].Parent.SpanID(),
		"Tool span should be child of invocation span")
}

func TestTracingInterceptor_ContentRecording(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordInputs(true),
		pluginotel.WithRecordOutputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Model call with messages
		req := &llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("Hello, what's the weather?")}},
			},
		}
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "test-model", provider: "test"},
			Req:                req,
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		_, _ = handler.Generate(ctx, req)

		return agent.FinishReasonStop, nil
	})

	// Find the chat span (name format: "chat {model_name}")
	spans := exporter.GetSpans()
	var chatSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, chatSpan, "Expected chat span")

	// Should have both input and output messages as span attributes
	var hasInputMessages, hasOutputMessages bool

	for _, attr := range chatSpan.Attributes {
		if string(attr.Key) == "gen_ai.input.messages" {
			hasInputMessages = true
			// Verify it's a JSON string
			assert.NotEmpty(t, attr.Value.AsString(), "gen_ai.input.messages should not be empty")
		}

		if string(attr.Key) == "gen_ai.output.messages" {
			hasOutputMessages = true
			// Verify it's a JSON string
			assert.NotEmpty(t, attr.Value.AsString(), "gen_ai.output.messages should not be empty")
		}
	}

	assert.True(t, hasInputMessages, "Expected gen_ai.input.messages attribute")
	assert.True(t, hasOutputMessages, "Expected gen_ai.output.messages attribute")
}

func TestTracingInterceptor_ContextPropagation(t *testing.T) {
	t.Parallel()

	// This test verifies that context values are properly propagated through the interceptor chain
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})

	// Create a context with a value to verify it's passed through
	type ctxKey string
	ctx := context.WithValue(t.Context(), ctxKey("test"), "value")

	var contextWasPropagated bool

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "test-model", provider: "test"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{
			generateFn: func(ctx context.Context, _ *llm.Request) (*llm.Response, error) {
				// Verify context value propagated to the innermost handler
				if ctx.Value(ctxKey("test")) == "value" {
					contextWasPropagated = true
				}

				return &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant},
					FinishReason: llm.FinishReasonStop,
				}, nil
			},
		})
		_, _ = handler.Generate(ctx, &llm.Request{})

		return agent.FinishReasonStop, nil
	})

	// Verify that the context value was propagated through the interceptor chain
	assert.True(t, contextWasPropagated, "Context value should propagate through interceptor chain")

	// Verify spans were still created
	spans := exporter.GetSpans()
	assert.NotEmpty(t, spans)
}

func TestTracingInterceptor_InterceptToolExecution_WithToolTypeAndDescription(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "get_weather",
			ID:        "tool-call-123",
			Arguments: json.RawMessage(`{"city": "Seattle"}`),
		}

		// Create a tool definition with type and description
		toolDef := &llm.ToolDefinition{
			Name:        "get_weather",
			Description: "Gets the current weather for a location",
			Type:        llm.ToolTypeFunction,
		}

		toolInfo := &agent.ToolCallInfo{
			Inv:        inv,
			Req:        req,
			Definition: toolDef,
		}

		resp, err := interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`"Sunny, 72F"`)}, nil
			})
		require.NoError(t, err)
		assert.NotNil(t, resp)

		return agent.FinishReasonStop, nil
	})

	// Find the tool span
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Check that tool type and description attributes are set
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.type", "function")
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.description", "Gets the current weather for a location")
}

func TestTracingInterceptor_InterceptToolExecution_ToolTypeDefaultsToFunction(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "custom_tool",
			ID:        "tool-call-456",
			Arguments: json.RawMessage(`{}`),
		}

		// Create a tool definition without specifying Type (should default to "function")
		toolDef := &llm.ToolDefinition{
			Name:        "custom_tool",
			Description: "A custom tool",
		}

		toolInfo := &agent.ToolCallInfo{
			Inv:        inv,
			Req:        req,
			Definition: toolDef,
		}

		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`{}`)}, nil
			})

		return agent.FinishReasonStop, nil
	})

	// Find the tool span
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Check that tool type defaults to "function"
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.type", "function")
}

func TestTracingInterceptor_InterceptToolExecution_WithDifferentToolTypes(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		toolType string
	}{
		{name: "function type", toolType: llm.ToolTypeFunction},
		{name: "extension type", toolType: llm.ToolTypeExtension},
		{name: "datastore type", toolType: llm.ToolTypeDatastore},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			exporter, tp := setupTracer()
			defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

			interceptor := pluginotel.New(
				pluginotel.WithTracerProvider(tp),
			)

			inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
				Name:        "test-agent",
				Description: "Test agent for OpenTelemetry tracing",
			})
			ctx := t.Context()

			_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
				req := &llm.ToolRequest{
					Name:      "test_tool",
					ID:        "tool-call-789",
					Arguments: json.RawMessage(`{}`),
				}

				toolDef := &llm.ToolDefinition{
					Name:        "test_tool",
					Description: "A test tool",
					Type:        tc.toolType,
				}

				toolInfo := &agent.ToolCallInfo{
					Inv:        inv,
					Req:        req,
					Definition: toolDef,
				}

				_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
					func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
						return &llm.ToolResponse{Result: json.RawMessage(`{}`)}, nil
					})

				return agent.FinishReasonStop, nil
			})

			// Find the tool span
			spans := exporter.GetSpans()
			var toolSpan *tracetest.SpanStub

			for i := range spans {
				if strings.HasPrefix(spans[i].Name, "execute_tool") {
					toolSpan = &spans[i]
					break
				}
			}

			require.NotNil(t, toolSpan, "Expected execute_tool span")

			// Check that the correct tool type is set
			assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.type", tc.toolType)
		})
	}
}

func TestTracingInterceptor_InterceptToolExecution_WithoutDefinition(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "unknown_tool",
			ID:        "tool-call-999",
			Arguments: json.RawMessage(`{}`),
		}

		// ToolCallInfo without Definition (simulates tool not found in registry)
		toolInfo := &agent.ToolCallInfo{
			Inv:        inv,
			Req:        req,
			Definition: nil, // No definition available
		}

		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`{}`)}, nil
			})

		return agent.FinishReasonStop, nil
	})

	// Find the tool span
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Should still have basic attributes
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.name", "unknown_tool")
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.call.id", "tool-call-999")

	// Should NOT have type or description attributes when Definition is nil
	assertMissingAttribute(t, toolSpan.Attributes, "gen_ai.tool.type")
	assertMissingAttribute(t, toolSpan.Attributes, "gen_ai.tool.description")
}

func TestTracingInterceptor_InterceptToolExecution_InvalidToolTypeDefaultsToFunction(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		req := &llm.ToolRequest{
			Name:      "invalid_type_tool",
			ID:        "tool-call-invalid",
			Arguments: json.RawMessage(`{}`),
		}

		// Create a tool definition with an invalid Type value
		toolDef := &llm.ToolDefinition{
			Name:        "invalid_type_tool",
			Description: "Tool with invalid type",
			Type:        "invalid_type_value", // Invalid - should default to "function"
		}

		toolInfo := &agent.ToolCallInfo{
			Inv:        inv,
			Req:        req,
			Definition: toolDef,
		}

		_, _ = interceptor.InterceptToolExecution(ctx, toolInfo,
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponse, error) {
				return &llm.ToolResponse{Result: json.RawMessage(`{}`)}, nil
			})

		return agent.FinishReasonStop, nil
	})

	// Find the tool span
	spans := exporter.GetSpans()
	var toolSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "execute_tool") {
			toolSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, toolSpan, "Expected execute_tool span")

	// Check that invalid tool type defaults to "function" for OTel compliance
	assertHasAttribute(t, toolSpan.Attributes, "gen_ai.tool.type", "function")
}

func TestTracingInterceptor_AgentIDAndVersion_Present(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent",
		ID:          "agent-123",
		Version:     "1.0",
	})
	ctx := t.Context()

	reason, err := interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	require.NoError(t, err)
	assert.Equal(t, agent.FinishReasonStop, reason)

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	assertHasAttribute(t, spans[0].Attributes, "gen_ai.agent.id", "agent-123")
	assertHasAttribute(t, spans[0].Attributes, "gen_ai.agent.version", "1.0")
}

func TestTracingInterceptor_AgentIDAndVersion_Absent(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:        "test-agent",
		Description: "Test agent",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	assertMissingAttribute(t, spans[0].Attributes, "gen_ai.agent.id")
	assertMissingAttribute(t, spans[0].Attributes, "gen_ai.agent.version")
}

func TestTracingInterceptor_CacheReadTokens_OnModelSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name: "test-agent",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{
			generateFn: func(_ context.Context, _ *llm.Request) (*llm.Response, error) {
				return &llm.Response{
					Message:      llm.Message{Role: llm.RoleAssistant},
					FinishReason: llm.FinishReasonStop,
					Usage: &llm.TokenUsage{
						InputTokens:  100,
						OutputTokens: 50,
						CachedTokens: 75,
					},
					ID: "resp-cache",
				}, nil
			},
		})
		_, _ = handler.Generate(ctx, &llm.Request{})

		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	var chatSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, chatSpan)
	assertHasAttribute(t, chatSpan.Attributes, "gen_ai.usage.cache_read.input_tokens", int64(75))
}

func TestTracingInterceptor_CacheReadTokens_AbsentWhenZero(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name: "test-agent",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		_, _ = handler.Generate(ctx, &llm.Request{})

		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	var chatSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "chat") {
			chatSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, chatSpan)
	assertMissingAttribute(t, chatSpan.Attributes, "gen_ai.usage.cache_read.input_tokens")
}

func TestTracingInterceptor_CacheReadTokens_OnInvocationSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name: "test-agent",
	})
	ctx := t.Context()

	// Simulate a model call that adds cached tokens to the invocation usage
	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Manually add usage with cached tokens to the invocation
		agent.AddUsage(inv, &llm.TokenUsage{
			InputTokens:  100,
			OutputTokens: 50,
			CachedTokens: 30,
		})

		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	var invSpan *tracetest.SpanStub

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "invoke_agent") {
			invSpan = &spans[i]
			break
		}
	}

	require.NotNil(t, invSpan)
	assertHasAttribute(t, invSpan.Attributes, "gen_ai.usage.cache_read.input_tokens", int64(30))
}

func TestTracingInterceptor_SystemInstructions_Emitted(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:         "test-agent",
		SystemPrompt: "You are a helpful assistant.",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	// Verify gen_ai.system_instructions is present as JSON array
	var found bool

	for _, attr := range spans[0].Attributes {
		if string(attr.Key) == "gen_ai.system_instructions" {
			found = true
			val := attr.Value.AsString()
			assert.Contains(t, val, "You are a helpful assistant.")
		}
	}

	assert.True(t, found, "Expected gen_ai.system_instructions attribute")
}

func TestTracingInterceptor_ModelAndProvider_OnInvocationSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name:         "test-agent",
		ModelName:    "gpt-4o",
		ProviderName: "openai",
	})
	ctx := t.Context()

	// Complete immediately without any model call
	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	// Even without a model call, invocation span should have model/provider from Info
	assertHasAttribute(t, spans[0].Attributes, "gen_ai.request.model", "gpt-4o")
	assertHasAttribute(t, spans[0].Attributes, "gen_ai.provider.name", "openai")
}

func TestTracingInterceptor_ModelAndProvider_AbsentForNonLLMAgent(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	// Non-LLM agent: no ModelName or ProviderName
	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Info{
		Name: "orchestrator",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(_ context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	require.Len(t, spans, 1)

	assertMissingAttribute(t, spans[0].Attributes, "gen_ai.request.model")
	assertMissingAttribute(t, spans[0].Attributes, "gen_ai.provider.name")
}

// Helper functions for attribute assertions

func assertMissingAttribute(t *testing.T, attrs []attribute.KeyValue, key string) {
	t.Helper()

	for _, attr := range attrs {
		if string(attr.Key) == key {
			t.Errorf("Attribute %s should not be present, but found with value %v", key, attr.Value)
			return
		}
	}
}

func assertHasAttribute(t *testing.T, attrs []attribute.KeyValue, key string, expected any) {
	t.Helper()

	for _, attr := range attrs {
		if string(attr.Key) == key {
			switch v := expected.(type) {
			case string:
				assert.Equal(t, v, attr.Value.AsString(), "Attribute %s", key)
			case int64:
				assert.Equal(t, v, attr.Value.AsInt64(), "Attribute %s", key)
			case int:
				assert.Equal(t, int64(v), attr.Value.AsInt64(), "Attribute %s", key)
			default:
				t.Fatalf("Unsupported expected type for attribute %s: %T", key, expected)
			}

			return
		}
	}

	t.Errorf("Attribute %s not found", key)
}

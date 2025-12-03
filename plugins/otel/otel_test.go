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

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptTurn_CreatesInvocationSpan(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
	assertHasAttribute(t, invocationSpan.Attributes, pluginotel.AttrGenAIOperationName, pluginotel.OperationInvokeAgent)
	assertHasAttribute(t, invocationSpan.Attributes, pluginotel.AttrGenAIConversationID, "sess-123")
	assertHasAttribute(t, invocationSpan.Attributes, pluginotel.AttrGenAIAgentName, "test-agent")
}

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptTurn_MultipleTurns(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
	assertHasAttribute(t, invocationSpan.Attributes, pluginotel.AttrGenAIOperationName, pluginotel.OperationInvokeAgent)
}

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptTurn_ErrorRecording(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptModel_Generate(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	// First create a turn to establish parent span
	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Within turn, intercept model call
		modelInfo := &agent.ModelCallInfo{
			Inv:   inv,
			Model: &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:   &llm.Request{},
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
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIOperationName, pluginotel.OperationChat)
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIRequestModel, "gpt-4")
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIProviderName, "openai")
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIResponseID, "resp-123")
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIUsageInputTokens, int64(10))
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIUsageOutputTokens, int64(20))

	// Verify SpanKind is Client for model calls
	assert.Equal(t, trace.SpanKindClient, chatSpan.SpanKind)
}

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptModel_GenerateEvents(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			Inv:   inv,
			Model: &mockModelInfo{name: "test-model", provider: "test"},
			Req:   &llm.Request{},
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
	assertHasAttribute(t, chatSpan.Attributes, pluginotel.AttrGenAIResponseID, "resp-456")
}

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptToolExecution(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
	assertHasAttribute(t, toolSpan.Attributes, pluginotel.AttrGenAIOperationName, pluginotel.OperationToolCall)
	assertHasAttribute(t, toolSpan.Attributes, pluginotel.AttrGenAIToolName, "get_weather")
	assertHasAttribute(t, toolSpan.Attributes, pluginotel.AttrGenAIToolCallID, "tool-call-123")

	// Verify SpanKind is Internal for tool calls
	assert.Equal(t, trace.SpanKindInternal, toolSpan.SpanKind)
}

//nolint:dupl,paralleltest // Similar to WithRecordOutputs test but tests different functionality; Uses shared tracer state
func TestTracingInterceptor_InterceptToolExecution_WithRecordInputs(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordInputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
	assertHasAttribute(t, toolSpan.Attributes, pluginotel.AttrGenAIToolCallArguments, `{"city": "Seattle"}`)
}

//nolint:dupl,paralleltest // Similar to WithRecordInputs test but tests different functionality; Uses shared tracer state
func TestTracingInterceptor_InterceptToolExecution_WithRecordOutputs(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordOutputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
	assertHasAttribute(t, toolSpan.Attributes, pluginotel.AttrGenAIToolCallResult, `{"temperature":"72F","conditions":"sunny"}`)
}

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_InterceptToolExecution_Error(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_SpanHierarchy(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})
	ctx := t.Context()

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		// Model call
		modelInfo := &agent.ModelCallInfo{
			Inv:   inv,
			Model: &mockModelInfo{name: "test-model", provider: "test"},
			Req:   &llm.Request{},
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

//nolint:paralleltest // Uses shared tracer state
func TestTracingInterceptor_ContentRecording(t *testing.T) {
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithRecordInputs(true),
		pluginotel.WithRecordOutputs(true),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
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
			Inv:   inv,
			Model: &mockModelInfo{name: "test-model", provider: "test"},
			Req:   req,
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

	// Should have both prompt and completion events
	var hasPromptEvent, hasCompletionEvent bool

	for _, evt := range chatSpan.Events {
		if evt.Name == pluginotel.EventGenAIContentPrompt {
			hasPromptEvent = true
		}

		if evt.Name == pluginotel.EventGenAIContentCompletion {
			hasCompletionEvent = true
		}
	}

	assert.True(t, hasPromptEvent, "Expected gen_ai.content.prompt event")
	assert.True(t, hasCompletionEvent, "Expected gen_ai.content.completion event")
}

func TestTracingInterceptor_ContextPropagation(t *testing.T) {
	t.Parallel()

	// This test verifies that context values are properly propagated through the interceptor chain
	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
	)

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-123"}, agent.Snapshot{
		Name:        "test-agent",
		Description: "Test agent for OpenTelemetry tracing",
	})

	// Create a context with a value to verify it's passed through
	type ctxKey string
	ctx := context.WithValue(t.Context(), ctxKey("test"), "value")

	var contextWasPropagated bool

	_, _ = interceptor.InterceptTurn(ctx, &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			Inv:   inv,
			Model: &mockModelInfo{name: "test-model", provider: "test"},
			Req:   &llm.Request{},
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

// Helper functions for attribute assertions

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

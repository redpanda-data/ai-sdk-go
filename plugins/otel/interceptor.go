package otel

import (
	"context"
	"encoding/json"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
)

// TracingInterceptor provides OpenTelemetry tracing for agent operations.
//
// It implements [agent.TurnInterceptor], [agent.ModelInterceptor], and [agent.ToolInterceptor]
// to create a span hierarchy following OTel Gen AI semantic conventions:
//
//	invoke_agent my-assistant
//	  - chat gpt-4o (model call)
//	  - execute_tool get_weather
//	  - execute_tool search_web
//	  - chat gpt-4o (model call)
//
// # Usage
//
//	tracer := otel.New(
//	    otel.WithTracerProvider(tp),
//	    otel.WithRecordToolDefinitions(true), // opt-in for tool definitions (disabled by default per spec)
//	)
//
//	agent, _ := llmagent.New(
//	    "assistant",
//	    "You are helpful",
//	    model,
//	    llmagent.WithInterceptors(tracer),
//	)
//
// # TracerProvider Configuration
//
// By default, the interceptor uses the global TracerProvider from otel.GetTracerProvider().
// You can provide a custom TracerProvider with WithTracerProvider:
//
//	tp := sdktrace.NewTracerProvider(...)
//	tracer := otel.New(otel.WithTracerProvider(tp))
//
// # Content Recording
//
// By default, prompt/completion content and tool definitions are NOT recorded to avoid
// capturing PII and to minimize span size per OTel Gen AI semantic conventions.
//
// Enable selectively with:
//   - WithRecordInputs(true) - record model prompts as gen_ai.input.messages span attribute (JSON string)
//     and tool arguments as gen_ai.tool.call.arguments span attributes
//   - WithRecordOutputs(true) - record model completions as gen_ai.output.messages span attribute (JSON string)
//     and tool results as gen_ai.tool.call.result span attributes
//   - WithRecordToolDefinitions(true) - record tool definitions as gen_ai.tool.definitions attribute
//
// Note: Tool definitions are "NOT RECOMMENDED to populate by default" per the OTel spec due to size.
type TracingInterceptor struct {
	tracer trace.Tracer
	cfg    config
}

// Compile-time interface checks.
var (
	_ agent.TurnInterceptor  = (*TracingInterceptor)(nil)
	_ agent.ModelInterceptor = (*TracingInterceptor)(nil)
	_ agent.ToolInterceptor  = (*TracingInterceptor)(nil)
)

// New creates a TracingInterceptor with the given options.
//
// If no TracerProvider is specified, the global provider from otel.GetTracerProvider() is used.
func New(opts ...Option) *TracingInterceptor {
	cfg := defaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	// Get tracer from provider
	tp := cfg.tracerProvider
	if tp == nil {
		tp = otel.GetTracerProvider()
	}

	return &TracingInterceptor{
		tracer: tp.Tracer(cfg.tracerName),
		cfg:    cfg,
	}
}

// InterceptTurn creates a span for the agent invocation.
//
// On turn 0, it creates the root "gen_ai.agent" span (invoke_agent) that covers the entire invocation.
// Model and tool spans are created as direct children of the invocation span.
func (t *TracingInterceptor) InterceptTurn(
	ctx context.Context,
	info *agent.TurnInfo,
	next agent.TurnNext,
) (agent.FinishReason, error) {
	inv := info.Inv

	// Ensure we have the invocation span in the context when we call next
	ctx = t.withInvocationSpan(ctx, inv)

	// Execute the turn with invocation context so model/tool spans are children of invocation span
	reason, err := next(ctx, info)

	// End invocation span on terminal conditions
	if reason != "" || err != nil {
		t.endInvocationSpan(inv, err)
	}

	return reason, err
}

// withInvocationSpan ensures the invocation span exists and is in the context.
// On turn 0, it creates the invocation span. On subsequent turns, it re-parents the context.
func (t *TracingInterceptor) withInvocationSpan(
	ctx context.Context,
	inv *agent.InvocationMetadata,
) context.Context {
	if inv.Turn() == 0 {
		ctx, span := t.startInvocationSpan(ctx, inv)
		// Store only the span (not context) for later retrieval
		inv.SetMetadata(metadataKeyInvocationSpan, span)

		return ctx
	}

	if span, ok := getInvocationSpan(inv); ok {
		// Re-parent context to the existing invocation span while
		// preserving deadlines/cancellation
		return trace.ContextWithSpan(ctx, span)
	}

	return ctx
}

// startInvocationSpan creates the root invocation span with all required attributes.
func (t *TracingInterceptor) startInvocationSpan(
	ctx context.Context,
	inv *agent.InvocationMetadata,
) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{
		genAIOperationName(operationInvokeAgent),
	}

	session := inv.Session()
	if session != nil {
		if session.ID != "" {
			attrs = append(attrs, genAIConversationID(session.ID))
		}
	}

	agentSnap := inv.Agent()

	// Add system instructions from agent snapshot (not from session messages)
	// Per OTel spec, system_instructions should be for separately-provided instructions
	if agentSnap.SystemPrompt != "" {
		// Transform to OTel format (array of parts)
		systemPart := otelPart{
			Type:    "text",
			Content: agentSnap.SystemPrompt,
		}
		if sysJSON, err := json.Marshal([]otelPart{systemPart}); err == nil {
			attrs = append(attrs, genAISystemInstructions(string(sysJSON)))
		}
	}

	if agentSnap.Name != "" {
		attrs = append(attrs, genAIAgentName(agentSnap.Name))
	}

	if agentSnap.Description != "" {
		attrs = append(attrs, genAIAgentDescription(agentSnap.Description))
	}

	// Build span name following OTel convention: "invoke_agent {gen_ai.agent.name}"
	spanName := "invoke_agent"
	if agentSnap.Name != "" {
		spanName = "invoke_agent " + agentSnap.Name
	}

	// Call attribute injector if configured (before span creation for sampling)
	if t.cfg.attributeInjector != nil {
		var sessionID string
		if session != nil {
			sessionID = session.ID
		}

		spanCtx := SpanContext{
			SpanType:  SpanTypeInvocation,
			SpanName:  spanName,
			SessionID: sessionID,
			Inv:       inv,
		}

		if customAttrs := t.cfg.attributeInjector(ctx, spanCtx); len(customAttrs) > 0 {
			attrs = append(attrs, customAttrs...)
		}
	}

	//nolint:spancheck // span is returned and stored in metadata by caller
	return t.tracer.Start(
		ctx,
		spanName,
		trace.WithSpanKind(trace.SpanKindInternal),
		trace.WithAttributes(attrs...),
	)
}

// getInvocationSpan retrieves the invocation span from metadata.
func getInvocationSpan(inv *agent.InvocationMetadata) (trace.Span, bool) {
	span, ok := inv.GetMetadata(metadataKeyInvocationSpan).(trace.Span)
	return span, ok
}

// endInvocationSpan finalizes the invocation span with usage stats and optional error.
func (t *TracingInterceptor) endInvocationSpan(inv *agent.InvocationMetadata, err error) {
	span, ok := getInvocationSpan(inv)
	if !ok {
		return
	}

	// Add final usage stats to invocation span
	usage := inv.TotalUsage()
	span.SetAttributes(
		genAIUsageInputTokens(usage.InputTokens),
		genAIUsageOutputTokens(usage.OutputTokens),
	)

	setSpanError(span, err)
	span.End()
}

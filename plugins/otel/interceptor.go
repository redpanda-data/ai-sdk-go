package otel

import (
	"context"

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
//   - WithRecordInputs(true) - record model prompts as gen_ai.content.prompt events and
//     tool arguments as gen_ai.tool.call.arguments span attributes
//   - WithRecordOutputs(true) - record model completions as gen_ai.content.completion events and
//     tool results as gen_ai.tool.call.result span attributes
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

	// On turn 0, create the root invocation span
	//nolint:nestif // Complex span initialization logic
	if inv.Turn() == 0 {
		var invSpan trace.Span

		// Build attributes for invocation span
		attrs := []attribute.KeyValue{
			GenAIOperationName(OperationInvokeAgent),
		}

		// Add conversation ID if session is available
		if session := inv.Session(); session != nil && session.ID != "" {
			attrs = append(attrs, GenAIConversationID(session.ID))
		}

		// Add agent metadata from snapshot
		agentSnap := inv.Agent()
		if agentSnap.Name != "" {
			attrs = append(attrs, GenAIAgentName(agentSnap.Name))
		}

		if agentSnap.Description != "" {
			attrs = append(attrs, GenAIAgentDescription(agentSnap.Description))
		}

		// Extract system prompt from session messages (first system role message)
		if systemPrompt := extractSystemPrompt(inv.Session().Messages); systemPrompt != "" {
			attrs = append(attrs, GenAISystemInstructions(systemPrompt))
		}

		// Build span name following OTel convention: "invoke_agent {gen_ai.agent.name}"
		// Falls back to just "invoke_agent" if agent name is not available
		spanName := "invoke_agent"
		if agentSnap.Name != "" {
			spanName = "invoke_agent " + agentSnap.Name
		}

		// Call attribute injector if configured (before span creation for sampling)
		if t.cfg.attributeInjector != nil {
			sessionID := ""
			if session := inv.Session(); session != nil {
				sessionID = session.ID
			}

			injectorCtx := AttributeContext{
				Ctx:       ctx,
				SpanType:  SpanTypeInvocation,
				SpanName:  spanName,
				SessionID: sessionID,
				Inv:       inv,
			}
			if customAttrs := t.cfg.attributeInjector(injectorCtx); len(customAttrs) > 0 {
				attrs = append(attrs, customAttrs...)
			}
		}

		ctx, invSpan = t.tracer.Start(ctx, spanName,
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(attrs...),
		)

		// Store only the span (not context) for later retrieval
		inv.SetMetadata(MetadataKeyInvocationSpan, invSpan)
	} else {
		// Retrieve the invocation span and re-parent the context
		// This preserves the incoming context's deadlines/cancellation while
		// maintaining the span hierarchy
		if invSpan, ok := inv.GetMetadata(MetadataKeyInvocationSpan).(trace.Span); ok {
			ctx = trace.ContextWithSpan(ctx, invSpan)
		}
	}

	// Execute the turn with invocation context so model/tool spans are children of invocation span
	reason, err := next(ctx, info)

	// End invocation span on terminal conditions
	if reason != "" || err != nil {
		t.endInvocationSpan(inv, err)
	}

	return reason, err
}

// endInvocationSpan finalizes the invocation span with usage stats and optional error.
func (t *TracingInterceptor) endInvocationSpan(inv *agent.InvocationMetadata, err error) {
	invSpan, ok := inv.GetMetadata(MetadataKeyInvocationSpan).(trace.Span)
	if !ok {
		return
	}

	// Add final usage stats to invocation span
	usage := inv.TotalUsage()
	invSpan.SetAttributes(
		GenAIUsageInputTokens(usage.InputTokens),
		GenAIUsageOutputTokens(usage.OutputTokens),
	)

	setSpanError(invSpan, err)
	invSpan.End()
}

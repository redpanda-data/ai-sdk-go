package otel

import (
	"context"
	"encoding/json"
	"iter"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// InterceptModel creates a "gen_ai.chat" span wrapping model generation calls.
func (t *TracingInterceptor) InterceptModel(
	_ context.Context,
	info *agent.ModelCallInfo,
	next agent.ModelCallHandler,
) agent.ModelCallHandler {
	// Pass the context through - it already has the invocation span as parent
	// from InterceptTurn calling next(ctx, info)
	convID := ""
	if session := info.InvocationMetadata.Session(); session != nil {
		convID = session.ID
	}

	return &tracingModelHandler{
		tracer:    t.tracer,
		cfg:       t.cfg,
		next:      next,
		modelInfo: info.Model, // Capture model info for span attributes
		convID:    convID,
		inv:       info.InvocationMetadata, // Capture invocation metadata for attribute injection
	}
}

// tracingModelHandler wraps model generation with tracing.
type tracingModelHandler struct {
	tracer    trace.Tracer
	cfg       config
	next      agent.ModelCallHandler
	modelInfo llm.ModelInfo             // Model info for span attributes (name, capabilities)
	convID    string                    // Conversation ID for correlation across spans
	inv       *agent.InvocationMetadata // Invocation metadata for attribute injection
}

// Generate wraps synchronous model generation with tracing.
func (h *tracingModelHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	ctx, span := h.startSpan(ctx, req)
	defer span.End()

	// Execute generation
	resp, err := h.next.Generate(ctx, req)
	if err != nil {
		setSpanError(span, err)
		return nil, err
	}

	// Record response attributes and output messages
	h.recordResponseAttributes(span, resp)

	if h.cfg.recordOutputs {
		h.recordOutputMessages(span, resp)
	}

	return resp, nil
}

// GenerateEvents wraps streaming model generation with tracing.
//
// The span is created when iteration begins (not when the iterator is obtained) and automatically
// ended via defer, ensuring proper cleanup even if consumers break from the loop early. This also
// captures the actual stream duration. Final response attributes (usage, finish reason) are recorded
// when available via StreamEndEvent.
func (h *tracingModelHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		// Start span when iteration begins (fixes leak if iterator never consumed)
		ctx, span := h.startSpan(ctx, req)
		defer span.End()

		for evt, err := range h.next.GenerateEvents(ctx, req) {
			if err != nil {
				setSpanError(span, err)
				yield(nil, err)

				return
			}

			// Check for StreamEndEvent to capture final stats
			if endEvt, ok := evt.(llm.StreamEndEvent); ok {
				setSpanError(span, endEvt.Error)
				h.recordResponseAttributes(span, endEvt.Response)

				if h.cfg.recordOutputs && endEvt.Response != nil {
					h.recordOutputMessages(span, endEvt.Response)
				}

				yield(evt, nil)

				return
			}

			if !yield(evt, nil) {
				return
			}
		}
	}
}

// startSpan initializes a chat span with common attributes and optional input recording.
func (h *tracingModelHandler) startSpan(ctx context.Context, req *llm.Request) (context.Context, trace.Span) {
	// Build span name following OTel convention: "chat {gen_ai.request.model}"
	// Falls back to just "chat" if model name is not available
	spanName := "chat"

	if h.modelInfo != nil {
		if modelName := h.modelInfo.Name(); modelName != "" {
			spanName = "chat " + modelName
		}
	}

	// Build base attributes
	attrs := []attribute.KeyValue{
		genAIOperationName(operationChat),
		genAIConversationID(h.convID),
	}

	// Call attribute injector if configured (before span creation for sampling)
	if h.cfg.attributeInjector != nil {
		spanCtx := SpanContext{
			SpanType:  SpanTypeModel,
			SpanName:  spanName,
			SessionID: h.convID,
			Inv:       h.inv,
		}
		if customAttrs := h.cfg.attributeInjector(ctx, spanCtx); len(customAttrs) > 0 {
			attrs = append(attrs, customAttrs...)
		}
	}

	ctx, span := h.tracer.Start(ctx, spanName,
		trace.WithSpanKind(trace.SpanKindClient), // Model calls are outbound client calls
		trace.WithAttributes(attrs...),
	)

	h.addRequestAttributes(span, req)

	if h.cfg.recordInputs {
		h.recordInputMessages(span, req)
	}

	return ctx, span
}

// addRequestAttributes adds request-related attributes to the span.
// Also propagates provider and model information to the parent invocation span.
func (h *tracingModelHandler) addRequestAttributes(span trace.Span, req *llm.Request) {
	// Get model name and provider from ModelInfo
	var modelName, providerName string

	if h.modelInfo != nil {
		if name := h.modelInfo.Name(); name != "" {
			modelName = name
			span.SetAttributes(genAIRequestModel(name))
		}

		if prov := h.modelInfo.Provider(); prov != "" {
			providerName = prov
			span.SetAttributes(genAIProviderName(prov))
		}
	}

	// Propagate provider and model to the invocation span if not already set at creation.
	// For LLM agents, startInvocationSpan sets these from Info().ModelName/ProviderName.
	// This fallback covers non-LLM agents or agents where Info() has empty model/provider.
	h.propagateModelToInvocation(providerName, modelName)

	// Optionally record available tools from the request (gen_ai.tool.definitions)
	// Disabled by default per OTel spec: "NOT RECOMMENDED to populate by default" due to size
	if h.cfg.recordToolDefinitions && req != nil && len(req.Tools) > 0 {
		if toolsJSON, err := json.Marshal(req.Tools); err == nil {
			span.SetAttributes(genAIToolDefinitions(string(toolsJSON)))
		}
	}
}

// propagateModelToInvocation sets provider/model on the invocation span when
// they were not already set at span creation time (i.e. for non-LLM agents or
// agents whose Info() returns empty model/provider).
func (h *tracingModelHandler) propagateModelToInvocation(providerName, modelName string) {
	agentSnap := h.inv.Agent()
	// Short-circuit: both model and provider were already set on the invocation
	// span at creation time (startInvocationSpan reads them from Info()). The
	// per-attribute guards below handle the mixed case where only one is set.
	if agentSnap.ModelName != "" && agentSnap.ProviderName != "" {
		return
	}

	invSpan, ok := getInvocationSpan(h.inv)
	if !ok {
		return
	}

	var invAttrs []attribute.KeyValue
	if providerName != "" && agentSnap.ProviderName == "" {
		invAttrs = append(invAttrs, genAIProviderName(providerName))
	}

	if modelName != "" && agentSnap.ModelName == "" {
		invAttrs = append(invAttrs, genAIRequestModel(modelName))
	}

	if len(invAttrs) > 0 {
		invSpan.SetAttributes(invAttrs...)
	}
}

// recordInputMessages records the input messages as a span attribute (gen_ai.input.messages).
// Per OTel spec, this is recorded as a JSON string following the Input messages JSON schema.
func (h *tracingModelHandler) recordInputMessages(span trace.Span, req *llm.Request) {
	if req == nil || len(req.Messages) == 0 {
		return
	}

	// Transform SDK messages to OTel-compliant format
	otelMessages := transformInputMessages(req.Messages)

	// Serialize OTel messages to JSON and record as span attribute
	if messagesJSON, err := json.Marshal(otelMessages); err == nil {
		span.SetAttributes(
			genAIInputMessages(string(messagesJSON)),
		)
	}
}

// recordOutputMessages records the output message as a span attribute (gen_ai.output.messages).
// Per OTel spec, this is recorded as a JSON string following the Output messages JSON schema.
// Output messages are recorded as an array containing a single message with finish_reason.
func (h *tracingModelHandler) recordOutputMessages(span trace.Span, resp *llm.Response) {
	if resp == nil {
		return
	}

	// Transform SDK message to OTel-compliant format with finish_reason
	// Use mapFinishReason to normalize to OTel enum
	otelMsg := transformOutputMessage(resp.Message, mapFinishReason(resp.FinishReason))

	// Output messages must be an array per the schema
	otelMessages := []otelMessage{otelMsg}

	// Serialize OTel messages to JSON and record as span attribute
	if messageJSON, err := json.Marshal(otelMessages); err == nil {
		span.SetAttributes(
			genAIOutputMessages(string(messageJSON)),
		)
	}
}

// recordResponseAttributes adds response attributes to the span.
func (h *tracingModelHandler) recordResponseAttributes(span trace.Span, resp *llm.Response) {
	if resp == nil {
		return
	}

	if resp.ID != "" {
		span.SetAttributes(genAIResponseID(resp.ID))
	}

	// Note: llm.Response doesn't have a Model field. The model used can be
	// set via WithModelName() option if the provider doesn't return it.
	// If we need response model, we would need to extend llm.Response.

	if resp.FinishReason != "" {
		span.SetAttributes(genAIResponseFinishReasons(mapFinishReason(resp.FinishReason)))
	}

	setUsageAttributes(span, resp.Usage)
}

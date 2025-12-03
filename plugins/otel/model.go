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
	if session := info.Inv.Session(); session != nil {
		convID = session.ID
	}

	return &tracingModelHandler{
		tracer:    t.tracer,
		cfg:       t.cfg,
		next:      next,
		modelInfo: info.Model, // Capture model info for span attributes
		convID:    convID,
		inv:       info.Inv, // Capture invocation metadata for attribute injection
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

	// Record response attributes and output event
	h.recordResponseAttributes(span, resp)

	if h.cfg.recordOutputs {
		h.recordOutputEvent(span, resp)
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
					h.recordOutputEvent(span, endEvt.Response)
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
		GenAIOperationName(OperationChat),
		GenAIConversationID(h.convID),
	}

	// Call attribute injector if configured (before span creation for sampling)
	if h.cfg.attributeInjector != nil {
		injectorCtx := AttributeContext{
			Ctx:       ctx,
			SpanType:  SpanTypeModel,
			SpanName:  spanName,
			SessionID: h.convID,
			Inv:       h.inv,
		}
		if customAttrs := h.cfg.attributeInjector(injectorCtx); len(customAttrs) > 0 {
			attrs = append(attrs, customAttrs...)
		}
	}

	ctx, span := h.tracer.Start(ctx, spanName,
		trace.WithSpanKind(trace.SpanKindClient), // Model calls are outbound client calls
		trace.WithAttributes(attrs...),
	)

	h.addRequestAttributes(span, req)

	if h.cfg.recordInputs {
		h.recordInputEvent(span, req)
	}

	return ctx, span
}

// addRequestAttributes adds request-related attributes to the span.
func (h *tracingModelHandler) addRequestAttributes(span trace.Span, req *llm.Request) {
	// Get model name and provider from ModelInfo
	if h.modelInfo != nil {
		if modelName := h.modelInfo.Name(); modelName != "" {
			span.SetAttributes(GenAIRequestModel(modelName))
		}

		if providerName := h.modelInfo.Provider(); providerName != "" {
			span.SetAttributes(GenAIProviderName(providerName))
		}
	}

	// Optionally record available tools from the request (gen_ai.tool.definitions)
	// Disabled by default per OTel spec: "NOT RECOMMENDED to populate by default" due to size
	if h.cfg.recordToolDefinitions && req != nil && len(req.Tools) > 0 {
		if toolsJSON, err := json.Marshal(req.Tools); err == nil {
			span.SetAttributes(attribute.String("gen_ai.tool.definitions", string(toolsJSON)))
		}
	}
}

// recordInputEvent records the input messages as a span event.
func (h *tracingModelHandler) recordInputEvent(span trace.Span, req *llm.Request) {
	if req == nil || len(req.Messages) == 0 {
		return
	}

	// Serialize messages to JSON for the event
	if messagesJSON, err := json.Marshal(req.Messages); err == nil {
		span.AddEvent(EventGenAIContentPrompt,
			trace.WithAttributes(
				attribute.String("content", string(messagesJSON)),
				attribute.Int("message_count", len(req.Messages)),
			),
		)
	}
}

// recordOutputEvent records the output message as a span event.
func (h *tracingModelHandler) recordOutputEvent(span trace.Span, resp *llm.Response) {
	if resp == nil {
		return
	}

	// Serialize the response message to JSON for the event
	if messageJSON, err := json.Marshal(resp.Message); err == nil {
		span.AddEvent(EventGenAIContentCompletion,
			trace.WithAttributes(
				attribute.String("content", string(messageJSON)),
				attribute.String("finish_reason", string(resp.FinishReason)),
			),
		)
	}
}

// recordResponseAttributes adds response attributes to the span.
func (h *tracingModelHandler) recordResponseAttributes(span trace.Span, resp *llm.Response) {
	if resp == nil {
		return
	}

	if resp.ID != "" {
		span.SetAttributes(GenAIResponseID(resp.ID))
	}

	// Note: llm.Response doesn't have a Model field. The model used can be
	// set via WithModelName() option if the provider doesn't return it.
	// If we need response model, we would need to extend llm.Response.

	if resp.FinishReason != "" {
		span.SetAttributes(GenAIResponseFinishReasons(string(resp.FinishReason)))
	}

	if resp.Usage != nil {
		span.SetAttributes(
			GenAIUsageInputTokens(resp.Usage.InputTokens),
			GenAIUsageOutputTokens(resp.Usage.OutputTokens),
		)
	}
}

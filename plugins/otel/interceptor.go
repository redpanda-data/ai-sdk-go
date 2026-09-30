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

package otel

import (
	"context"
	"encoding/json"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/plugins/otel/genai"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TracingInterceptor provides OpenTelemetry tracing for agent operations.
//
// It implements [agent.InvocationInterceptor], [agent.ModelInterceptor], and [agent.ToolInterceptor]
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
//	    "support-triage",
//	    "You are helpful",
//	    model,
//	    llmagent.WithID("support-triage-prod"),
//	    llmagent.WithVersion("1.4.2"),
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
	_ agent.InvocationInterceptor = (*TracingInterceptor)(nil)
	_ agent.ModelInterceptor      = (*TracingInterceptor)(nil)
	_ agent.ToolInterceptor       = (*TracingInterceptor)(nil)
	_ agent.EventObserver         = (*TracingInterceptor)(nil)
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

// InterceptInvocation creates the root "gen_ai.agent" span (invoke_agent)
// covering the entire invocation. Model and tool spans are its direct children
// through the context passed to next.
func (t *TracingInterceptor) InterceptInvocation(
	ctx context.Context,
	info *agent.InvocationInfo,
	next agent.InvocationNext,
) (agent.FinishReason, error) {
	ctx, span := t.startInvocationSpan(ctx, info.Inv)
	// Model and compaction spans read it back to annotate the invocation span.
	info.Inv.SetMetadata(metadataKeyInvocationSpan, span)

	var (
		reason agent.FinishReason
		err    error
	)

	defer func() {
		if r := recover(); r != nil {
			recordPanic(span, r)
			endInvocationSpan(span, info.Inv, "", nil)
			panic(r) //nolint:forbidigo // re-raise after recording; the caller still sees the panic
		}

		endInvocationSpan(span, info.Inv, reason, err)
	}()

	reason, err = next(ctx, info)

	return reason, err
}

// ObserveEvent implements [agent.EventObserver].
func (t *TracingInterceptor) ObserveEvent(ctx context.Context, inv *agent.InvocationMetadata, event agent.Event) {
	if ce, ok := event.(agent.CompactionEvent); ok {
		t.recordCompaction(ctx, inv, ce)
	}
}

// startInvocationSpan creates the root invocation span with all required attributes.
func (t *TracingInterceptor) startInvocationSpan(
	ctx context.Context,
	inv *agent.InvocationMetadata,
) (context.Context, trace.Span) {
	attrs := []attribute.KeyValue{
		genAIOperationName(genai.OperationInvokeAgent),
	}

	sess := inv.Session()
	// Group under the conversation id (the parent/root for a sub-agent),
	// not necessarily this session's own (unique) storage id.
	if cid := session.ConversationID(sess); cid != "" {
		attrs = append(attrs, genAIConversationID(cid))
	}

	attrs = append(attrs, invocationAttributes(inv)...)

	agentSnap := inv.Agent()

	// Add system instructions from agent snapshot (not from session messages).
	// Per OTel spec, system_instructions carries separately-provided instruction
	// content and is Opt-In — recorded only when input recording is enabled.
	if t.cfg.recordInputs && agentSnap.SystemPrompt != "" {
		// Transform to OTel format (array of parts)
		systemPart := genai.Part{
			Type:    genai.PartTypeText,
			Content: agentSnap.SystemPrompt,
		}
		if sysJSON, err := json.Marshal([]genai.Part{systemPart}); err == nil {
			attrs = append(attrs, genAISystemInstructions(string(sysJSON)))
		}
	}

	if agentSnap.Name != "" {
		attrs = append(attrs, genAIAgentName(agentSnap.Name))
	}

	if agentSnap.Description != "" {
		attrs = append(attrs, genAIAgentDescription(agentSnap.Description))
	}

	if agentSnap.ID != "" {
		attrs = append(attrs, genAIAgentID(agentSnap.ID))
	}

	if agentSnap.Version != "" {
		attrs = append(attrs, genAIAgentVersion(agentSnap.Version))
	}

	if agentSnap.ModelName != "" {
		attrs = append(attrs, genAIRequestModel(agentSnap.ModelName))
	}

	if agentSnap.ProviderName != "" {
		attrs = append(attrs, genAIProviderName(agentSnap.ProviderName))
	}

	// Build span name following OTel convention: "invoke_agent {gen_ai.agent.name}"
	spanName := "invoke_agent"
	if agentSnap.Name != "" {
		spanName = "invoke_agent " + agentSnap.Name
	}

	// Call attribute injector if configured (before span creation for sampling)
	if t.cfg.attributeInjector != nil {
		spanCtx := SpanContext{
			SpanType:       SpanTypeInvocation,
			SpanName:       spanName,
			ConversationID: session.ConversationID(sess),
			Inv:            inv,
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

// endInvocationSpan finalizes the invocation span with usage stats, the finish
// reason and optional error.
func endInvocationSpan(span trace.Span, inv *agent.InvocationMetadata, reason agent.FinishReason, err error) {
	// Add final usage stats to invocation span
	usage := inv.TotalUsage()
	setUsageAttributes(span, &usage)

	if reason != "" {
		span.SetAttributes(genAIResponseFinishReasons(string(reason)))
	}

	if err != nil {
		setSpanError(span, err)
	} else if failedFinishReason(reason) {
		span.SetStatus(codes.Error, "agent stopped: "+string(reason))
		span.SetAttributes(errorType(string(reason)))
	}

	span.End()
}

// recordPanic marks the span failed with an exception event. It runs in the
// deferred recover, before the stack unwinds, so the recorded stack trace
// still points at the panic site.
func recordPanic(span trace.Span, r any) {
	err := fmt.Errorf("panic: %v", r)
	span.RecordError(err, trace.WithStackTrace(true))
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(errorType("panic"))
}

// failedFinishReason reports whether the agent stopped without completing its
// task. Callers still get a normal InvocationEndEvent, but the a2a executor
// and agenttool both surface these as failures, so the span must too.
func failedFinishReason(reason agent.FinishReason) bool {
	switch reason {
	case agent.FinishReasonMaxTurns, agent.FinishReasonContextOverflow, agent.FinishReasonError:
		return true
	case agent.FinishReasonStop, agent.FinishReasonLength, agent.FinishReasonInputRequired,
		agent.FinishReasonInterrupted, agent.FinishReasonTransfer:
		return false
	default:
		return false
	}
}

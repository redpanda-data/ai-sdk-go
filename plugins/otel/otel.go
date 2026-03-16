// Package otel provides OpenTelemetry tracing for AI agent operations.
//
// This plugin implements comprehensive tracing following the OpenTelemetry Gen AI
// semantic conventions, creating a structured span hierarchy for agent invocations,
// model calls, and tool executions.
//
// # Quick Start
//
//	import (
//	    "github.com/redpanda-data/ai-sdk-go/agent/llmagent"
//	    "github.com/redpanda-data/ai-sdk-go/plugins/otel"
//	)
//
//	tracer := otel.New()
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
// # Span Hierarchy
//
// The plugin creates a three-level span hierarchy:
//
//	invoke_agent my-assistant
//	  - chat gpt-4o (model call)
//	  - execute_tool get_weather
//	  - execute_tool search_web
//	  - chat gpt-4o (model call)
//
// # Security & Privacy
//
// By default, sensitive content (prompts, completions, tool arguments/results) is NOT recorded
// to prevent capturing PII and minimize span size. Enable selectively for debugging:
//
//	tracer := otel.New(
//	    otel.WithRecordInputs(true),   // Record model prompts and tool arguments
//	    otel.WithRecordOutputs(true),  // Record model completions and tool results
//	)
//
// # Custom TracerProvider
//
// Use a custom tracer provider for specific backends:
//
//	tp := sdktrace.NewTracerProvider(
//	    sdktrace.WithBatcher(exporter),
//	    sdktrace.WithResource(resource.NewWithAttributes(
//	        semconv.ServiceNameKey.String("my-ai-service"),
//	    )),
//	)
//
//	tracer := otel.New(otel.WithTracerProvider(tp))
//
// # Attribute Injection
//
// Add custom attributes for sampling, correlation, or backend-specific needs:
//
//	tracer := otel.New(
//	    otel.WithAttributeInjector(func(ctx context.Context, spanCtx otel.SpanContext) []attribute.KeyValue {
//	        return []attribute.KeyValue{
//	            attribute.String("environment", os.Getenv("ENV")),
//	            attribute.String("tenant_id", getTenantID(ctx)),
//	        }
//	    }),
//	)
package otel

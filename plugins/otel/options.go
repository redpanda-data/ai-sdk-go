package otel

import "go.opentelemetry.io/otel/trace"

const (
	// DefaultTracerName is the default instrumentation name for the tracer.
	DefaultTracerName = "github.com/redpanda-data/ai-sdk-go/plugins/otel"
)

// config holds the configuration for the TracingInterceptor.
type config struct {
	tracerProvider trace.TracerProvider
	tracerName     string

	// Content recording options (opt-in due to PII concerns)
	// When enabled, content is recorded as span events (gen_ai.content.prompt/completion)
	recordInputs  bool
	recordOutputs bool

	// Tool definitions recording (opt-in due to size concerns)
	// When enabled, tool definitions are recorded as gen_ai.tool.definitions attribute
	recordToolDefinitions bool

	// Custom attribute injection (optional)
	// Allows users to add platform-specific or custom attributes to spans
	attributeInjector AttributeInjector
}

// Option configures the TracingInterceptor.
type Option func(*config)

// defaultConfig returns the default configuration.
func defaultConfig() config {
	return config{
		tracerProvider:        nil, // Will use global provider
		tracerName:            DefaultTracerName,
		recordInputs:          false,
		recordOutputs:         false,
		recordToolDefinitions: false,
		attributeInjector:     nil, // No custom attributes by default
	}
}

// WithTracerProvider sets a custom TracerProvider.
// If not set, the global TracerProvider from otel.GetTracerProvider() is used.
func WithTracerProvider(tp trace.TracerProvider) Option {
	return func(c *config) {
		c.tracerProvider = tp
	}
}

// WithTracerName sets the instrumentation name for the tracer.
// Defaults to "github.com/redpanda-data/ai-sdk-go/plugins/otel".
func WithTracerName(name string) Option {
	return func(c *config) {
		c.tracerName = name
	}
}

// WithRecordInputs enables recording of input/prompt content as span events.
// Disabled by default due to potential PII in prompts.
//
// When enabled, prompts are recorded as gen_ai.content.prompt span events.
// Span events are preferred over attributes for large payloads per OTel GenAI conventions.
// Use with caution in production environments.
func WithRecordInputs(enabled bool) Option {
	return func(c *config) {
		c.recordInputs = enabled
	}
}

// WithRecordOutputs enables recording of output/completion content as span events.
// Disabled by default due to potential PII in responses.
//
// When enabled, completions are recorded as gen_ai.content.completion span events.
// Span events are preferred over attributes for large payloads per OTel GenAI conventions.
// Use with caution in production environments.
func WithRecordOutputs(enabled bool) Option {
	return func(c *config) {
		c.recordOutputs = enabled
	}
}

// WithRecordToolDefinitions enables recording of tool definitions as span attributes.
// Disabled by default due to potential size concerns per OTel GenAI semantic conventions.
//
// When enabled, tool definitions are recorded as gen_ai.tool.definitions attribute on model call spans.
// The OTel spec notes this is "NOT RECOMMENDED to populate by default" due to potentially large payloads.
// Only enable if you need this data for debugging or if your tools are small.
func WithRecordToolDefinitions(enabled bool) Option {
	return func(c *config) {
		c.recordToolDefinitions = enabled
	}
}

// WithAttributeInjector sets a custom attribute injector callback.
//
// The injector allows you to add platform-specific or custom attributes to spans
// based on runtime context. It is called before span creation for all span types
// (invocation, model, tool), ensuring attributes are available for sampling decisions.
//
// Common use cases:
//   - Adding platform-specific attributes (e.g., langfuse.trace.name for Langfuse)
//   - Injecting business context (customer ID, tenant ID, etc.)
//   - Adding environment-specific tags
//
// Example for Langfuse compatibility:
//
//	pluginotel.New(
//	    pluginotel.WithAttributeInjector(func(ctx pluginotel.AttributeContext) []attribute.KeyValue {
//	        if ctx.SpanType == pluginotel.SpanTypeInvocation {
//	            return []attribute.KeyValue{
//	                attribute.String("langfuse.trace.name", ctx.SpanName),
//	                attribute.String("langfuse.session.id", ctx.SessionID),
//	            }
//	        }
//	        return nil
//	    }),
//	)
//
// Important notes:
//   - The injector must be thread-safe (tools may execute concurrently)
//   - Return nil or empty slice if no attributes should be added
//   - Attributes are set before span creation (affects sampling)
//   - The injector receives full invocation context for dynamic values
func WithAttributeInjector(injector AttributeInjector) Option {
	return func(c *config) {
		c.attributeInjector = injector
	}
}

package agentpack

import (
	"context"
	"fmt"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"

	"github.com/redpanda-data/ai-sdk-go/agent"
	sdkotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
)

// otelSetup holds OTEL resources that need cleanup.
type otelSetup struct {
	interceptor agent.Interceptor
	shutdown    func(context.Context) error
}

// setupOTEL configures OpenTelemetry tracing if enabled.
//
// Env vars:
//   - OTEL_ENABLED: true to enable
//   - OTEL_SERVICE_NAME: service name (falls back to agent name)
//   - Standard OTEL env vars (OTEL_EXPORTER_OTLP_ENDPOINT, etc.)
func setupOTEL(ctx context.Context, agentName string) (*otelSetup, error) {
	if os.Getenv("OTEL_ENABLED") != "true" {
		return nil, nil
	}

	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = agentName
	}

	client := otlptracehttp.NewClient()
	exporter, err := otlptrace.New(ctx, client)
	if err != nil {
		return nil, fmt.Errorf("create OTLP exporter: %w", err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(semconv.ServiceName(serviceName)),
	)
	if err != nil {
		return nil, fmt.Errorf("create OTEL resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)

	interceptor := sdkotel.New(sdkotel.WithTracerProvider(tp))

	return &otelSetup{
		interceptor: interceptor,
		shutdown:    tp.Shutdown,
	}, nil
}

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

package genai

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"
)

func TestStampModelCallSpan(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		attrs       *ModelCallAttrs
		wantKVs     map[string]any
		wantMissing []string
	}{
		{
			name: "full attributes",
			attrs: &ModelCallAttrs{
				Provider:      "openai",
				RequestModel:  "gpt-4o",
				ResponseModel: "gpt-4o-2024-08-06",
				ResponseID:    "chatcmpl-abc123",
				FinishReason:  "stop",
				InputTokens:   100,
				OutputTokens:  50,
				CachedTokens:  25,
			},
			wantKVs: map[string]any{
				AttrGenAIOperationName:             OperationChat,
				AttrGenAIProviderName:              "openai",
				AttrGenAIRequestModel:              "gpt-4o",
				AttrGenAIResponseID:                "chatcmpl-abc123",
				AttrGenAIResponseFinishReasons:     []string{"stop"},
				AttrGenAIUsageInputTokens:          100,
				AttrGenAIUsageOutputTokens:         50,
				AttrGenAIUsageCacheReadInputTokens: 25,
			},
		},
		{
			name: "request model empty falls back to response model",
			attrs: &ModelCallAttrs{
				Provider:      "anthropic",
				ResponseModel: "claude-3-5-sonnet-20241022",
				InputTokens:   200,
				OutputTokens:  100,
			},
			wantKVs: map[string]any{
				AttrGenAIOperationName:     OperationChat,
				AttrGenAIProviderName:      "anthropic",
				AttrGenAIRequestModel:      "claude-3-5-sonnet-20241022",
				AttrGenAIUsageInputTokens:  200,
				AttrGenAIUsageOutputTokens: 100,
			},
			wantMissing: []string{
				AttrGenAIResponseID,
				AttrGenAIResponseFinishReasons,
				AttrGenAIUsageCacheReadInputTokens,
			},
		},
		{
			name: "zero cached tokens omitted",
			attrs: &ModelCallAttrs{
				Provider:     "google",
				RequestModel: "gemini-2.0-flash",
				FinishReason: "stop",
				InputTokens:  50,
				OutputTokens: 30,
				CachedTokens: 0,
			},
			wantKVs: map[string]any{
				AttrGenAIOperationName:     OperationChat,
				AttrGenAIProviderName:      "google",
				AttrGenAIRequestModel:      "gemini-2.0-flash",
				AttrGenAIUsageInputTokens:  50,
				AttrGenAIUsageOutputTokens: 30,
			},
			wantMissing: []string{AttrGenAIUsageCacheReadInputTokens},
		},
		{
			name:    "nil attrs is no-op",
			attrs:   nil,
			wantKVs: map[string]any{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			exporter := tracetest.NewInMemoryExporter()
			tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
			defer tp.Shutdown(context.Background()) //nolint:errcheck // test cleanup

			tracer := tp.Tracer("test")
			_, span := tracer.Start(context.Background(), "test-span",
				trace.WithSpanKind(trace.SpanKindClient))

			StampModelCallSpan(span, tt.attrs)
			span.End()

			spans := exporter.GetSpans()
			require.Len(t, spans, 1)

			attrMap := make(map[string]attribute.Value)
			for _, a := range spans[0].Attributes {
				attrMap[string(a.Key)] = a.Value
			}

			for key, want := range tt.wantKVs {
				val, ok := attrMap[key]
				if !assert.True(t, ok, "missing attribute %q", key) {
					continue
				}
				switch w := want.(type) {
				case string:
					assert.Equal(t, w, val.AsString(), "attribute %q", key)
				case int:
					assert.Equal(t, int64(w), val.AsInt64(), "attribute %q", key)
				case []string:
					assert.Equal(t, w, val.AsStringSlice(), "attribute %q", key)
				default:
					t.Fatalf("unsupported want type %T for key %q", want, key)
				}
			}

			for _, key := range tt.wantMissing {
				_, ok := attrMap[key]
				assert.False(t, ok, "attribute %q should not be present", key)
			}
		})
	}
}

func TestSpanName(t *testing.T) {
	t.Parallel()

	assert.Equal(t, "chat", SpanName(""))
	assert.Equal(t, "chat gpt-4o", SpanName("gpt-4o"))
	assert.Equal(t, "chat claude-3-5-sonnet-20241022", SpanName("claude-3-5-sonnet-20241022"))
}

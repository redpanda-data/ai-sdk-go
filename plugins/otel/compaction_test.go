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

package otel_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	pluginotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

const compactionSpanName = "redpanda.compaction"

// TestTracingInterceptor_EmitsCompactionSpan: a compaction event becomes a
// redpanda.compaction span with phase, counts and before/after breakdown.
// Observes with a span-free ctx, exercising the metadata fallback path.
func TestTracingInterceptor_EmitsCompactionSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(pluginotel.WithTracerProvider(tp))

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-compact"}, agent.Info{Name: "test-agent"})

	report := agent.CompactionReport{
		At:              time.Now().UTC(),
		Phase:           agent.CompactionPhaseProactive,
		PrunedResults:   3,
		DroppedMessages: 1,
		Before: agent.ContextUsage{
			Total: 150_000, SystemPrompt: 2_000, ToolDefinitions: 1_000,
			Text: 20_000, ToolCalls: 7_000, ToolResults: 119_500, Framing: 500,
		},
		After: agent.ContextUsage{
			Total: 90_000, SystemPrompt: 2_000, ToolDefinitions: 1_000,
			Text: 20_000, ToolCalls: 7_000, ToolResults: 59_600, Framing: 400,
		},
	}

	_, err := interceptor.InterceptTurn(t.Context(), &agent.TurnInfo{Inv: inv},
		func(ctx context.Context, info *agent.TurnInfo) (agent.FinishReason, error) {
			// Strip the span from the context to force the fallback.
			ctx = trace.ContextWithSpanContext(ctx, trace.SpanContext{})
			interceptor.ObserveEvent(ctx, info.Inv, agent.CompactionEvent{Report: report})

			return agent.FinishReasonStop, nil
		})
	require.NoError(t, err)

	var compactionSpan *tracetest.SpanStub

	spans := exporter.GetSpans()
	for i := range spans {
		if spans[i].Name == compactionSpanName {
			compactionSpan = &spans[i]
		}
	}

	require.NotNil(t, compactionSpan, "a compaction report must become a redpanda.compaction span")

	attrs := make(map[string]any, len(compactionSpan.Attributes))
	for _, kv := range compactionSpan.Attributes {
		attrs[string(kv.Key)] = kv.Value.AsInterface()
	}

	assert.Equal(t, "proactive", attrs["redpanda.compaction.phase"])
	assert.Equal(t, int64(3), attrs["redpanda.compaction.pruned_results"])
	assert.Equal(t, int64(1), attrs["redpanda.compaction.dropped_messages"])
	assert.Equal(t, int64(150_000), attrs["redpanda.compaction.before.tokens"])
	assert.Equal(t, int64(90_000), attrs["redpanda.compaction.after.tokens"])
	assert.Equal(t, int64(119_500), attrs["redpanda.compaction.before.tool_results"])
	assert.Equal(t, int64(2_000), attrs["redpanda.compaction.after.system_prompt"])

	// Child of the invocation span, stamped at the pass time.
	var invocationSpan *tracetest.SpanStub

	for i := range spans {
		if spans[i].Name != compactionSpanName {
			invocationSpan = &spans[i]
		}
	}

	require.NotNil(t, invocationSpan)
	assert.Equal(t, invocationSpan.SpanContext.SpanID(), compactionSpan.Parent.SpanID())
	assert.Equal(t, report.At, compactionSpan.StartTime)
}

// TestTracingInterceptor_CompactionParentsUnderEmissionSpan: with a live
// span in ctx, the compaction span parents under it, not under the
// invocation span stored in metadata.
func TestTracingInterceptor_CompactionParentsUnderEmissionSpan(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(pluginotel.WithTracerProvider(tp))

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-emission"}, agent.Info{Name: "test-agent"})

	var emissionSpanID trace.SpanID

	_, err := interceptor.InterceptTurn(t.Context(), &agent.TurnInfo{Inv: inv},
		func(ctx context.Context, info *agent.TurnInfo) (agent.FinishReason, error) {
			// Distinct from the invocation span stored in metadata.
			ctx, emissionSpan := tp.Tracer("test").Start(ctx, "emission-scope")
			defer emissionSpan.End()

			emissionSpanID = emissionSpan.SpanContext().SpanID()

			interceptor.ObserveEvent(ctx, info.Inv, agent.CompactionEvent{Report: agent.CompactionReport{
				At:    time.Now().UTC(),
				Phase: agent.CompactionPhaseProactive,
			}})

			return agent.FinishReasonStop, nil
		})
	require.NoError(t, err)

	var compactionSpan *tracetest.SpanStub

	spans := exporter.GetSpans()
	for i := range spans {
		if spans[i].Name == compactionSpanName {
			compactionSpan = &spans[i]
		}
	}

	require.NotNil(t, compactionSpan)
	assert.Equal(t, emissionSpanID, compactionSpan.Parent.SpanID(),
		"compaction span must parent under the emission-scope span from ctx, not the invocation span from metadata")
}

// TestTracingInterceptor_MarksLaterChatSpansCompacted: chat spans after a
// compaction carry gen_ai.conversation.compacted=true, earlier ones carry
// nothing, and the flag persists into later invocations over the session.
func TestTracingInterceptor_MarksLaterChatSpansCompacted(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	interceptor := pluginotel.New(pluginotel.WithTracerProvider(tp))

	sess := &session.State{ID: "sess-compact-flag"}

	chat := func(ctx context.Context, inv *agent.InvocationMetadata) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:                &llm.Request{},
		}

		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		_, err := handler.Generate(ctx, &llm.Request{})
		require.NoError(t, err)
	}

	inv := agent.NewInvocationMetadata(sess, agent.Info{Name: "test-agent"})

	_, err := interceptor.InterceptTurn(t.Context(), &agent.TurnInfo{Inv: inv},
		func(ctx context.Context, info *agent.TurnInfo) (agent.FinishReason, error) {
			chat(ctx, info.Inv) // before compaction

			interceptor.ObserveEvent(ctx, info.Inv, agent.CompactionEvent{Report: agent.CompactionReport{
				At:    time.Now().UTC(),
				Phase: agent.CompactionPhaseReactive,
			}})

			chat(ctx, info.Inv) // after compaction

			return agent.FinishReasonStop, nil
		})
	require.NoError(t, err)

	// Second invocation over the same (rewritten, persisted) session.
	inv2 := agent.NewInvocationMetadata(sess, agent.Info{Name: "test-agent"})

	_, err = interceptor.InterceptTurn(t.Context(), &agent.TurnInfo{Inv: inv2},
		func(ctx context.Context, info *agent.TurnInfo) (agent.FinishReason, error) {
			chat(ctx, info.Inv)

			return agent.FinishReasonStop, nil
		})
	require.NoError(t, err)

	var chatSpans []tracetest.SpanStub

	for _, span := range exporter.GetSpans() {
		if strings.HasPrefix(span.Name, "chat") {
			chatSpans = append(chatSpans, span)
		}
	}

	require.Len(t, chatSpans, 3)

	compacted := func(span tracetest.SpanStub) (bool, bool) {
		for _, kv := range span.Attributes {
			if kv.Key == "gen_ai.conversation.compacted" {
				return kv.Value.AsBool(), true
			}
		}

		return false, false
	}

	_, present := compacted(chatSpans[0])
	assert.False(t, present, "pre-compaction chat span must not carry the attribute (never false, only unset)")

	value, present := compacted(chatSpans[1])
	assert.True(t, present && value, "post-compaction chat span must carry gen_ai.conversation.compacted=true")

	value, present = compacted(chatSpans[2])
	assert.True(t, present && value, "chat span in a later invocation over the same session must carry the attribute")
}

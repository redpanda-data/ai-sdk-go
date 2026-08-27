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

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/redpanda-data/ai-sdk-go/agent"
)

// CompactionSpanName is the name of the span emitted for each context
// compaction pass. There is no gen_ai semantic convention for compaction yet,
// so the span and its attributes live under the redpanda namespace. Exported
// so consumers that filter or match compaction spans can import the name
// instead of copying the string.
const CompactionSpanName = "redpanda.compaction"

// ObserveEvent implements agent.EventObserver: each CompactionEvent becomes a
// zero-duration redpanda.compaction child span of the current emission span,
// falling back to the invocation span, stamped at the time the pass ran.
// Other events are ignored.
func (t *TracingInterceptor) ObserveEvent(ctx context.Context, inv *agent.InvocationMetadata, event agent.Event) {
	ce, ok := event.(agent.CompactionEvent)
	if !ok {
		return
	}

	// Prefer the span in ctx; fall back to the stored invocation span.
	if !trace.SpanContextFromContext(ctx).IsValid() {
		if invSpan, found := getInvocationSpan(inv); found {
			ctx = trace.ContextWithSpan(ctx, invSpan)
		}
	}

	report := ce.Report

	attrs := make([]attribute.KeyValue, 0, 19)
	attrs = append(attrs,
		attribute.String("redpanda.compaction.phase", string(report.Phase)),
		attribute.Int("redpanda.compaction.pruned_results", report.PrunedResults),
		attribute.Int("redpanda.compaction.dropped_messages", report.DroppedMessages),
	)
	attrs = append(attrs, contextUsageAttrs("redpanda.compaction.before", report.Before)...)
	attrs = append(attrs, contextUsageAttrs("redpanda.compaction.after", report.After)...)

	_, span := t.tracer.Start(ctx, CompactionSpanName,
		trace.WithTimestamp(report.At),
		trace.WithAttributes(attrs...),
	)
	span.End(trace.WithTimestamp(report.At))

	// Later chat spans carry gen_ai.conversation.compacted.
	markConversationCompacted(inv)
}

// contextUsageAttrs flattens one side of a report's context breakdown.
// Values are conservative heuristic token estimates.
func contextUsageAttrs(prefix string, u agent.ContextUsage) []attribute.KeyValue {
	return []attribute.KeyValue{
		attribute.Int(prefix+".tokens", u.Total),
		attribute.Int(prefix+".system_prompt", u.SystemPrompt),
		attribute.Int(prefix+".tool_definitions", u.ToolDefinitions),
		attribute.Int(prefix+".text", u.Text),
		attribute.Int(prefix+".reasoning", u.Reasoning),
		attribute.Int(prefix+".tool_calls", u.ToolCalls),
		attribute.Int(prefix+".tool_results", u.ToolResults),
		attribute.Int(prefix+".framing", u.Framing),
	}
}

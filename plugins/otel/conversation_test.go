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
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel/attribute"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	pluginotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// TestTracingInterceptor_SubAgentConversationGrouping verifies that a
// sub-agent session (own storage id + ConversationID override) reports the
// ROOT conversation id — not its own storage id — as gen_ai.conversation.id
// on invocation, model, and tool spans, and as SpanContext.ConversationID to
// the attribute injector.
func TestTracingInterceptor_SubAgentConversationGrouping(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	var (
		mu       sync.Mutex
		injected = map[pluginotel.SpanType]string{}
	)

	interceptor := pluginotel.New(
		pluginotel.WithTracerProvider(tp),
		pluginotel.WithAttributeInjector(func(_ context.Context, spanCtx pluginotel.SpanContext) []attribute.KeyValue {
			mu.Lock()
			defer mu.Unlock()

			injected[spanCtx.SpanType] = spanCtx.ConversationID

			return nil
		}),
	)

	// A sub-agent session: unique storage id, grouped under the root
	// conversation via the ConversationID override.
	inv := agent.NewInvocationMetadata(&session.State{
		ID:             "agent-tool-child-1",
		ConversationID: "root-conv",
	}, agent.Info{Name: "child-agent"})

	_, _ = interceptor.InterceptTurn(t.Context(), &agent.TurnInfo{Inv: inv}, func(ctx context.Context, _ *agent.TurnInfo) (agent.FinishReason, error) {
		modelInfo := &agent.ModelCallInfo{
			InvocationMetadata: inv,
			Model:              &mockModelInfo{name: "gpt-4", provider: "openai"},
			Req:                &llm.Request{},
		}
		handler := interceptor.InterceptModel(ctx, modelInfo, &mockModelHandler{})
		_, err := handler.Generate(ctx, &llm.Request{})
		require.NoError(t, err)

		req := &llm.ToolRequestPart{
			Name:      "get_weather",
			ID:        "tool-call-1",
			Arguments: json.RawMessage(`{}`),
		}
		_, err = interceptor.InterceptToolExecution(ctx, &agent.ToolCallInfo{Inv: inv, Req: req},
			func(_ context.Context, _ *agent.ToolCallInfo) (*llm.ToolResponsePart, error) {
				return &llm.ToolResponsePart{Result: json.RawMessage(`"ok"`)}, nil
			})
		require.NoError(t, err)

		return agent.FinishReasonStop, nil
	})

	spans := exporter.GetSpans()
	require.Len(t, spans, 3, "expected invocation, chat, and tool spans")

	// Every span reports the root conversation, never the storage session id.
	for i := range spans {
		assertHasAttribute(t, spans[i].Attributes, "gen_ai.conversation.id", "root-conv")
	}

	// The injector saw the same grouping id for all three span types.
	wantSpanTypes := []pluginotel.SpanType{
		pluginotel.SpanTypeInvocation,
		pluginotel.SpanTypeModel,
		pluginotel.SpanTypeTool,
	}
	for _, st := range wantSpanTypes {
		assert.Equal(t, "root-conv", injected[st], "SpanContext.ConversationID for %s", st)
	}

	// Sanity: no span leaked the storage id as the conversation id.
	for i := range spans {
		for _, attr := range spans[i].Attributes {
			if string(attr.Key) == "gen_ai.conversation.id" {
				assert.NotEqual(t, "agent-tool-child-1", attr.Value.AsString(),
					"span %s must not group under the storage session id", spans[i].Name)
			}
		}
	}

	// The spans still nest in one trace (parent/child linkage is unchanged).
	var invocationSpan *string

	for i := range spans {
		if strings.HasPrefix(spans[i].Name, "invoke_agent") {
			name := spans[i].Name
			invocationSpan = &name
		}
	}

	require.NotNil(t, invocationSpan)
}

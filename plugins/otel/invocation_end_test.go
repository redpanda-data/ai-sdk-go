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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	pluginotel "github.com/redpanda-data/ai-sdk-go/plugins/otel"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

type loopTool struct{}

func (loopTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{Name: "loop", Description: "always asks for another turn"}
}

func (loopTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

// The loop, not a turn, ends an invocation that exhausts its turn budget, so
// the invocation span must be closed from the InvocationEndEvent; otherwise
// every chat span of the run points at a parent that is never exported.
func TestTracingInterceptor_EndsInvocationSpanOnMaxTurns(t *testing.T) {
	t.Parallel()

	exporter, tp := setupTracer()
	defer tp.Shutdown(t.Context()) //nolint:errcheck // Test cleanup

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(loopTool{}))

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondWithToolCall("loop", map[string]any{})

	ag, err := llmagent.New(
		"looping-agent",
		"You are a test assistant",
		model,
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(2),
		llmagent.WithInterceptors(pluginotel.New(pluginotel.WithTracerProvider(tp))),
	)
	require.NoError(t, err)

	sess := &session.State{
		ID:       "sess-max-turns",
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("go"))},
	}

	var finish agent.FinishReason

	for ev, err := range ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{Name: "looping-agent"})) {
		require.NoError(t, err)

		if end, ok := ev.(agent.InvocationEndEvent); ok {
			finish = end.FinishReason
		}
	}

	require.Equal(t, agent.FinishReasonMaxTurns, finish)

	spans := exporter.GetSpans()
	exported := make(map[string]bool, len(spans))

	var invocationSpans int

	for _, s := range spans {
		exported[s.SpanContext.SpanID().String()] = true

		if s.Name == "invoke_agent looping-agent" {
			invocationSpans++
		}
	}

	require.Equal(t, 1, invocationSpans, "invocation span must be exported exactly once")

	for _, s := range spans {
		if s.Parent.IsValid() {
			assert.True(t, exported[s.Parent.SpanID().String()], "span %q has an unexported parent", s.Name)
		}
	}
}

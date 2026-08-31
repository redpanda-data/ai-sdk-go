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

package llmagent_test

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

type turnSentinelKey struct{}

// observedEvent records what an observer saw and with which context scope.
type observedEvent struct {
	event       agent.Event
	hasSentinel bool
}

// turnCtxObserver derives a sentinel-carrying turn context (as the otel
// plugin does with its invocation span) and records which observed events
// carried it.
type turnCtxObserver struct {
	mu       sync.Mutex
	observed []observedEvent
}

func (o *turnCtxObserver) InterceptTurn(ctx context.Context, info *agent.TurnInfo, next agent.TurnNext) (agent.FinishReason, error) {
	return next(context.WithValue(ctx, turnSentinelKey{}, true), info)
}

func (o *turnCtxObserver) ObserveEvent(ctx context.Context, _ *agent.InvocationMetadata, event agent.Event) {
	o.mu.Lock()
	defer o.mu.Unlock()

	sentinel, _ := ctx.Value(turnSentinelKey{}).(bool)
	o.observed = append(o.observed, observedEvent{event: event, hasSentinel: sentinel})
}

// TestEventObserver_EmissionScopeContext: turn events are observed with the
// interceptor-derived turn context, lifecycle events with the run context.
func TestEventObserver_EmissionScopeContext(t *testing.T) {
	t.Parallel()

	obs := &turnCtxObserver{}
	ag, _ := compactionAgent(t, 16_000,
		llmagent.WithCompaction(llmagent.CompactionConfig{}),
		llmagent.WithInterceptors(obs),
	)

	sess := &session.State{ID: "observer"}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("start")))

	for i := range 4 {
		oldTurn(sess, i, 3_000)
	}

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser,
		llm.NewTextPart(strings.Repeat("new question ", 800))))

	events, err := runOnce(t, ag, sess)
	require.NoError(t, err)
	require.NotEmpty(t, compactionEvents(events), "setup must trigger compaction")

	// Every consumed event was also observed, in the same order.
	require.Len(t, obs.observed, len(events), "observer must see exactly the consumer's events")

	for i, oe := range obs.observed {
		assert.IsType(t, events[i], oe.event, "event %d: observer order must match consumer order", i)
	}

	var sawCompaction, sawMessage, sawTurnStarted, sawEnd bool

	for _, oe := range obs.observed {
		switch ev := oe.event.(type) {
		case agent.CompactionEvent:
			sawCompaction = true

			assert.True(t, oe.hasSentinel, "compaction events are emitted inside the turn and must carry the turn context")
		case agent.MessageEvent:
			sawMessage = true

			assert.True(t, oe.hasSentinel, "message events are emitted inside the turn and must carry the turn context")
		case agent.StatusEvent:
			if ev.Stage == agent.StatusStageTurnStarted {
				sawTurnStarted = true

				assert.False(t, oe.hasSentinel, "turn-started is emitted outside the turn and must not carry the turn context")
			}
		case agent.InvocationEndEvent:
			sawEnd = true

			assert.False(t, oe.hasSentinel, "invocation-end is emitted outside the turn and must not carry the turn context")
		}
	}

	assert.True(t, sawCompaction, "observer must see the compaction event")
	assert.True(t, sawMessage, "observer must see the assistant message")
	assert.True(t, sawTurnStarted, "observer must see turn-started")
	assert.True(t, sawEnd, "observer must see invocation-end")
}

// observerOnly implements just EventObserver.
type observerOnly struct{}

func (observerOnly) ObserveEvent(context.Context, *agent.InvocationMetadata, agent.Event) {}

// TestEventObserver_ObserverOnlyInterceptorIsValid: an observer-only
// interceptor passes registration validation.
func TestEventObserver_ObserverOnlyInterceptorIsValid(t *testing.T) {
	t.Parallel()

	model := fakellm.NewFakeModel()
	model.When(fakellm.Any()).ThenRespondText("done")

	_, err := llmagent.New("observer-agent", "You are a test assistant.", model,
		llmagent.WithInterceptors(observerOnly{}))
	require.NoError(t, err)
}

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

package a2a

import (
	"context"
	"iter"
	"log/slog"
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// testUserKey stands in for an authenticating middleware's context key.
type testUserKey struct{}

func resolveTestUser(ctx context.Context) string {
	v, _ := ctx.Value(testUserKey{}).(string)

	return v
}

// attributeCapturingAgent records the attributes it was handed and ends
// immediately, so tests need no model.
type attributeCapturingAgent struct {
	got map[string]string
}

func (*attributeCapturingAgent) Info() agent.Info {
	return agent.Info{Name: "capture"}
}

func (*attributeCapturingAgent) InputSchema() map[string]any {
	return nil
}

func (a *attributeCapturingAgent) Run(
	_ context.Context,
	inv *agent.InvocationMetadata,
) iter.Seq2[agent.Event, error] {
	a.got = inv.Attributes()

	return func(yield func(agent.Event, error) bool) {
		yield(agent.InvocationEndEvent{FinishReason: agent.FinishReasonStop}, nil)
	}
}

// executeWithUserID drives one Execute, attaching userInCtx to the request
// context when non-empty, and returns the attributes the invocation carried.
func executeWithUserID(t *testing.T, userInCtx string, opts ...Option) map[string]string {
	t.Helper()

	ag := &attributeCapturingAgent{}
	runnerInstance, err := runner.New(ag, session.NewInMemoryStore())
	require.NoError(t, err)

	executor := NewExecutor(ag, runnerInstance, slog.Default(), opts...)

	reqCtx := &a2asrv.RequestContext{
		ContextID: "test-context",
		TaskID:    "test-task",
		Message:   a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "hi"}),
	}

	ctx := context.Background()
	if userInCtx != "" {
		ctx = context.WithValue(ctx, testUserKey{}, userInCtx)
	}

	// Buffered past the events one run writes, so Execute never blocks.
	queueMgr := eventqueue.NewInMemoryManager(eventqueue.WithQueueBufferSize(100))
	queue, err := queueMgr.GetOrCreate(ctx, reqCtx.TaskID)
	require.NoError(t, err)

	require.NoError(t, executor.Execute(ctx, reqCtx, queue))

	return ag.got
}

// The executor is built once at startup while the caller's identity arrives
// per request, so attributes can only come from the request context.
func TestExecutor_AttributesFunc(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		userInCtx string
		opts      []Option
		want      map[string]string
	}{
		{
			name:      "no function asserts nothing",
			userInCtx: "alice@example.test",
			want:      map[string]string{},
		},
		{
			name:      "resolved values reach the invocation",
			userInCtx: "alice@example.test",
			opts:      []Option{WithAttributesFunc(testAttributes)},
			want: map[string]string{
				agent.AttrUserID: "alice@example.test",
				"tenant.id":      "acme",
			},
		},
		{
			name: "empty values assert nothing",
			opts: []Option{WithAttributesFunc(testAttributes)},
			want: map[string]string{"tenant.id": "acme"},
		},
		{
			name: "a nil function is ignored rather than panicking",
			opts: []Option{WithAttributesFunc(nil)},
			want: map[string]string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, executeWithUserID(t, tt.userInCtx, tt.opts...))
		})
	}
}

// testAttributes stands in for middleware that knows the user and tenant.
func testAttributes(ctx context.Context) map[string]string {
	return map[string]string{
		agent.AttrUserID: resolveTestUser(ctx),
		"tenant.id":      "acme",
	}
}

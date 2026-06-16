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

package agent_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func TestContextWithInvocation_RoundTrip(t *testing.T) {
	t.Parallel()

	inv := agent.NewInvocationMetadata(&session.State{ID: "sess-1"}, agent.Info{Name: "a"})

	ctx := agent.ContextWithInvocation(context.Background(), inv)

	got, ok := agent.InvocationFromContext(ctx)
	require.True(t, ok)
	assert.Same(t, inv, got)
}

func TestInvocationFromContext_Absent(t *testing.T) {
	t.Parallel()

	got, ok := agent.InvocationFromContext(context.Background())
	assert.False(t, ok)
	assert.Nil(t, got)
}

func TestContextWithInvocation_Nil(t *testing.T) {
	t.Parallel()

	// A nil invocation must not be stored; the context is returned unchanged.
	ctx := agent.ContextWithInvocation(context.Background(), nil)

	got, ok := agent.InvocationFromContext(ctx)
	assert.False(t, ok)
	assert.Nil(t, got)
}

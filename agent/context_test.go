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

	"github.com/redpanda-data/ai-sdk-go/agent"
)

func TestContextWithConversationID_RoundTrip(t *testing.T) {
	t.Parallel()

	ctx := agent.ContextWithConversationID(context.Background(), "conv-1")

	assert.Equal(t, "conv-1", agent.ConversationIDFromContext(ctx))
}

func TestConversationIDFromContext_Absent(t *testing.T) {
	t.Parallel()

	assert.Empty(t, agent.ConversationIDFromContext(context.Background()))
}

func TestContextWithConversationID_EmptyInherits(t *testing.T) {
	t.Parallel()

	// An empty id is a no-op: conversation identity is inherited, so an
	// enclosing conversation id must stay visible.
	ctx := agent.ContextWithConversationID(context.Background(), "parent")
	ctx = agent.ContextWithConversationID(ctx, "")

	assert.Equal(t, "parent", agent.ConversationIDFromContext(ctx))
}

func TestContextWithConversationID_NestedOverrides(t *testing.T) {
	t.Parallel()

	// A non-empty id shadows an inherited one for the derived context only.
	outer := agent.ContextWithConversationID(context.Background(), "root")
	inner := agent.ContextWithConversationID(outer, "child")

	assert.Equal(t, "child", agent.ConversationIDFromContext(inner))
	assert.Equal(t, "root", agent.ConversationIDFromContext(outer))
}

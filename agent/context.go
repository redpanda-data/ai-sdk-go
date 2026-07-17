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

package agent

import "context"

// conversationIDCtxKey is the unexported key under which the conversation
// grouping id is stored in a context.
type conversationIDCtxKey struct{}

// ContextWithConversationID returns a copy of ctx carrying the conversation
// grouping id (see session.ConversationID). Agent implementations that execute
// tools should set it — from session.ConversationID of their invocation's
// session — for the duration of each tool call, so tools that spawn in-process
// sub-agents (notably agenttool) can group the sub-agent's telemetry under the
// calling conversation. Tools that do not read it are unaffected.
//
// An empty id returns ctx unchanged: conversation identity is inherited, so an
// enclosing conversation id stays visible when a caller has none of its own.
// The resolved id of a session (session.ConversationID) is never empty for a
// session with an ID, so producers only hit this path without a usable session.
func ContextWithConversationID(ctx context.Context, id string) context.Context {
	if id == "" {
		return ctx
	}

	return context.WithValue(ctx, conversationIDCtxKey{}, id)
}

// ConversationIDFromContext returns the conversation grouping id stored in ctx
// by ContextWithConversationID, or "" when none is set.
func ConversationIDFromContext(ctx context.Context) string {
	id, _ := ctx.Value(conversationIDCtxKey{}).(string)
	return id
}

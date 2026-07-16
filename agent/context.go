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

// invocationCtxKey is the unexported key under which the active invocation
// metadata is stored in a context.
type invocationCtxKey struct{}

// ContextWithInvocation returns a copy of ctx carrying the given invocation
// metadata. Agents put their invocation into the context before executing tools
// so that tools (notably agenttool) can access the calling agent's invocation —
// for example to share the parent's session id with an in-process sub-agent.
//
// A nil invocation returns ctx unchanged.
func ContextWithInvocation(ctx context.Context, inv *InvocationMetadata) context.Context {
	if inv == nil {
		return ctx
	}

	return context.WithValue(ctx, invocationCtxKey{}, inv)
}

// InvocationFromContext returns the invocation metadata stored in ctx by
// ContextWithInvocation, if any. The second return value reports whether an
// invocation was present.
func InvocationFromContext(ctx context.Context) (*InvocationMetadata, bool) {
	inv, ok := ctx.Value(invocationCtxKey{}).(*InvocationMetadata)
	return inv, ok
}

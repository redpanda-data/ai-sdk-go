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

package tool

import "context"

// toolCallIDKey keys the LLM-issued tool-call id in an execution context.
type toolCallIDKey struct{}

// WithCallID returns a child context carrying the LLM-issued tool-call id
// (llm.ToolRequestPart.ID) for the tool execution about to run. The Registry
// sets this immediately before invoking a tool so that transport-level tool
// implementations — notably the MCP client — can forward the id to the server
// for cross-layer correlation (e.g. so a gateway can stamp gen_ai.tool.call.id
// on its tool-execution span and tie it back to the model turn that requested
// it). Tools that don't care simply ignore it.
func WithCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, toolCallIDKey{}, id)
}

// CallIDFromContext returns the tool-call id set by WithCallID, or
// ("", false) when absent.
func CallIDFromContext(ctx context.Context) (string, bool) {
	id, ok := ctx.Value(toolCallIDKey{}).(string)
	return id, ok
}

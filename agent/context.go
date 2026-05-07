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

type contextKey string

const (
	globalInstructionsKey contextKey = "gen_ai.global_instructions"
)

// ContextWithGlobalInstructions returns a new context with the provided global instructions.
//
// Global instructions are system-wide directives that apply to all agents in an
// invocation tree. When an agent is called (directly or as a tool), these
// instructions are appended to its base system prompt.
//
// Use this to propagate cross-cutting constraints like "Always respond in JSON"
// or "Be extremely concise" across hierarchical agent calls.
func ContextWithGlobalInstructions(ctx context.Context, instructions string) context.Context {
	return context.WithValue(ctx, globalInstructionsKey, instructions)
}

// GlobalInstructions retrieves the global instructions from the context.
// Returns an empty string if no instructions are set.
func GlobalInstructions(ctx context.Context) string {
	val, ok := ctx.Value(globalInstructionsKey).(string)
	if !ok {
		return ""
	}
	return val
}

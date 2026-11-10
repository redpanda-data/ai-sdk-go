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

package llmagent

import (
	"context"
	"fmt"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolloading"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Toolset resolves a caller's configured tools before each model step. The
// context and invocation are the same ones supplied to the turn interceptor.
// A nil registry means no tools. Errors stop the invocation before the model
// call; reactive context-overflow retries reuse the resolved snapshot.
//
// Return a registry built with tool.NewRegistry, or a custom registry supporting
// tool.Snapshot. Registration options (timeouts, result limits, deferral and
// groups) are preserved. The SDK owns only the snapshot, not the tools or their
// connections: callers manage their lifetimes and concurrent access. Toolsets
// may be called concurrently by separate invocations of the same agent.
// Use WithToolsets to compose these with the agent's static registry.
type Toolset func(context.Context, *agent.InvocationMetadata) (tool.Registry, error)

type stepTools struct {
	registry tool.Registry
	loader   *toolloading.Loader
}

func (a *LLMAgent) resolveTools(ctx context.Context, inv *agent.InvocationMetadata) (*stepTools, error) {
	if len(a.config.toolsets) == 0 {
		return &stepTools{registry: a.config.tools, loader: a.loader}, nil
	}

	registries := make([]tool.Registry, 0, len(a.config.toolsets)+1)
	if a.config.tools != nil {
		snapshot, err := tool.Snapshot(a.config.tools)
		if err != nil {
			return nil, fmt.Errorf("llmagent: static tools: %w", err)
		}

		registries = append(registries, snapshot)
	}

	for i, set := range a.config.toolsets {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		registry, err := set(ctx, inv)
		if err != nil {
			return nil, fmt.Errorf("llmagent: toolset %d: %w", i, err)
		}

		if err := ctx.Err(); err != nil {
			return nil, err
		}

		if registry == nil {
			continue
		}

		snapshot, err := tool.Snapshot(registry)
		if err != nil {
			return nil, fmt.Errorf("llmagent: toolset %d: %w", i, err)
		}

		registries = append(registries, snapshot)
	}

	registry, err := tool.Snapshot(registries...)
	if err != nil {
		return nil, fmt.Errorf("llmagent: toolsets: %w", err)
	}

	defs := registry.List()
	if slices.ContainsFunc(defs, isDeferred) && slices.ContainsFunc(defs, func(d llm.ToolDefinition) bool {
		return d.Name == toolloading.SearchToolName
	}) {
		return nil, fmt.Errorf("llmagent: lazy loading reserves the tool name %q; rename the registered tool", toolloading.SearchToolName)
	}

	return &stepTools{registry: registry, loader: toolloading.New(registry, a.config.model, toolloading.Config{
		ForceLocal:    a.config.toolLoading.ForceLocal,
		MaxLoadTokens: a.config.toolLoading.MaxLoadTokens,
	})}, nil
}

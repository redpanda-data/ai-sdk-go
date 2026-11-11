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

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolloading"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

type stepTools struct {
	registry    tool.Registry
	definitions []llm.ToolDefinition
	loader      *toolloading.Loader
}

func (a *LLMAgent) resolveTools(ctx context.Context) (*stepTools, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(a.config.registries) == 0 {
		return &stepTools{}, nil
	}

	registry, err := tool.Snapshot(ctx, a.config.registries...)
	if err != nil {
		return nil, fmt.Errorf("llmagent: tools: %w", err)
	}

	defs, err := registry.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("llmagent: tools: %w", err)
	}

	if slices.ContainsFunc(defs, isDeferred) && slices.ContainsFunc(defs, func(d llm.ToolDefinition) bool {
		return d.Name == toolloading.SearchToolName
	}) {
		return nil, fmt.Errorf("llmagent: lazy loading reserves the tool name %q; rename the registered tool", toolloading.SearchToolName)
	}

	return &stepTools{registry: registry, definitions: defs, loader: toolloading.New(defs, a.config.model, toolloading.Config{
		ForceLocal:    a.config.toolLoading.ForceLocal,
		MaxLoadTokens: a.config.toolLoading.MaxLoadTokens,
	})}, nil
}

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

package mcp

import (
	"context"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestListToolsSortedAndStable guards prompt caching on the direct MCP-client
// path. ListTools() is documented as something callers "Pass to an LLM for tool
// calling", so its order becomes part of the request prefix every provider
// serializes. Ranging the internal tools map without sorting reorders the slice
// on every call, busting the upstream cache. The agent path goes through the
// registry (covered separately), but this public API must be deterministic on
// its own.
func TestListToolsSortedAndStable(t *testing.T) {
	t.Parallel()

	// Names deliberately NOT in sorted order, and enough that randomized map
	// iteration is overwhelmingly unlikely to emit them sorted by chance.
	names := []string{
		"srv__zeta", "srv__yankee", "srv__xray", "srv__whiskey",
		"srv__victor", "srv__uniform", "srv__tango", "srv__sierra",
		"srv__romeo", "srv__quebec", "srv__papa", "srv__oscar",
		"srv__november", "srv__mike", "srv__lima", "srv__kilo",
	}

	c := &clientImpl{
		bgCtx: context.Background(),
		tools: make(map[string]*toolWrapper, len(names)),
	}
	for _, n := range names {
		c.tools[n] = &toolWrapper{definition: llm.ToolDefinition{Name: n}}
	}

	want := make([]string, len(names))
	copy(want, names)
	sort.Strings(want)

	defs, err := c.ListTools(context.Background())
	require.NoError(t, err)
	assert.Equal(t, want, toolNames(defs), "ListTools() must return definitions sorted by name")

	// Order must be identical across many calls.
	const iterations = 50

	orderings := make(map[string]struct{})

	for range iterations {
		got, err := c.ListTools(context.Background())
		require.NoError(t, err)

		orderings[strings.Join(toolNames(got), ",")] = struct{}{}
	}

	assert.Lenf(t, orderings, 1,
		"ListTools() order must be stable across %d calls, got %d distinct orderings",
		iterations, len(orderings))
}

func toolNames(defs []llm.ToolDefinition) []string {
	names := make([]string, len(defs))
	for i, d := range defs {
		names[i] = d.Name
	}

	return names
}

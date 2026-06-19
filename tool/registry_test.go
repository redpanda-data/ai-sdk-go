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

package tool_test

import (
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TestRegistryListSortedAndStable guards prompt caching: every provider
// serializes req.Tools verbatim into the request prefix (the tools block
// precedes system + messages for Anthropic, and is part of the cached prefix
// everywhere). If List() returns tools in a different order on each call, the
// prefix changes every turn and the upstream cache never hits.
//
// List() must therefore return a deterministic, name-sorted order regardless of
// registration order, and that order must be identical across repeated calls.
func TestRegistryListSortedAndStable(t *testing.T) {
	t.Parallel()

	// Names deliberately NOT in sorted order, and enough of them that Go's
	// randomized map iteration is overwhelmingly unlikely to emit them sorted
	// by chance (1/16! ~ 5e-14).
	names := []string{
		"zeta", "yankee", "x-ray", "whiskey", "victor", "uniform",
		"tango", "sierra", "romeo", "quebec", "papa", "oscar",
		"november", "mike", "lima", "kilo",
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	for _, n := range names {
		require.NoError(t, registry.Register(&mockTool{name: n}))
	}

	want := make([]string, len(names))
	copy(want, names)
	sort.Strings(want)

	// Order must equal the name-sorted expectation.
	assert.Equal(t, want, definitionNames(registry.List()),
		"List() must return tool definitions sorted by name")

	// Order must be identical across many calls (no map-iteration randomization).
	// Pre-fix this almost surely yields >= 2 distinct orderings; post-fix exactly 1.
	const iterations = 50

	orderings := make(map[string]struct{})
	for range iterations {
		orderings[strings.Join(definitionNames(registry.List()), ",")] = struct{}{}
	}

	assert.Lenf(t, orderings, 1,
		"List() order must be stable across %d calls, got %d distinct orderings",
		iterations, len(orderings))
}

// definitionNames extracts tool names from a slice of definitions, preserving order.
func definitionNames(defs []llm.ToolDefinition) []string {
	names := make([]string, len(defs))
	for i, d := range defs {
		names[i] = d.Name
	}

	return names
}

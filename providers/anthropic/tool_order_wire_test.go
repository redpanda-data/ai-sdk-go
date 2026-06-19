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

package anthropic

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TestToolOrderReachesTheWire proves the real-world impact behind the
// registry/MCP sort: the order of req.Tools is serialized verbatim into the
// actual Anthropic request, so a reordered tool list is a byte-different request
// prefix — a guaranteed cache miss. Together with TestRegistryListSortedAndStable
// (which shows List() returned a randomized order before the fix), this closes
// the causal chain: randomized List() -> randomized wire -> cache never hits.
func TestToolOrderReachesTheWire(t *testing.T) {
	t.Parallel()

	m := newWireTestModel(t)

	orderA, bytesA := serializedRequest(t, m, toolDefs("alpha", "bravo", "charlie"))
	orderB, bytesB := serializedRequest(t, m, toolDefs("charlie", "bravo", "alpha"))

	// The mapper preserves req.Tools order verbatim — it does not re-sort. So
	// whatever order the tool list arrives in is exactly what hits the wire.
	assert.Equal(t, []string{"alpha", "bravo", "charlie"}, orderA)
	assert.Equal(t, []string{"charlie", "bravo", "alpha"}, orderB)

	// And that difference is real bytes on the wire: the two marshaled requests
	// differ solely because the tools were ordered differently.
	assert.NotEqual(t, bytesA, bytesB,
		"reordering tools must change the serialized request — this is what busts the cache")
}

// TestRegistryToolOrderIsStableOnTheWire closes the loop end to end: tools
// registered in arbitrary order produce a single deterministic wire order
// (sorted), identical across independent List()->map cycles. Before the
// registry sort this produced a different wire order on essentially every call.
func TestRegistryToolOrderIsStableOnTheWire(t *testing.T) {
	t.Parallel()

	m := newWireTestModel(t)

	build := func() string {
		reg := tool.NewRegistry(tool.RegistryConfig{})
		for _, n := range []string{"zeta", "mike", "alpha", "romeo", "bravo"} {
			require.NoError(t, reg.Register(fakeTool{name: n}))
		}

		order, _ := serializedRequest(t, m, reg.List())

		return strings.Join(order, ",")
	}

	const want = "alpha,bravo,mike,romeo,zeta"

	assert.Equal(t, want, build(), "registry tool order must reach the wire sorted")

	// Rebuild many times: the wire order must never vary.
	for range 20 {
		assert.Equal(t, want, build())
	}
}

// serializedRequest maps a request carrying the given tools through the real
// Anthropic request mapper and returns the tool names in wire order plus the
// fully marshaled request body.
func serializedRequest(t *testing.T, m *Model, tools []llm.ToolDefinition) ([]string, string) {
	t.Helper()

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleSystem,
				Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
			},
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart("hi")},
			},
		},
		Tools: tools,
	}

	apiReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)

	names := make([]string, len(apiReq.Tools))

	for i, tn := range apiReq.Tools {
		require.NotNil(t, tn.OfTool, "expected a function tool")
		names[i] = tn.OfTool.Name
	}

	raw, err := json.Marshal(apiReq)
	require.NoError(t, err)

	return names, string(raw)
}

func toolDefs(names ...string) []llm.ToolDefinition {
	defs := make([]llm.ToolDefinition, len(names))
	for i, n := range names {
		defs[i] = llm.ToolDefinition{
			Name:        n,
			Description: n,
			Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
		}
	}

	return defs
}

func newWireTestModel(t *testing.T) *Model {
	t.Helper()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet45)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	return m
}

type fakeTool struct{ name string }

func (f fakeTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        f.name,
		Description: f.name,
		Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
	}
}

func (fakeTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

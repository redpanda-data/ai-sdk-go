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
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/tokens"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// The measurement fixture: six MCP servers of ten tools each, the shape this
// design exists for. Schemas are deliberately verbose in the way real MCP
// schemas are - per-property descriptions, enums, nested objects - because the
// whole claim is about how many bytes of schema ride on every request.
var measureGroups = []string{"servicenow", "jira", "confluence", "github", "datadog", "salesforce"}

var measureVerbs = []string{
	"search", "get", "create", "update", "close", "comment", "assign", "link", "list", "export",
}

// buildMeasureRegistry returns a 60-tool registry. With lazy=true every tool
// of every server is deferred except each server's "search" entry point, which
// is the configuration the design recommends to operators.
func buildMeasureRegistry(tb testing.TB, lazy bool) tool.Registry {
	tb.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	require.NoError(tb, registry.Register(&stubTool{def: llm.ToolDefinition{
		Name:        "todo_write",
		Description: "Record the plan for a multi-step task so progress survives a long conversation.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"items":{"type":"array","items":{"type":"string"}}}}`),
	}}))

	for _, group := range measureGroups {
		for _, verb := range measureVerbs {
			name := group + "__" + verb + "_record"

			var opts []tool.Option
			opts = append(opts, tool.WithGroup(llm.ToolGroup{Name: group}))

			if lazy && verb != "search" {
				opts = append(opts, tool.WithDeferred())
			}

			require.NoError(tb, registry.Register(&stubTool{def: llm.ToolDefinition{
				Name: name,
				Description: fmt.Sprintf(
					"%s a %s record. Accepts the record identifier, an optional field mask and "+
						"an optional pagination cursor. Returns the full record body including "+
						"system fields, audit metadata and every linked entity.", verb, group),
				Parameters: json.RawMessage(fmt.Sprintf(`{
  "type": "object",
  "properties": {
    "record_id": {"type": "string", "description": "The %[1]s record identifier, e.g. a sys_id or key."},
    "fields": {"type": "array", "items": {"type": "string"}, "description": "Field mask. Omit for every field."},
    "cursor": {"type": "string", "description": "Opaque pagination cursor from a previous response."},
    "state": {"type": "string", "enum": ["new", "in_progress", "on_hold", "resolved", "closed"], "description": "Lifecycle state filter."},
    "assignment": {
      "type": "object",
      "description": "Assignment target.",
      "properties": {
        "group": {"type": "string", "description": "Assignment group name."},
        "user": {"type": "string", "description": "Assignee identifier."}
      }
    }
  },
  "required": ["record_id"],
  "additionalProperties": false
}`, group)),
			}}, opts...))
		}
	}

	return registry
}

func toolBytes(defs []llm.ToolDefinition) int {
	encoded, err := json.Marshal(defs)
	if err != nil {
		return 0
	}

	return len(encoded)
}

// TestToolLoadingPrefixCost measures what lazy loading actually buys on the
// 60-tool registry this design targets, with no provider involved: the tool
// schema carried in the request prefix, before and after.
func TestToolLoadingPrefixCost(t *testing.T) {
	t.Parallel()

	script := func(model *fakellm.FakeModel) {
		model.When(fakellm.Not(fakellm.HasTool("servicenow__create_record"))).
			ThenRespondWithToolCall("tool_search", map[string]any{
				"query": "select:servicenow__create_record,jira__create_record",
			})
		model.When(fakellm.And(
			fakellm.HasTool("servicenow__create_record"),
			fakellm.Not(fakellm.LastMessageHasToolResponse("servicenow__create_record")),
		)).ThenRespondWithToolCall("servicenow__create_record", map[string]any{"record_id": "INC1"})
		model.When(fakellm.Any()).ThenRespondText("Done.")
	}

	run := func(t *testing.T, lazy bool) []fakellm.Call {
		t.Helper()

		registry := buildMeasureRegistry(t, lazy)
		require.Len(t, registry.List(), 61, "60 MCP tools plus one built-in")

		model := fakellm.NewFakeModel()
		script(model)

		opts := []Option{WithTools(registry)}

		ag, err := New("desk", "You are a service desk agent.", model, opts...)
		require.NoError(t, err)

		sess := &session.State{
			ID:       "measure",
			Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Open an incident"))},
		}
		runAgent(t, ag, sess)

		calls := model.CallsMatching(fakellm.Any())
		require.NotEmpty(t, calls)

		return calls
	}

	baseline := run(t, false)
	lazy := run(t, true)

	baseBytes := toolBytes(baseline[0].Request.Tools)
	lazyBytes := toolBytes(lazy[0].Request.Tools)
	manifestBytes := len(lazy[0].Request.Messages[0].TextContent()) -
		len(baseline[0].Request.Messages[0].TextContent())

	// The same measurement in tokens, using the counter the agent itself
	// budgets against (estimateToolTokens / estimateMessageTokens). It counts
	// high on purpose - ~3 chars per token against a real tokenizer's ~4 - so
	// treat the absolute figures as a conservative ceiling. The ratio is
	// tokenizer-independent: both sides go through the same counter.
	baseTokens := tokens.Tools(baseline[0].Request.Tools)
	lazyTokens := tokens.Tools(lazy[0].Request.Tools)
	manifestTokens := tokens.Message(lazy[0].Request.Messages[0]) -
		tokens.Message(baseline[0].Request.Messages[0])

	byteReduction := 100 * (1 - float64(lazyBytes+manifestBytes)/float64(baseBytes))
	tokenReduction := 100 * (1 - float64(lazyTokens+manifestTokens)/float64(baseTokens))

	t.Logf("first request, tool schemas: baseline %d tools = %d tokens (%d B) -> lazy %d tools = %d tokens (%d B)",
		len(baseline[0].Request.Tools), baseTokens, baseBytes,
		len(lazy[0].Request.Tools), lazyTokens, lazyBytes)
	t.Logf("manifest added to the system prompt: %d tokens (%d B)", manifestTokens, manifestBytes)
	t.Logf("net prefix reduction: %.1f%% by token, %.1f%% by byte", tokenReduction, byteReduction)
	t.Logf("model calls: baseline %d, lazy %d", len(baseline), len(lazy))

	assert.Greater(t, tokenReduction, 80.0,
		"lazy loading must cut the first request's prefix by more than 80%% net of the manifest")
	assert.InDelta(t, byteReduction, tokenReduction, 2.0,
		"the saving must not depend on the counter: bytes and tokens should agree closely")
	assert.Less(t, manifestTokens, baseTokens/8,
		"the manifest must cost a small fraction of the schemas it replaces")

	// One extra model call for the search, and only one.
	assert.Len(t, lazy, len(baseline)+1,
		"discovery must cost exactly one extra round trip")

	// The tools array grows once - at the load - then converges. Every request
	// after the load is byte-identical in the tools block, which is what makes
	// this one cache write rather than one per turn.
	shapes := make([]string, 0, len(lazy))

	for _, call := range lazy {
		names := make([]string, 0, len(call.Request.Tools))
		for _, def := range call.Request.Tools {
			names = append(names, def.Name)
		}

		shapes = append(shapes, strings.Join(names, ","))
	}

	distinct := map[string]int{}
	for _, shape := range shapes {
		distinct[shape]++
	}

	assert.Len(t, distinct, 2,
		"the tools array must take exactly two shapes: before the load and after it")
	assert.Equal(t, shapes[len(shapes)-1], shapes[1],
		"every request after the load must carry the same tools array")

	// And the manifest is stable across all of them, in both directions: it
	// does not shrink as tools load, and it does not grow.
	for i, call := range lazy {
		assert.Equal(t, lazy[0].Request.Messages[0].TextContent(),
			call.Request.Messages[0].TextContent(),
			"call %d changed the system prompt", i)
	}

	// The always-on set is exactly what the operator declared: one built-in
	// plus each server's search entry point, plus tool_search.
	firstNames := make([]string, 0, len(lazy[0].Request.Tools))
	for _, def := range lazy[0].Request.Tools {
		firstNames = append(firstNames, def.Name)
	}

	assert.Equal(t, []string{
		"confluence__search_record",
		"datadog__search_record",
		"github__search_record",
		"jira__search_record",
		"salesforce__search_record",
		"servicenow__search_record",
		"todo_write",
		"tool_search",
	}, firstNames)
}

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
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

//
// Fixtures shared by every tool-loading test in the package.
//

// stubTool is a registry tool with a canned result that counts its runs and
// keeps the arguments it received.
type stubTool struct {
	def    llm.ToolDefinition
	result json.RawMessage
	calls  atomic.Int32

	mu       sync.Mutex
	received []json.RawMessage
}

func (t *stubTool) Definition() llm.ToolDefinition { return t.def }

func (t *stubTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	t.calls.Add(1)

	t.mu.Lock()
	t.received = append(t.received, append(json.RawMessage(nil), args...))
	t.mu.Unlock()

	if t.result != nil {
		return t.result, nil
	}

	return json.RawMessage(`{"ok":true}`), nil
}

func (t *stubTool) arguments() []json.RawMessage {
	t.mu.Lock()
	defer t.mu.Unlock()

	return append([]json.RawMessage(nil), t.received...)
}

type fixtureTool struct {
	name        string
	group       string
	description string
	params      string
	deferred    bool
	groupInfo   llm.ToolGroup // optional description/instructions for the group
}

// withGroupInfo stamps group metadata onto the fixture tools of the matching
// groups, the way an MCP client does for a server.
func withGroupInfo(tools []fixtureTool, groups ...llm.ToolGroup) []fixtureTool {
	out := append([]fixtureTool(nil), tools...)

	for i := range out {
		for _, group := range groups {
			if out[i].group == group.Name {
				out[i].groupInfo = group
			}
		}
	}

	return out
}

func newFixtureRegistry(tb testing.TB, tools ...fixtureTool) tool.Registry {
	tb.Helper()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	for _, spec := range tools {
		params := spec.params
		if params == "" {
			params = `{"type":"object"}`
		}

		var opts []tool.Option
		if spec.deferred {
			opts = append(opts, tool.WithDeferred())
		}

		if spec.group != "" {
			group := spec.groupInfo
			group.Name = spec.group
			opts = append(opts, tool.WithGroup(group))
		}

		require.NoError(tb, registry.Register(&stubTool{def: llm.ToolDefinition{
			Name: spec.name, Description: spec.description, Parameters: json.RawMessage(params),
		}}, opts...))
	}

	return registry
}

// serviceDeskTools is the shape this feature targets: a couple of always-on
// tools and several MCP servers' worth of deferred ones.
func serviceDeskTools() []fixtureTool {
	return []fixtureTool{
		{name: "todo_write", description: "Record the plan for a multi-step task."},
		{name: "servicenow__search_incidents", group: "servicenow", description: "Search ServiceNow incidents by text, state or assignment group."},
		{
			name: "servicenow__create_incident", group: "servicenow", deferred: true,
			description: "Open a ServiceNow incident on behalf of a caller. Requires the caller's sys_id.",
			params: `{"type":"object","properties":{
				"caller_sys_id":{"type":"string","description":"sys_id of the affected user"},
				"short_description":{"type":"string"},
				"urgency":{"type":"string","enum":["low","medium","high"]}},
				"required":["caller_sys_id","short_description"]}`,
		},
		{name: "servicenow__close_incident", group: "servicenow", deferred: true, description: "Close a ServiceNow incident with a resolution code and notes."},
		{name: "jira__create_issue", group: "jira", deferred: true, description: "Create a Jira issue in a project."},
		{name: "jira__search_issues", group: "jira", deferred: true, description: "Search Jira issues with JQL."},
		{name: "confluence__get_page", group: "confluence", deferred: true, description: "Fetch a Confluence page by id or title. Returns the page body as storage-format XHTML, which is verbose."},
		{name: "legacy_export", deferred: true, description: "Export the legacy report bundle."},
	}
}

func newFixtureLoader(tb testing.TB, tools []fixtureTool, configs ...ToolLoadingConfig) *toolLoader {
	tb.Helper()

	cfg := ToolLoadingConfig{}
	if len(configs) > 0 {
		cfg = configs[0]
	}

	return newToolLoader(newFixtureRegistry(tb, tools...), cfg)
}

func defNames(defs []llm.ToolDefinition) []string {
	names := make([]string, len(defs))
	for i, def := range defs {
		names[i] = def.Name
	}

	return names
}

// batchFor builds the batch executeTools would build for sess: the visible set
// is exactly what prepare sends.
func batchFor(loader *toolLoader, sess *session.State, schemaRoom int) *loadBatch {
	sent, _ := loader.prepare(loader.tools.List(), sess)

	return loader.newBatch(sent, schemaRoom)
}

func searchRequest(id, query string) *llm.ToolRequestPart {
	return &llm.ToolRequestPart{ID: id, Name: toolSearchName, Arguments: json.RawMessage(`{"query":` + strconv.Quote(query) + `}`)}
}

func decodeSearch(t *testing.T, resp *llm.ToolResponsePart) searchOutput {
	t.Helper()

	var out searchOutput
	require.NoError(t, json.Unmarshal(resp.Result, &out))

	return out
}

// runSearch resolves one search the way the agent does and commits its loads.
func runSearch(t *testing.T, loader *toolLoader, sess *session.State, query string) searchOutput {
	t.Helper()

	req := searchRequest("call-1", query)
	resp, loads, owned := loader.resolve(batchFor(loader, sess, unboundedSchemaRoom), req)
	require.True(t, owned)
	require.False(t, resp.IsError, string(resp.Result))
	loader.commit(sess, loads)

	return decodeSearch(t, resp)
}

// runAgent drives one invocation to completion and returns its events.
func runAgent(t *testing.T, ag *LLMAgent, sess *session.State) []agent.Event {
	t.Helper()

	events := make([]agent.Event, 0, 32)

	for evt, err := range ag.Run(t.Context(), agent.NewInvocationMetadata(sess, agent.Info{})) {
		require.NoError(t, err)

		events = append(events, evt)
	}

	return events
}

func finishReason(events []agent.Event) agent.FinishReason {
	for i := len(events) - 1; i >= 0; i-- {
		if end, ok := events[i].(agent.InvocationEndEvent); ok {
			return end.FinishReason
		}
	}

	return ""
}

func requestToolNames(req *llm.Request) []string {
	return defNames(req.Tools)
}

// toolResultsFor returns every result recorded for name, in order.
func toolResultsFor(msgs []llm.Message, name string) []string {
	var results []string

	for _, msg := range msgs {
		for _, resp := range msg.ToolResponses() {
			if resp.Name == name {
				results = append(results, string(resp.Result))
			}
		}
	}

	return results
}

func userMessage(text string) []llm.Message {
	return []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart(text))}
}

//
// prepare
//

func TestPrepare(t *testing.T) {
	t.Parallel()

	t.Run("scopes the tools array to the loaded set", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, serviceDeskTools())
		sess := &session.State{ID: "s"}

		defs, section := loader.prepare(loader.tools.List(), sess)
		assert.Equal(t, []string{"servicenow__search_incidents", "todo_write", "tool_search"}, defNames(defs),
			"always-on tools plus tool_search, name-sorted, before anything is loaded")
		assert.Contains(t, section, "## Additional tools")

		loader.commit(sess, []string{"jira__create_issue", "servicenow__create_incident"})

		defs, sectionAfter := loader.prepare(loader.tools.List(), sess)
		assert.Equal(t, []string{"jira__create_issue", "servicenow__create_incident", "servicenow__search_incidents", "todo_write", "tool_search"}, defNames(defs))
		assert.Equal(t, section, sectionAfter, "the catalog lists every deferred tool for the life of the registry")
	})

	t.Run("is a no-op with nothing deferred", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, []fixtureTool{{name: "alpha", description: "A"}, {name: "beta", description: "B"}})
		in := loader.tools.List()

		out, section := loader.prepare(in, &session.State{ID: "s"})
		assert.Equal(t, in, out)
		assert.Empty(t, section)
	})

	t.Run("reads a loaded set that went through a session store", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, serviceDeskTools())
		live := &session.State{ID: "s"}
		loader.commit(live, []string{"jira__search_issues", "confluence__get_page"})

		encoded, err := json.Marshal(live)
		require.NoError(t, err)

		var restored session.State
		require.NoError(t, json.Unmarshal(encoded, &restored))
		require.IsType(t, []any{}, restored.Metadata[loadedToolsMetadataKey], "the round trip must widen the type, or this proves nothing")

		before, _ := loader.prepare(loader.tools.List(), live)
		after, _ := loader.prepare(loader.tools.List(), &restored)
		assert.Equal(t, defNames(before), defNames(after))
	})
}

// TestLoadedSetPersistence: the set only grows, stays sorted and unique, and is
// stored as []any because the protobuf session store rejects []string.
func TestLoadedSetPersistence(t *testing.T) {
	t.Parallel()

	loader := newFixtureLoader(t, serviceDeskTools())
	sess := &session.State{ID: "s"}

	loader.commit(sess, []string{"jira__create_issue", "legacy_export"})
	loader.commit(sess, []string{"legacy_export", "confluence__get_page"})
	assert.Equal(t, []string{"confluence__get_page", "jira__create_issue", "legacy_export"}, loadedTools(sess))

	stored := sess.Metadata[loadedToolsMetadataKey]
	loader.commit(sess, []string{"jira__create_issue"})
	loader.commit(sess, nil)
	assert.Equal(t, stored, sess.Metadata[loadedToolsMetadataKey], "a no-op commit leaves the stored value alone")

	require.IsType(t, []any{}, stored)

	_, err := structpb.NewStruct(sess.Metadata)
	require.NoError(t, err, "session metadata must survive protobuf serialization")

	loader.commit(nil, []string{"x"}) // must not panic
}

//
// Catalog and instructions
//

// TestManifestGolden pins the exact prompt text; the model's behaviour depends
// on this wording, so a change should be a deliberate diff.
func TestManifestGolden(t *testing.T) {
	t.Parallel()

	defs := newFixtureRegistry(t, withGroupInfo(serviceDeskTools(),
		llm.ToolGroup{Name: "servicenow", Description: "ServiceNow incidents, changes and CMDB lookups"},
		llm.ToolGroup{Name: "jira", Description: "Jira issues and sprints"},
	)...).List()

	got := renderToolManifest(buildToolManifest(deferredOnly(defs)))

	golden := filepath.Join("testdata", "tool_manifest.golden")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		require.NoError(t, os.WriteFile(golden, []byte(got), 0o600))
	}

	want, err := os.ReadFile(golden)
	require.NoError(t, err, "run with UPDATE_GOLDEN=1 to create the golden file")
	assert.Equal(t, string(want), got)
}

// TestManifestDeterministic: the catalog sits in the cached prefix, so it must
// be byte-identical whatever order the tools were registered in.
func TestManifestDeterministic(t *testing.T) {
	t.Parallel()

	outputs := make(map[string]struct{})

	for range 100 {
		shuffled := serviceDeskTools()
		rand.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })

		defs := newFixtureRegistry(t, shuffled...).List()
		outputs[renderToolManifest(buildToolManifest(deferredOnly(defs)))] = struct{}{}
	}

	assert.Len(t, outputs, 1)
}

func deferredOnly(defs []llm.ToolDefinition) []llm.ToolDefinition {
	var out []llm.ToolDefinition

	for _, def := range defs {
		if def.Deferred {
			out = append(out, def)
		}
	}

	return out
}

func TestSummarize(t *testing.T) {
	t.Parallel()

	tests := map[string]string{
		"":                                  "",
		"  Fetch a page.   Returns XHTML. ": "Fetch a page",
		"Search issues with JQL":            "Search issues with JQL",
		"Open an incident! Then close it.":  "Open an incident",
	}

	for input, want := range tests {
		assert.Equal(t, want, summarize(input), "%q", input)
	}

	long := summarize(strings.Repeat("word ", 40))
	assert.True(t, strings.HasSuffix(long, "..."))
	assert.LessOrEqual(t, len(long), summaryLimit+3)

	// Truncation never splits a multi-byte rune.
	multibyte := summarize(strings.Repeat("é", 100))
	assert.Equal(t, multibyte, strings.ToValidUTF8(multibyte, "?"))
}

// TestInstructionsFollowVisibleTools: a group's instructions sit in the system
// prompt exactly while one of its tools is in the model's tool list.
func TestInstructionsFollowVisibleTools(t *testing.T) {
	t.Parallel()

	const (
		snowRule = "Resolve the caller's sys_id with search_users before opening an incident."
		jiraRule = "Always set a project key."
	)

	groups := []llm.ToolGroup{
		{Name: "servicenow", Instructions: snowRule},
		{Name: "jira", Instructions: jiraRule},
		{Name: "confluence", Description: "Confluence pages"},
	}

	instructionsOf := func(section string) string {
		_, after, _ := strings.Cut(section, "## Tool instructions")

		return after
	}

	tests := []struct {
		name        string
		tools       []fixtureTool
		loaded      any
		want, avoid []string
	}{
		{name: "always-on tool from the first request", tools: serviceDeskTools(), want: []string{snowRule}, avoid: []string{jiraRule}},
		{name: "deferred group once loaded", tools: serviceDeskTools(), loaded: []any{"jira__create_issue", "confluence__get_page"}, want: []string{jiraRule, snowRule}, avoid: []string{"### confluence"}},
		{name: "resumed session", tools: serviceDeskTools(), loaded: []any{"jira__search_issues"}, want: []string{jiraRule}},
		{name: "nothing deferred but an always-on group", tools: []fixtureTool{{name: "jira__search_issues", group: "jira", description: "Search."}}, want: []string{jiraRule}, avoid: []string{"## Additional tools"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			loader := newFixtureLoader(t, withGroupInfo(tt.tools, groups...))

			sess := &session.State{ID: "s"}
			if tt.loaded != nil {
				sess.Metadata = map[string]any{loadedToolsMetadataKey: tt.loaded}
			}

			_, section := loader.prepare(loader.tools.List(), sess)

			for _, want := range tt.want {
				assert.Contains(t, section, want)
			}

			for _, avoid := range tt.avoid {
				assert.NotContains(t, instructionsOf(section), avoid)
			}
		})
	}

	// Groups render sorted by name.
	loader := newFixtureLoader(t, withGroupInfo(serviceDeskTools(), groups...))
	sess := &session.State{ID: "s", Metadata: map[string]any{loadedToolsMetadataKey: []any{"jira__create_issue"}}}
	_, section := loader.prepare(loader.tools.List(), sess)
	assert.Less(t, strings.Index(section, jiraRule), strings.Index(section, snowRule))
}

func TestGroupMetadataDoesNotDependOnWhichMemberIsLoaded(t *testing.T) {
	t.Parallel()

	const instructions = "Confirm the tenant before every lookup."
	loader := newFixtureLoader(t, []fixtureTool{
		{name: "lookup", group: "svc"},
		{name: "create", group: "svc", deferred: true, groupInfo: llm.ToolGroup{
			Description: "Taxonomies", Instructions: instructions,
		}},
		{name: "update", group: "svc", deferred: true},
	})
	sess := &session.State{ID: "groups"}
	_, prompt := loader.prepare(loader.tools.List(), sess)
	assert.Contains(t, prompt, instructions, "an unloaded sibling can supply a visible group's instructions")

	out := runSearch(t, loader, sess, "taxonomies")
	assert.Equal(t, []string{"create", "update"}, out.Loaded, "group description is searchable on every member")

	// When no group member is visible, loading any member must pay for its instructions.
	loader = newFixtureLoader(t, []fixtureTool{
		{name: "create", group: "svc", deferred: true, groupInfo: llm.ToolGroup{
			Instructions: strings.Repeat("Confirm the tenant. ", 100),
		}},
		{name: "update", group: "svc", deferred: true},
	})
	resp, loads, _ := loader.resolve(batchFor(loader, &session.State{ID: "budget"}, 100), searchRequest("c", "select:update"))
	assert.Empty(t, loads)
	assert.Equal(t, []string{"update"}, decodeSearch(t, resp).TooLarge)
}

//
// Search
//

func TestSearch(t *testing.T) {
	t.Parallel()

	t.Run("select classifies names and returns a flat result", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, withGroupInfo(serviceDeskTools(),
			llm.ToolGroup{Name: "servicenow", Description: "ServiceNow incidents"},
			llm.ToolGroup{Name: "jira", Description: "Jira issues"},
		))
		sess := &session.State{ID: "s"}

		out := runSearch(t, loader, sess,
			"select:servicenow__close_incident,legacy_export,jira__create_issue,jira__create_issue,servicenow__search_incidents,nope")

		assert.Equal(t, []string{"jira__create_issue", "legacy_export", "servicenow__close_incident"},
			out.Loaded, "names are sorted and duplicates folded")
		assert.Equal(t, []string{"servicenow__search_incidents"}, out.AlreadyAvailable, "an always-on tool is already available")
		assert.Equal(t, []string{"nope"}, out.NotFound)
		assert.Contains(t, out.Note, "3 tool(s) loaded")
		assert.Contains(t, out.Note, "matched no tool")
		assert.ElementsMatch(t, out.Loaded, loadedTools(sess))

		again := runSearch(t, loader, sess, "select:jira__create_issue")
		assert.Empty(t, again.Loaded)
		assert.Equal(t, []string{"jira__create_issue"}, again.AlreadyAvailable)
		assert.Contains(t, again.Note, "already available")
	})

	t.Run("keyword query loads the best matches", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, serviceDeskTools())
		sess := &session.State{ID: "s"}
		loader.commit(sess, []string{"servicenow__close_incident"})

		out := runSearch(t, loader, sess, "incident")
		assert.Equal(t, []string{"servicenow__create_incident"}, out.Loaded)
		assert.Equal(t, []string{"servicenow__close_incident"}, out.AlreadyAvailable, "a visible hit is reported without taking a result slot")

		dead := runSearch(t, loader, sess, "kubernetes helm")
		assert.Empty(t, dead.Loaded)
		assert.Contains(t, dead.Note, "No tool matched")
	})

	t.Run("rejects bad input with an actionable error", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, serviceDeskTools())

		for _, arguments := range []string{`["query"]`, `{}`, `{"query":"   "}`} {
			req := &llm.ToolRequestPart{ID: "c", Name: toolSearchName, Arguments: json.RawMessage(arguments)}
			resp, loads, owned := loader.resolve(batchFor(loader, &session.State{ID: "s"}, unboundedSchemaRoom), req)

			require.True(t, owned)
			assert.True(t, resp.IsError, arguments)
			assert.Contains(t, string(resp.Result), "invalid_arguments")
			assert.Empty(t, loads)
		}
	})
}

// oversizedTools: one schema of ~8k estimated tokens, two of a few dozen.
func oversizedTools() []fixtureTool {
	huge := `{"type":"object","properties":{"blob":{"type":"string","description":"` + strings.Repeat("padding ", 3000) + `"}}}`

	return []fixtureTool{
		{name: "svc__huge", group: "svc", deferred: true, description: "One enormous tool.", params: huge},
		{name: "svc__small", group: "svc", deferred: true, description: "A small tool."},
		{name: "svc__other", group: "svc", deferred: true, description: "Another small tool."},
	}
}

// TestAdmission covers the two lines every load passes: the per-search token
// budget, which paces discovery and always admits a first tool, and the
// context-window room, which is absolute and shared by every call in a
// response, in request order.
func TestAdmission(t *testing.T) {
	t.Parallel()

	// Each fat schema is ~1,400 estimated tokens.
	fat := `{"type":"object","properties":{"blob":{"type":"string","description":"` + strings.Repeat("padding ", 520) + `"}}}`

	fatTools := make([]fixtureTool, 0, 6)
	fatNames := make([]string, 0, 6)

	for i := range 6 {
		name := fmt.Sprintf("svc__fat_%02d", i)
		fatNames = append(fatNames, name)
		fatTools = append(fatTools, fixtureTool{name: name, group: "svc", deferred: true, description: "A fat tool.", params: fat})
	}

	t.Run("token budget paces a search and reports the rest", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, fatTools, ToolLoadingConfig{MaxLoadTokens: 3000})
		sess := &session.State{ID: "s"}

		out := runSearch(t, loader, sess, "select:"+strings.Join(fatNames, ","))
		assert.Len(t, out.Loaded, 2)
		assert.Len(t, out.OverBudget, 4)
		assert.Contains(t, out.Note, "did not fit")

		again := runSearch(t, loader, sess, "select:"+strings.Join(out.OverBudget, ","))
		assert.Len(t, again.Loaded, 2, "a second search picks up where the first stopped")
		assert.Len(t, loadedTools(sess), 4)
	})

	t.Run("first tool is admitted regardless of the budget", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, oversizedTools(), ToolLoadingConfig{MaxLoadTokens: 500})
		out := runSearch(t, loader, &session.State{ID: "s"}, "select:svc__huge")
		assert.Equal(t, []string{"svc__huge"}, out.Loaded)
	})

	t.Run("explicit selection preserves priority and ignores duplicates", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, fatTools, ToolLoadingConfig{MaxLoadTokens: 1500})
		out := runSearch(t, loader, &session.State{ID: "priority"},
			"select:svc__fat_05,svc__fat_05,svc__fat_00,svc__fat_01")
		assert.Equal(t, []string{"svc__fat_05"}, out.Loaded)
		assert.Equal(t, []string{"svc__fat_00", "svc__fat_01"}, out.OverBudget)
	})

	t.Run("context room refuses what the window cannot hold", func(t *testing.T) {
		t.Parallel()

		loader := newFixtureLoader(t, oversizedTools())
		req := searchRequest("c1", "select:svc__huge,svc__small")

		resp, loads, _ := loader.resolve(batchFor(loader, &session.State{ID: "s"}, 200), req)
		out := decodeSearch(t, resp)
		assert.Equal(t, []string{"svc__small"}, loads)
		assert.Equal(t, []string{"svc__huge"}, out.TooLarge)
		assert.Empty(t, out.OverBudget, "a refused tool consumes no budget")
		assert.Contains(t, out.Note, "cannot be loaded")
		assert.Contains(t, out.Note, "new session")

		// Instructions of a newly visible group count toward the room.
		loader = newFixtureLoader(t, withGroupInfo(oversizedTools(), llm.ToolGroup{Name: "svc", Instructions: strings.Repeat("Always confirm first. ", 200)}))
		resp, _, _ = loader.resolve(batchFor(loader, &session.State{ID: "s"}, 200), searchRequest("c2", "select:svc__small"))
		assert.Equal(t, []string{"svc__small"}, decodeSearch(t, resp).TooLarge)

		// An unknown window disables the check.
		resp, _, _ = loader.resolve(batchFor(loader, &session.State{ID: "s"}, unboundedSchemaRoom), searchRequest("c3", "select:svc__huge"))
		assert.Equal(t, []string{"svc__huge"}, decodeSearch(t, resp).Loaded)
	})

	t.Run("room is shared across the response in request order", func(t *testing.T) {
		t.Parallel()

		const room = 20 // one small schema (~15 tokens), not two

		loader := newFixtureLoader(t, oversizedTools())
		first := searchRequest("c1", "select:svc__small")
		second := searchRequest("c2", "select:svc__other")

		batch := batchFor(loader, &session.State{ID: "s"}, room)
		firstResp, _, _ := loader.resolve(batch, first)
		secondResp, secondLoads, _ := loader.resolve(batch, second)

		assert.Equal(t, []string{"svc__small"}, decodeSearch(t, firstResp).Loaded)
		assert.Equal(t, []string{"svc__other"}, decodeSearch(t, secondResp).TooLarge)
		assert.Empty(t, secondLoads)

		// A repeated name is charged once.
		batch = batchFor(loader, &session.State{ID: "s"}, room)
		loader.resolve(batch, first)
		resp, _, _ := loader.resolve(batch, searchRequest("c3", "select:svc__small"))
		assert.Equal(t, []string{"svc__small"}, decodeSearch(t, resp).Loaded)

		// The self-heal path is admitted like a search.
		batch = batchFor(loader, &session.State{ID: "s"}, room)
		resp, loads, owned := loader.resolve(batch, &llm.ToolRequestPart{ID: "h", Name: "svc__huge", Arguments: json.RawMessage(`{}`)})
		require.True(t, owned)
		assert.Contains(t, string(resp.Result), "tool_too_large")
		assert.Empty(t, loads)

		resp, loads, _ = loader.resolve(batch, &llm.ToolRequestPart{ID: "s", Name: "svc__small", Arguments: json.RawMessage(`{}`)})
		assert.Contains(t, string(resp.Result), "tool_not_loaded")
		assert.Equal(t, []string{"svc__small"}, loads)
	})
}

//
// resolve: which calls the loader owns
//

// TestResolve: tool_search and blind calls to deferred tools the model could
// not see are the loader's; everything else reaches the registry. Eligibility
// comes from the request snapshot, so a response that loads a tool and calls it
// in the same breath gets a recovery error whichever call runs first.
func TestResolve(t *testing.T) {
	t.Parallel()

	const name = "servicenow__create_incident"

	loader := newFixtureLoader(t, serviceDeskTools())
	sess := &session.State{ID: "s"}
	batch := batchFor(loader, sess, unboundedSchemaRoom)

	blind := &llm.ToolRequestPart{ID: "c1", Name: name, Arguments: json.RawMessage(`{"guessed":"argument"}`)}

	for _, req := range []*llm.ToolRequestPart{searchRequest("s1", "select:"+name), blind} {
		resp, loads, owned := loader.resolve(batch, req)
		require.True(t, owned, req.Name)
		assert.Equal(t, []string{name}, loads, "both propose the load")

		if req == blind {
			assert.Contains(t, string(resp.Result), "tool_not_loaded")
		}

		loader.commit(sess, loads)
	}

	assert.Equal(t, []string{name}, loadedTools(sess))

	// Against a batch that saw the schema it is an ordinary call, as are
	// always-on tools and names the model invented.
	batch = batchFor(loader, sess, unboundedSchemaRoom)

	for _, req := range []*llm.ToolRequestPart{blind, {ID: "c", Name: "todo_write"}, {ID: "c", Name: "servicenow__search_incidents"}, {ID: "c", Name: "does_not_exist"}} {
		_, _, owned := loader.resolve(batch, req)
		assert.False(t, owned, "%s must reach the registry", req.Name)
	}

	_, _, owned := loader.resolve(batch, nil)
	assert.False(t, owned)
}

// TestToolSearchNameIsReserved: a registry tool of the same name would be
// declared twice and unreachable, so construction fails with a clear message.
func TestToolSearchNameIsReserved(t *testing.T) {
	t.Parallel()

	registry := newFixtureRegistry(t,
		fixtureTool{name: toolSearchName, description: "an operator's own tool of the same name"},
		fixtureTool{name: "svc__thing", group: "svc", deferred: true, description: "A thing."},
	)

	_, err := New("agent", "prompt", fakellm.NewFakeModel(), WithTools(registry))
	require.ErrorContains(t, err, toolSearchName)

	_, err = New("agent", "prompt", fakellm.NewFakeModel(), WithToolLoadingConfig(ToolLoadingConfig{MaxLoadTokens: -1}))
	require.ErrorContains(t, err, "MaxLoadTokens")

	// prepare never double-declares even if a registry gains the name later.
	defs, _ := newToolLoader(registry, ToolLoadingConfig{}).prepare(registry.List(), &session.State{ID: "s"})
	assert.Equal(t, 1, strings.Count(strings.Join(defNames(defs), ","), toolSearchName))
}

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
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolsearch"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	toolSearchName = "tool_search"
	// Session metadata survives compaction and uses the session store's persistence.
	loadedToolsMetadataKey = "ai-sdk-go/loaded_tools"
	defaultMaxLoadTokens   = 4000
	defaultMaxResults      = 5
	// An unknown model context window disables schema admission checks.
	unboundedSchemaRoom = -1
)

// ToolLoadingConfig tunes lazy tool loading. Zero values select defaults.
type ToolLoadingConfig struct {
	// ForceLocal uses the local tool_search tool even when the model supports
	// native hosted search. By default, supported models use native search.
	ForceLocal bool

	// MaxLoadTokens limits the estimated schema tokens one local search may load.
	// Native hosted search controls its own selection and ignores this limit.
	// The first tool of a search is always admitted; every load must also fit
	// the model's context window. Default 4000.
	MaxLoadTokens int
}

// toolLoader holds immutable policy. Only the turn goroutine writes session state.
type toolLoader struct {
	tools         tool.Registry
	maxLoadTokens int
}

func newToolLoader(tools tool.Registry, cfg ToolLoadingConfig) *toolLoader {
	return &toolLoader{
		tools:         tools,
		maxLoadTokens: cmp.Or(cfg.MaxLoadTokens, defaultMaxLoadTokens),
	}
}

// prepare returns visible schemas and the generated system prompt. The manifest
// contains every deferred tool so it stays stable as tools are loaded.
func (l *toolLoader) prepare(defs []llm.ToolDefinition, sess *session.State) ([]llm.ToolDefinition, string) {
	deferred := make([]llm.ToolDefinition, 0, len(defs))

	for _, def := range defs {
		if def.Deferred {
			deferred = append(deferred, def)
		}
	}

	loaded := loadedToolSet(sess)

	scoped := make([]llm.ToolDefinition, 0, len(defs)+1)

	for _, def := range defs {
		if def.Deferred && !loaded[def.Name] {
			continue
		}

		scoped = append(scoped, def)
	}

	var sections []string

	if len(deferred) > 0 {
		// tool_search is a reserved name (config.validate rejects a registry that
		// already holds it). Guard anyway: a registry is mutable, and declaring the
		// same tool name twice is a request the provider rejects outright.
		if !slices.ContainsFunc(scoped, func(def llm.ToolDefinition) bool { return def.Name == toolSearchName }) {
			scoped = append(scoped, toolSearchDefinition())
		}

		sections = append(sections, renderToolManifest(buildToolManifest(deferred)))
	}

	if instructions := renderGroupInstructions(scoped); instructions != "" {
		sections = append(sections, instructions)
	}

	if len(sections) == 0 {
		return defs, ""
	}

	// Re-sort for the same reason tool.Registry.List does: providers serialize
	// req.Tools verbatim into the request prefix, so the order has to be a
	// deterministic function of the set, not of how it was assembled.
	slices.SortFunc(scoped, func(a, b llm.ToolDefinition) int {
		return cmp.Compare(a.Name, b.Name)
	})

	return scoped, strings.Join(sections, "\n\n")
}

// loadBatch tracks the tools the model saw in one response and the context
// room its loads share. Loader-owned calls resolve sequentially, so no lock.
type loadBatch struct {
	visible       map[string]bool
	byName        map[string]llm.ToolDefinition
	schemaRoom    int
	spentRoom     int
	admitted      map[string]bool
	charged       map[string]bool
	searchEnabled bool
}

func (l *toolLoader) newBatch(sent []llm.ToolDefinition, schemaRoom int) *loadBatch {
	all := l.tools.List()

	b := &loadBatch{
		visible:    make(map[string]bool, len(sent)),
		byName:     make(map[string]llm.ToolDefinition, len(all)),
		schemaRoom: schemaRoom,
		admitted:   make(map[string]bool),
		charged:    make(map[string]bool),
	}
	for _, def := range all {
		b.byName[def.Name] = def
	}

	for _, def := range sent {
		b.visible[def.Name] = true

		if def.Group.Name != "" {
			b.charged[def.Group.Name] = true
		}
	}

	_, registeredSearch := b.byName[toolSearchName]
	b.searchEnabled = b.visible[toolSearchName] && !registeredSearch

	return b
}

func (b *loadBatch) owns(req *llm.ToolRequestPart) bool {
	if b == nil || req == nil {
		return false
	}

	return (req.Name == toolSearchName && b.searchEnabled) || (b.byName[req.Name].Deferred && !b.visible[req.Name])
}

// deferred returns the registered deferred tools in name order.
func (b *loadBatch) deferred() []llm.ToolDefinition {
	defs := make([]llm.ToolDefinition, 0, len(b.byName))

	for _, def := range b.byName {
		if def.Deferred {
			defs = append(defs, def)
		}
	}

	slices.SortFunc(defs, func(x, y llm.ToolDefinition) int {
		return cmp.Compare(x.Name, y.Name)
	})

	return defs
}

// classify sorts selected names, deduplicated, into loadable candidates, names
// the model can already call, and names that match no tool.
func (b *loadBatch) classify(selected []string) ([]string, []string, []string) {
	var candidates, available, notFound []string

	seen := make(map[string]bool, len(selected))

	for _, name := range selected {
		if seen[name] {
			continue
		}

		seen[name] = true
		def, exists := b.byName[name]

		switch {
		case !exists:
			notFound = append(notFound, name)
		case b.visible[name] || !def.Deferred:
			available = append(available, name)
		default:
			candidates = append(candidates, name)
		}
	}

	return candidates, available, notFound
}

// resolve answers a loader-owned call as the interceptors left it and returns
// the loads that answer caused. The caller commits them once the result is
// recorded.
func (l *toolLoader) resolve(b *loadBatch, req *llm.ToolRequestPart) (*llm.ToolResponsePart, []string, bool) {
	if !b.owns(req) {
		return nil, nil, false
	}

	if req.Name == toolSearchName {
		resp, loads := l.search(b, req)

		return resp, loads, true
	}

	// A blind call loads the schema for a retry without executing guessed arguments.
	fits, _, tooLarge := l.applyLoadBudget([]string{req.Name}, b)
	if len(tooLarge) > 0 {
		return tooLargeResponse(req), nil, true
	}

	return recoveryResponse(req), fits, true
}

// searchInput is the model-facing tool_search schema.
type searchInput struct {
	Query      string `json:"query"`
	MaxResults int    `json:"max_results,omitempty"`
}

// searchOutput reports names only; schemas arrive in the next request.
type searchOutput struct {
	Loaded           []string `json:"loaded,omitempty"`
	AlreadyAvailable []string `json:"already_available,omitempty"`
	NotFound         []string `json:"not_found,omitempty"`
	OverBudget       []string `json:"over_budget,omitempty"`
	TooLarge         []string `json:"too_large,omitempty"`
	Note             string   `json:"note"`
}

// search returns the response and loads caused by one tool_search execution.
func (l *toolLoader) search(b *loadBatch, req *llm.ToolRequestPart) (*llm.ToolResponsePart, []string) {
	var input searchInput

	if len(req.Arguments) > 0 {
		err := json.Unmarshal(req.Arguments, &input)
		if err != nil {
			return errorResponse(req, "invalid_arguments",
				fmt.Sprintf("could not parse tool_search arguments: %v", err)), nil
		}
	}

	query := strings.TrimSpace(input.Query)
	if query == "" {
		return errorResponse(req, "invalid_arguments",
			`query is required, for example {"query": "select:some_tool,other_tool"}`), nil
	}

	deferred := b.deferred()
	parsed := toolsearch.Parse(query)

	var (
		out        searchOutput
		candidates []string
	)

	// Classify before budgeting: a name the model can already see is answered,
	// not loaded, and must not consume budget.
	if parsed.Selected != nil {
		candidates, out.AlreadyAvailable, out.NotFound = b.classify(parsed.Selected)
	} else {
		limit := input.MaxResults
		if limit <= 0 {
			limit = defaultMaxResults
		}

		// Visible hits are reported without using a result slot, so the
		// model still learns about the next-best tools it has not loaded.
		for _, name := range toolsearch.Rank(deferred, parsed, 0) {
			if b.visible[name] {
				out.AlreadyAvailable = append(out.AlreadyAvailable, name)
				continue
			}

			if len(candidates) < limit {
				candidates = append(candidates, name)
			}
		}
	}

	loads, overBudget, tooLarge := l.applyLoadBudget(candidates, b)
	sort.Strings(loads)
	sort.Strings(out.AlreadyAvailable)

	out.Loaded = loads
	out.OverBudget = overBudget
	out.TooLarge = tooLarge
	out.Note = searchNote(&out, parsed, len(deferred))

	payload, err := json.Marshal(out)
	if err != nil {
		return errorResponse(req, "internal", "could not encode tool_search result"), nil
	}

	return &llm.ToolResponsePart{ID: req.ID, Name: req.Name, Result: payload}, loads
}

// searchNote describes partial successes and refusals.
func searchNote(out *searchOutput, parsed toolsearch.Query, deferredCount int) string {
	var parts []string

	if n := len(out.Loaded); n > 0 {
		parts = append(parts, fmt.Sprintf(
			"%d tool(s) loaded. Their schemas are available now - call them directly, no further search needed.", n))
	}

	if n := len(out.AlreadyAvailable); n > 0 {
		parts = append(parts, fmt.Sprintf(
			"%d tool(s) named are already available - call them directly.", n))
	}

	if n := len(out.OverBudget); n > 0 {
		parts = append(parts, fmt.Sprintf(
			"%d did not fit this call's load budget - search again for those.", n))
	}

	if n := len(out.TooLarge); n > 0 {
		parts = append(parts, fmt.Sprintf(
			"%d cannot be loaded: the schema does not fit the context room left after this session's loaded tools. "+
				"Use another tool; a new session, with nothing loaded yet, may have room.", n))
	}

	if n := len(out.NotFound); n > 0 && len(parts) > 0 {
		parts = append(parts, fmt.Sprintf(
			"%d name(s) matched no tool - copy names from the manifest exactly.", n))
	}

	if len(parts) > 0 {
		return strings.Join(parts, " ")
	}

	if parsed.Selected != nil {
		return fmt.Sprintf(
			"No tool matched those exact names. %d tools are deferred; copy a name from the catalog exactly, or retry with a keyword query.",
			deferredCount)
	}

	return fmt.Sprintf(
		"No tool matched. %d tools are deferred; retry with different keywords, or with select: and an exact name from the catalog.",
		deferredCount)
}

// applyLoadBudget admits candidates against the batch's remaining context
// room. The first tool of a search is always admitted; already-loaded tools
// are free, and group instructions are charged once.
func (l *toolLoader) applyLoadBudget(candidates []string, b *loadBatch) ([]string, []string, []string) {
	fits := make([]string, 0, len(candidates))

	var overBudget, tooLarge []string

	spentBudget := 0
	newLoads := 0

	for _, name := range candidates {
		if b.admitted[name] {
			fits = append(fits, name)
			continue
		}

		def := b.byName[name]
		cost := estimateToolTokens([]llm.ToolDefinition{def})

		if def.Group.Name != "" && !b.charged[def.Group.Name] {
			cost += estimateTextTokens(strings.TrimSpace(def.Group.Instructions))
		}

		if b.schemaRoom != unboundedSchemaRoom && b.spentRoom+cost > b.schemaRoom {
			tooLarge = append(tooLarge, name)
			continue
		}

		if newLoads > 0 && spentBudget+cost > l.maxLoadTokens {
			overBudget = append(overBudget, name)
			continue
		}

		spentBudget += cost
		newLoads++
		b.spentRoom += cost
		b.admitted[name] = true

		if def.Group.Name != "" {
			b.charged[def.Group.Name] = true
		}

		fits = append(fits, name)
	}

	return fits, overBudget, tooLarge
}

// recoveryResponse is the reply to a call on a deferred tool the model could
// not see: an error the model can act on.
func recoveryResponse(req *llm.ToolRequestPart) *llm.ToolResponsePart {
	return errorResponse(req, "tool_not_loaded",
		req.Name+" was not loaded when you called it, so its schema was unavailable. It is loaded now "+
			"and its schema is in your tool list - call it again with arguments matching that schema.")
}

// tooLargeResponse is the reply to a blind call on a deferred tool whose schema
// no longer fits the context room this session has left.
func tooLargeResponse(req *llm.ToolRequestPart) *llm.ToolResponsePart {
	return errorResponse(req, "tool_too_large",
		req.Name+" cannot be loaded: its schema does not fit the context room left after this session's loaded tools. "+
			"Use another tool; a new session, with nothing loaded yet, may have room.")
}

func errorResponse(req *llm.ToolRequestPart, code, message string) *llm.ToolResponsePart {
	payload, err := json.Marshal(map[string]string{"error": code, "message": message})
	if err != nil {
		payload = []byte(`{"error":"internal"}`)
	}

	return &llm.ToolResponsePart{ID: req.ID, Name: req.Name, Result: payload, IsError: true}
}

// toolSearchDefinition is the tool_search definition offered to the model. It
// is a static value, so it never varies the request prefix.
func toolSearchDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name: toolSearchName,
		Description: "Load the input schemas of tools listed in the \"Additional tools\" section of " +
			"your instructions, so you can call them. Those tools are known to you by name and " +
			"summary only; until you load one, calling it fails.\n\n" +
			"Load everything you expect to need in ONE call by listing the names: " +
			`{"query": "select:name_a,name_b"}. Loading one tool per call wastes a round trip.` +
			"\n\nQuery forms:\n" +
			"- \"select:name_a,name_b\" - load these exact tools by name. Prefer this whenever you " +
			"can read the names off the manifest.\n" +
			"- \"incident priority\" - keyword search over tool names, summaries and argument " +
			"names, best matches first (max_results, default 5).\n" +
			"- \"+servicenow incident\" - require \"servicenow\" in the tool name, rank by the " +
			"remaining words.\n\n" +
			"The result names the tools that became available; their schemas are then in your tool " +
			"list, so call them directly rather than searching again. A tool becomes callable only " +
			"once this call's result is returned to you - do not call it in the same response as " +
			"the search that loads it.",
		Parameters: json.RawMessage(`{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "\"select:name_a,name_b\" to load exact tools by name, or keywords to search. Prefix a word with + to require it in the tool name."
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum matches to load for a keyword query. Default 5. Ignored by select:.",
      "minimum": 1
    }
  },
  "required": ["query"],
  "additionalProperties": false
}`),
		Type: llm.ToolTypeFunction,
	}
}

//
// Session state
//

// commit adds names to the session on the turn goroutine. Loaded tools stay
// available across turns, compaction, and restarts.
func (*toolLoader) commit(sess *session.State, names []string) {
	if sess == nil || len(names) == 0 {
		return
	}

	current := loadedToolSet(sess)
	before := len(current)

	for _, name := range names {
		current[name] = true
	}

	if len(current) == before {
		return
	}

	union := make([]string, 0, len(current))
	for name := range current {
		union = append(union, name)
	}

	sort.Strings(union)
	setLoadedTools(sess, union)
}

// loadedToolSet returns the session's loaded names as a set.
func loadedToolSet(sess *session.State) map[string]bool {
	names := loadedTools(sess)

	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}

	return set
}

func loadedTools(sess *session.State) []string {
	if sess == nil {
		return nil
	}

	switch stored := sess.Metadata[loadedToolsMetadataKey].(type) {
	case []string:
		return stored
	case []any:
		names := make([]string, 0, len(stored))

		for _, entry := range stored {
			name, ok := entry.(string)
			if ok && name != "" {
				names = append(names, name)
			}
		}

		return names
	default:
		return nil
	}
}

// setLoadedTools uses []any because the protobuf session store cannot encode []string.
func setLoadedTools(sess *session.State, names []string) {
	if sess == nil {
		return
	}

	if sess.Metadata == nil {
		sess.Metadata = make(map[string]any, 1)
	}

	stored := make([]any, len(names))
	for i, name := range names {
		stored[i] = name
	}

	sess.Metadata[loadedToolsMetadataKey] = stored
}

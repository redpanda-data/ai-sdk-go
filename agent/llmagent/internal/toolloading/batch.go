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

package toolloading

import (
	"cmp"
	"encoding/json"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/tokens"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/toolsearch"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Batch tracks the tools the model saw in one response and the context
// room its loads share. Loader-owned calls resolve sequentially, so no lock.
type Batch struct {
	visible       map[string]bool
	byName        map[string]llm.ToolDefinition
	schemaRoom    int
	spentRoom     int
	admitted      map[string]bool
	charged       map[string]bool
	searchEnabled bool
}

// NewBatch snapshots the tools the model saw for one response. It returns nil
// on a nil loader; Owns and Resolve treat a nil batch as owning nothing.
func (l *Loader) NewBatch(sent []llm.ToolDefinition, schemaRoom int) *Batch {
	if l == nil {
		return nil
	}

	all := l.tools.List()

	b := &Batch{
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

	_, registeredSearch := b.byName[SearchToolName]
	b.searchEnabled = b.visible[SearchToolName] && !registeredSearch

	return b
}

// Owns reports whether the loader answers this call: a tool_search call, or
// a call on a deferred tool the model could not see.
func (b *Batch) Owns(req *llm.ToolRequestPart) bool {
	if b == nil || req == nil {
		return false
	}

	return (req.Name == SearchToolName && b.searchEnabled) || (b.byName[req.Name].Deferred && !b.visible[req.Name])
}

// deferred returns the registered deferred tools in name order.
func (b *Batch) deferred() []llm.ToolDefinition {
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
func (b *Batch) classify(selected []string) ([]string, []string, []string) {
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

// Resolve answers a loader-owned call as the interceptors left it and returns
// the loads that answer caused. The caller commits them once the result is
// recorded.
func (l *Loader) Resolve(b *Batch, req *llm.ToolRequestPart) (*llm.ToolResponsePart, []string, bool) {
	if !b.Owns(req) {
		return nil, nil, false
	}

	if req.Name == SearchToolName {
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
func (l *Loader) search(b *Batch, req *llm.ToolRequestPart) (*llm.ToolResponsePart, []string) {
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
func (l *Loader) applyLoadBudget(candidates []string, b *Batch) ([]string, []string, []string) {
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
		cost := tokens.Tools([]llm.ToolDefinition{def})

		if def.Group.Name != "" && !b.charged[def.Group.Name] {
			cost += tokens.Text(strings.TrimSpace(def.Group.Instructions))
		}

		if b.schemaRoom != UnboundedSchemaRoom && b.spentRoom+cost > b.schemaRoom {
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
		Name: SearchToolName,
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

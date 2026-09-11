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

// Package toolloading implements lazy tool loading for the LLM agent: which
// schemas a request carries, how the model discovers the rest, and what the
// session remembers about it.
package toolloading

import (
	"cmp"
	"slices"
	"sort"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

const (
	// SearchToolName is the local discovery tool offered to the model. The
	// agent reserves it: a registry tool of that name cannot coexist with
	// deferred tools.
	SearchToolName = "tool_search"

	// LoadedToolsMetadataKey holds the session's loaded tool names. Session
	// metadata survives compaction and uses the session store's persistence.
	LoadedToolsMetadataKey = "ai-sdk-go/loaded_tools"

	// UnboundedSchemaRoom disables schema admission checks when the model's
	// context window is unknown.
	UnboundedSchemaRoom = -1

	defaultMaxLoadTokens = 4000
	defaultMaxResults    = 5
)

// Config tunes lazy tool loading. Zero values select defaults.
type Config struct {
	// ForceLocal uses the local tool_search tool even when the model supports
	// native hosted search.
	ForceLocal bool

	// MaxLoadTokens limits the estimated schema tokens one local search may
	// load. The first tool of a search is always admitted; every load must
	// also fit the model's context window. Default 4000.
	MaxLoadTokens int
}

// Loader decides which tool schemas a request carries, how the model
// discovers the rest, and what the session remembers about it. It holds
// immutable policy; only the turn goroutine writes session state.
//
// Two discovery modes exist. Local mode withholds deferred schemas, lists them
// by name and summary in the system prompt, and answers a tool_search call by
// loading schemas into the next request. Native mode sends the complete
// catalog with deferral flags and lets the provider search it inside the
// response; the tools array and system prompt then stay stable across
// discovery.
//
// Every method is safe on a nil *Loader and behaves as if nothing were
// deferred, so an agent without a registry needs no special casing.
type Loader struct {
	tools         tool.Registry
	model         llm.Model
	forceLocal    bool
	maxLoadTokens int
}

// New builds a loader over the registry for the given model.
func New(tools tool.Registry, model llm.Model, cfg Config) *Loader {
	return &Loader{
		tools:         tools,
		model:         model,
		forceLocal:    cfg.ForceLocal,
		maxLoadTokens: cmp.Or(cfg.MaxLoadTokens, defaultMaxLoadTokens),
	}
}

// Plan is what one request sends for tools: the definitions, the system
// prompt section that describes discovery, and the discovery mode.
type Plan struct {
	Tools  []llm.ToolDefinition
	Prompt string
	Native bool
}

// Prepare plans the tools for one request from the registry definitions and
// the session's loaded set. In native mode Prepare is idempotent on its own
// output, so a caller may re-plan after compaction changed the history.
func (l *Loader) Prepare(defs []llm.ToolDefinition, sess *session.State) Plan {
	if l == nil {
		return Plan{Tools: defs}
	}

	if l.native() {
		tools, prompt := l.prepareNative(defs, sess)

		return Plan{Tools: tools, Prompt: prompt, Native: true}
	}

	tools, prompt := l.prepareLocal(defs, sess)

	return Plan{Tools: tools, Prompt: prompt}
}

//
// Session state
//

// Commit adds names to the session on the turn goroutine. Loaded tools stay
// available across turns, compaction, and restarts.
func (*Loader) Commit(sess *session.State, names []string) {
	if sess == nil || len(names) == 0 {
		return
	}

	current := LoadedToolSet(sess)
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

// LoadedToolSet returns the session's loaded names as a set.
func LoadedToolSet(sess *session.State) map[string]bool {
	names := LoadedTools(sess)

	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}

	return set
}

// LoadedTools returns the session's loaded names in stored order.
func LoadedTools(sess *session.State) []string {
	if sess == nil {
		return nil
	}

	switch stored := sess.Metadata[LoadedToolsMetadataKey].(type) {
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

	sess.Metadata[LoadedToolsMetadataKey] = stored
}

// prepareLocal returns visible schemas and the generated system prompt. The manifest
// contains every deferred tool so it stays stable as tools are loaded.
func (l *Loader) prepareLocal(defs []llm.ToolDefinition, sess *session.State) ([]llm.ToolDefinition, string) {
	deferred := make([]llm.ToolDefinition, 0, len(defs))

	for _, def := range defs {
		if def.Deferred {
			deferred = append(deferred, def)
		}
	}

	loaded := LoadedToolSet(sess)

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
		if !slices.ContainsFunc(scoped, func(def llm.ToolDefinition) bool { return def.Name == SearchToolName }) {
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

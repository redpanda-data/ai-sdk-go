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
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/tokens"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

// Providers whose hosted search protocol the agent speaks. Compatible
// endpoints and Bedrock Converse do not, whatever capabilities they claim.
const (
	nativeOpenAI    = "openai"
	nativeAnthropic = "anthropic"
)

// native reports whether this request uses provider-hosted discovery.
func (l *Loader) native() bool {
	if l.forceLocal || !l.model.Capabilities().ToolSearch {
		return false
	}

	switch l.model.Provider() {
	case nativeOpenAI, nativeAnthropic:
		return slices.ContainsFunc(l.tools.List(), func(d llm.ToolDefinition) bool { return d.Deferred })
	default:
		return false
	}
}

// prepareNative returns the complete catalog with deferral flags and a stable
// system prompt section.
func (l *Loader) prepareNative(defs []llm.ToolDefinition, sess *session.State) ([]llm.ToolDefinition, string) {
	// References carry discovery forward in the native protocol. When compaction
	// removes them (or the provider changes), restore those tools as eager. This
	// changes the prefix once, at compaction, rather than on every discovery.
	referenced := NativeLoadedTools(sess.Messages, l.model.Provider())
	loaded := LoadedToolSet(sess)

	defs = slices.Clone(defs)
	for i := range defs {
		if loaded[defs[i].Name] && !referenced[defs[i].Name] {
			defs[i].Deferred = false
		}
	}
	// Deferred tools are invisible to the model until discovered, so a group
	// directory tells it what a search can find. It is built from the registry,
	// not the restored set, so it stays stable for the life of the registry.
	var sections []string

	if directory := renderGroupDirectory(l.tools.List()); directory != "" {
		sections = append(sections, directory)
	}

	if instructions := renderGroupInstructions(defs); instructions != "" {
		sections = append(sections, instructions)
	}

	return defs, strings.Join(sections, "\n\n")
}

// NativeLoadedTools returns the tools that hosted search results still in the
// history have loaded for the given provider.
func NativeLoadedTools(messages []llm.Message, provider string) map[string]bool {
	loaded := make(map[string]bool)

	for _, msg := range messages {
		for _, part := range msg.Content {
			if search, ok := part.(*llm.ToolSearchPart); ok && search != nil && search.Provider == provider {
				for _, name := range search.Tools {
					loaded[name] = true
				}
			}
		}
	}

	return loaded
}

// Visible returns the tools the model can call from a sent set: in native
// mode, deferred tools it has not loaded are excluded.
func (*Loader) Visible(defs []llm.ToolDefinition, sess *session.State, native bool) []llm.ToolDefinition {
	if !native {
		return defs
	}

	loaded := LoadedToolSet(sess)

	visible := make([]llm.ToolDefinition, 0, len(defs))
	for _, def := range defs {
		if !def.Deferred || loaded[def.Name] {
			visible = append(visible, def)
		}
	}

	return visible
}

// Tokens estimates the tool-definition cost of a request with the given
// tools. In native mode only visible schemas count in full; the search tool
// and the names and descriptions the provider exposes are added on top.
func (l *Loader) Tokens(defs []llm.ToolDefinition, sess *session.State, native bool) int {
	total := tokens.Tools(l.Visible(defs, sess, native))
	if !native {
		return total
	}
	// Native search has a small schema. OpenAI also exposes names and descriptions
	// of standalone deferred functions before discovery.
	total += 200
	loaded := LoadedToolSet(sess)

	if l.model.Provider() == nativeOpenAI {
		groups := make(map[string]bool)
		for _, def := range defs {
			if def.Group.Name != "" && !groups[def.Group.Name] {
				groups[def.Group.Name] = true
				total += tokens.Text(def.Group.Name) + tokens.Text(def.Group.Description)
			}

			if def.Deferred && !loaded[def.Name] && def.Group.Name == "" {
				total += tokens.Text(def.Name) + tokens.Text(def.Description)
			}
		}
	}

	return total
}

// Record commits the tools a response's hosted search results loaded. Native
// discovery may be followed by a real tool call in the same response, so the
// agent records before the message is persisted.
func (l *Loader) Record(sess *session.State, message llm.Message) {
	if l == nil {
		return
	}

	for _, part := range message.Content {
		if search, ok := part.(*llm.ToolSearchPart); ok && search != nil && search.Provider == l.model.Provider() {
			l.Commit(sess, search.Tools)
		}
	}
}

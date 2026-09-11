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
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent/internal/tokens"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

const (
	nativeOpenAI    = "openai"
	nativeAnthropic = "anthropic"
)

func (a *LLMAgent) nativeToolSearch(defs []llm.ToolDefinition) bool {
	if a.loader == nil || a.config.toolLoading.ForceLocal || !a.config.model.Capabilities().ToolSearch {
		return false
	}
	// Compatible endpoints and Bedrock Converse do not speak these protocols.
	switch a.config.model.Provider() {
	case nativeOpenAI, nativeAnthropic:
		return slices.ContainsFunc(defs, func(d llm.ToolDefinition) bool { return d.Deferred })
	default:
		return false
	}
}

func (a *LLMAgent) prepareTools(defs []llm.ToolDefinition, sess *session.State, native bool) ([]llm.ToolDefinition, string) {
	if !native {
		return a.loader.prepare(defs, sess)
	}
	// References carry discovery forward in the native protocol. When compaction
	// removes them (or the provider changes), restore those tools as eager. This
	// changes the prefix once, at compaction, rather than on every discovery.
	referenced := nativeLoadedTools(sess.Messages, a.config.model.Provider())
	loaded := loadedToolSet(sess)

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

	if directory := renderGroupDirectory(a.config.tools.List()); directory != "" {
		sections = append(sections, directory)
	}

	if instructions := renderGroupInstructions(defs); instructions != "" {
		sections = append(sections, instructions)
	}

	return defs, strings.Join(sections, "\n\n")
}

// renderGroupDirectory lists the groups that hold deferred tools, one line per
// group, plus a count of ungrouped deferred tools. Hosted search indexes names
// and descriptions server-side, so listing individual tools would only spend
// the tokens deferral saves.
func renderGroupDirectory(defs []llm.ToolDefinition) string {
	descriptions := make(map[string]string)
	ungrouped := 0

	for _, def := range defs {
		if !def.Deferred {
			continue
		}

		if def.Group.Name == "" {
			ungrouped++
			continue
		}

		descriptions[def.Group.Name] = strings.TrimSpace(def.Group.Description)
	}

	if len(descriptions) == 0 && ungrouped == 0 {
		return ""
	}

	names := slices.Sorted(maps.Keys(descriptions))

	var out strings.Builder

	out.WriteString("## Searchable tools\n\n")
	out.WriteString("More tools can be discovered with tool search. Never tell the user you lack a capability without searching for it first. Available groups:\n")

	for _, name := range names {
		out.WriteString("- " + name)

		if descriptions[name] != "" {
			out.WriteString(" - " + descriptions[name])
		}

		out.WriteString("\n")
	}

	if ungrouped > 0 {
		fmt.Fprintf(&out, "- %s - %d ungrouped tools\n", ungroupedHeading, ungrouped)
	}

	return strings.TrimRight(out.String(), "\n")
}

func nativeLoadedTools(messages []llm.Message, provider string) map[string]bool {
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

func visibleTools(defs []llm.ToolDefinition, sess *session.State, native bool) []llm.ToolDefinition {
	if !native {
		return defs
	}

	loaded := loadedToolSet(sess)

	visible := make([]llm.ToolDefinition, 0, len(defs))
	for _, def := range defs {
		if !def.Deferred || loaded[def.Name] {
			visible = append(visible, def)
		}
	}

	return visible
}

func (a *LLMAgent) toolTokens(defs []llm.ToolDefinition, sess *session.State, native bool) int {
	total := tokens.Tools(visibleTools(defs, sess, native))
	if !native {
		return total
	}
	// Native search has a small schema. OpenAI also exposes names and descriptions
	// of standalone deferred functions before discovery.
	total += 200
	loaded := loadedToolSet(sess)

	if a.config.model.Provider() == nativeOpenAI {
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

func (a *LLMAgent) recordNativeLoads(sess *session.State, message llm.Message) {
	if a.loader == nil {
		return
	}

	for _, part := range message.Content {
		if search, ok := part.(*llm.ToolSearchPart); ok && search != nil && search.Provider == a.config.model.Provider() {
			a.loader.commit(sess, search.Tools)
		}
	}
}

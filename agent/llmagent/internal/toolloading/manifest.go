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
	"fmt"
	"maps"
	"slices"
	"sort"
	"strings"
	"text/template"
	"unicode/utf8"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

type manifestTool struct {
	Name    string
	Summary string
}

type manifestGroup struct {
	Name        string
	Description string
	Tools       []manifestTool
}

type toolManifest struct {
	Count      int
	SearchTool string
	Examples   []string
	Groups     []manifestGroup
}

// ungroupedHeading heads deferred tools registered without a group. It always
// renders last.
const ungroupedHeading = "Other"

// buildToolManifest groups name-sorted deferred definitions. It includes loaded
// tools so the manifest stays unchanged as the session loads schemas.
func buildToolManifest(deferred []llm.ToolDefinition) toolManifest {
	byGroup := make(map[string][]manifestTool)
	descriptions := make(map[string]string)

	for _, def := range deferred {
		key := def.Group.Name
		if key == "" {
			key = ungroupedHeading
		}

		byGroup[key] = append(byGroup[key], manifestTool{
			Name:    def.Name,
			Summary: summarize(def.Description),
		})
		descriptions[key] = strings.TrimSpace(def.Group.Description)
	}

	names := make([]string, 0, len(byGroup))
	for name := range byGroup {
		if name != ungroupedHeading {
			names = append(names, name)
		}
	}

	sort.Strings(names)

	if _, exists := byGroup[ungroupedHeading]; exists {
		names = append(names, ungroupedHeading)
	}

	manifest := toolManifest{
		Count:      len(deferred),
		SearchTool: SearchToolName,
		Groups:     make([]manifestGroup, 0, len(names)),
	}

	for _, name := range names {
		manifest.Groups = append(manifest.Groups, manifestGroup{
			Name:        name,
			Description: descriptions[name],
			Tools:       byGroup[name],
		})
	}

	// Two real names make the batching example concrete. With fewer than two
	// deferred tools the example line is dropped rather than half-filled.
	if len(deferred) >= 2 {
		manifest.Examples = []string{deferred[0].Name, deferred[1].Name}
	}

	return manifest
}

// summarize reduces a tool description to one manifest line: the first
// sentence, whitespace-collapsed, truncated on a rune boundary at the last word
// break inside the byte budget.
func summarize(description string) string {
	collapsed := strings.Join(strings.Fields(description), " ")
	if collapsed == "" {
		return ""
	}

	if idx := strings.IndexAny(collapsed, ".!?\n"); idx > 0 {
		collapsed = strings.TrimSpace(collapsed[:idx])
	}

	if len(collapsed) <= summaryLimit {
		return collapsed
	}

	truncated := collapsed[:summaryLimit]
	for !utf8.ValidString(truncated) && len(truncated) > 0 {
		truncated = truncated[:len(truncated)-1]
	}

	if idx := strings.LastIndex(truncated, " "); idx > summaryLimit/2 {
		truncated = truncated[:idx]
	}

	return strings.TrimRight(truncated, " ,;:-") + "..."
}

// toolManifestTemplate stays stable as tools are loaded.
var toolManifestTemplate = template.Must(template.New("toolManifest").Parse(
	`## Additional tools

This catalog lists {{.Count}} tools available for loading. If a tool is already in your tool list,
call it directly. Otherwise, load its schema with ` + "`{{.SearchTool}}`" + ` and then call it.
Calling a tool before loading its schema will fail.
{{if .Examples}}
Load everything you expect to need in ONE ` + "`{{.SearchTool}}`" + ` call, for example
` + "`" + `{"query": "select:{{index .Examples 0}},{{index .Examples 1}}"}` + "`" + ` - loading one tool per call wastes a
round trip.
{{end}}
Never tell the user you lack a capability without searching for it first.
{{range .Groups}}
### {{.Name}}{{if .Description}} - {{.Description}}{{end}}
{{range .Tools}}- ` + "`{{.Name}}`" + `{{if .Summary}} - {{.Summary}}{{end}}
{{end}}{{end}}`))

func renderToolManifest(manifest toolManifest) string {
	var out strings.Builder

	err := toolManifestTemplate.Execute(&out, manifest)
	if err != nil {
		// The static template and data types cannot fail during execution.
		return ""
	}

	return strings.TrimRight(out.String(), "\n")
}

// summaryLimit is the byte budget for a tool's one-line manifest summary.
const summaryLimit = 120

//
// Group instructions
//

type instructionGroup struct {
	Name         string
	Instructions string
}

// groupInstructionsTemplate renders the instructions of every group with a
// visible tool. Its output changes only when the tools array does.
var groupInstructionsTemplate = template.Must(template.New("groupInstructions").Parse(
	`## Tool instructions
{{range .}}
### {{.Name}}

{{.Instructions}}
{{end}}`))

// renderGroupInstructions includes instructions for groups with visible tools,
// sorted by group name.
func renderGroupInstructions(visible []llm.ToolDefinition) string {
	seen := make(map[string]bool)
	var rendered []instructionGroup

	for _, def := range visible {
		group := def.Group
		if group.Name == "" || seen[group.Name] {
			continue
		}

		seen[group.Name] = true
		if text := strings.TrimSpace(group.Instructions); text != "" {
			rendered = append(rendered, instructionGroup{Name: group.Name, Instructions: text})
		}
	}

	if len(rendered) == 0 {
		return ""
	}

	sort.Slice(rendered, func(i, j int) bool { return rendered[i].Name < rendered[j].Name })

	var out strings.Builder

	err := groupInstructionsTemplate.Execute(&out, rendered)
	if err != nil {
		// Unreachable, as in renderToolManifest.
		return ""
	}

	return strings.TrimRight(out.String(), "\n")
}

//
// Group directory (native mode)
//

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

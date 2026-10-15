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

package main

import (
	"encoding/json"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

func TestRulesFile(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, problems := reconcile.ParseRules(raw)
	assert.Empty(t, problems)
	assert.NotEmpty(t, rules.Lines, "RULES.md lists no rules")
	assert.Contains(t, rules.Questions, reconcile.JudgmentRule, "rows with no rule need a question too")
	assert.Contains(t, rules.Lines, regionRule, "the waiting section links this rule")

	// RULES.md and the validator must list the same values; the schema is
	// held to the validator by TestSchemaMatchesValidator.
	assert.ElementsMatch(t, reasons, slices.Collect(maps.Keys(rules.Reasons)), "skip reasons in RULES.md vs report.go")
	assert.ElementsMatch(t, fields, slices.Collect(maps.Keys(rules.Fields)), "fields in RULES.md vs report.go")
}

func TestRulesReachTheAgent(t *testing.T) {
	t.Parallel()

	workflow, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)
	assert.Contains(t, string(workflow), "--append-system-prompt-file "+reconcile.RulesPath,
		"the workflow must hand RULES.md to the agent")

	prompt, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "prompt.md"))
	require.NoError(t, err)

	assert.Contains(t, string(prompt), ".claude/skills/reconcile-models/hosts/", "the prompt must send the agent to the host pages")

	// Every package a run owns has a how-to page, next to the shared one.
	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	var runs map[string]struct {
		Providers []string `json:"providers"`
	}
	require.NoError(t, json.Unmarshal(raw, &runs))

	pages := map[string]string{"prompt.md": string(prompt)}

	// Both skills follow RULES.md in every run, so they hold no rule either.
	for _, skill := range []string{"reconcile-models", "add-model"} {
		text, err := os.ReadFile(filepath.Join("..", "..", ".claude", "skills", skill, "SKILL.md"))
		require.NoError(t, err)

		pages[skill+"/SKILL.md"] = string(text)
	}

	for _, run := range runs {
		for _, pkg := range append([]string{"shared"}, run.Providers...) {
			page, err := os.ReadFile(filepath.Join("..", "..", ".claude", "skills", "reconcile-models", "hosts", pkg+".md"))
			require.NoError(t, err, "no host page for %s", pkg)

			pages["hosts/"+pkg+".md"] = string(page)
		}
	}

	// The rules live in RULES.md alone: a skip reason or field named in the
	// prompt or a host page means a rule was copied there.
	restated := slices.Concat(reasons, fields)
	for name, text := range pages {
		for _, word := range restated {
			assert.NotContains(t, text, "`"+word+"`", "%s restates a rule", name)
		}
	}

	// A rule that goes is deleted, so every rule a page cites in parentheses,
	// RULES.md's own prose included, must still exist.
	rulesFile, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, _ := reconcile.ParseRules(rulesFile)
	pages[reconcile.RulesPath] = string(rulesFile)
	cited := regexp.MustCompile(`\(([a-z]+(?:-[a-z0-9]+)+(?:, [a-z]+(?:-[a-z0-9]+)+)*)\)`)

	for name, text := range pages {
		for _, match := range cited.FindAllStringSubmatch(text, -1) {
			for id := range strings.SplitSeq(match[1], ", ") {
				if !slices.Contains(reconcile.AWSRegions, id) {
					assert.Contains(t, rules.Lines, id, "%s cites %s, which RULES.md doesn't define", name, id)
				}
			}
		}
	}
}

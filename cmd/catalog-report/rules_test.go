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
	assert.ElementsMatch(t, fields, slices.Collect(maps.Keys(kindNouns)), "every field needs a noun for the summary line")

	// ValidateAdded picks new_model or new_copy from the catalogue before the
	// patch, so a model this run adds on one host isn't a copy on the next.
	for _, field := range []string{reconcile.FieldNewModel, reconcile.FieldNewCopy} {
		assert.Contains(t, rules.Fields[field], "before this run", "the %s cell must say which catalogue it reads", field)
	}

	// A host's served locations change under their own Default.
	located := Report{Changes: []Change{{
		Offering: "vertex:claude-sonnet-5", Field: "location", Old: "global us-east5", New: "global us-east5 europe-west1",
		SourceURL: "https://docs.cloud.google.com/gemini-enterprise-agent-platform/resources/locations", Rule: "location-follow",
	}}}
	assert.Empty(t, Validate(located, []string{"docs.cloud.google.com"}, rules), "a location change under location-follow")

	// The locations table has listed a location the model page leaves out,
	// so the Default must say which of the two wins.
	require.Contains(t, rules.Lines, "location-follow")
	follow := strings.Split(string(raw), "\n")[rules.Lines["location-follow"]-1]
	assert.Contains(t, follow, "wins over the locations table", "location-follow must say which source wins")

	// On Bedrock a source region routes bare names, so changing one moves
	// callers and goes to a person.
	routed := Report{Changes: []Change{{
		Offering: "bedrock:us.anthropic.claude-sonnet-5-5", Field: "location", Old: "ca-west-1 us-east-1", New: "us-east-1",
		SourceURL: "https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-5-5.md", Rule: "location-follow",
	}}}
	assert.Equal(t, []string{"changes[0]: a location change on bedrock re-routes callers, so it goes to a person"},
		Validate(routed, []string{"docs.aws.amazon.com"}, rules))
}

func TestRulesReachTheAgent(t *testing.T) {
	t.Parallel()

	workflow, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", vendorWorkflow))
	require.NoError(t, err)
	assert.Contains(t, string(workflow), "--append-system-prompt-file "+reconcile.RulesPath,
		"the workflow must hand RULES.md to the agent")

	prompt, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "prompt.md"))
	require.NoError(t, err)

	assert.Contains(t, string(prompt), ".claude/skills/reconcile-models/hosts/", "the prompt must send the agent to the host pages")

	// Every package a run owns has a how-to page, next to the shared one.
	runs := namedSources(t)
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

// TestReviewGuideMatchesTheBody checks that REVIEW.md has the section every
// bot PR body links to, and that each PR body section the guide names in
// quotes is still one Render writes.
func TestReviewGuideMatchesTheBody(t *testing.T) {
	t.Parallel()

	guide, err := os.ReadFile(filepath.Join("..", "..", "REVIEW.md"))
	require.NoError(t, err)

	_, section, found := strings.Cut(string(guide), "\n## Bot catalog PRs\n")
	require.True(t, found, `REVIEW.md has no "Bot catalog PRs" section for the PR body's link`)

	section, _, _ = strings.Cut(section, "\n## ")

	renderer, err := os.ReadFile("report.go")
	require.NoError(t, err)

	names := regexp.MustCompile(`"([A-Z][a-z]+(?: [a-z]+)*)"`).FindAllStringSubmatch(section, -1)
	require.NotEmpty(t, names, "the guide names no PR body sections")

	for _, name := range names {
		assert.Contains(t, string(renderer), "### "+name[1], "REVIEW.md names a PR body section %q that Render doesn't write", name[1])
	}
}

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
	"slices"
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

	// The rules live in RULES.md alone: a skip reason or field named in the
	// prompt means a rule was copied there.
	for _, word := range slices.Concat(reasons, fields) {
		assert.NotContains(t, string(prompt), "`"+word+"`", "prompt.md restates a rule")
	}
}

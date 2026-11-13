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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

func TestFireworksRunsBelongToCreators(t *testing.T) {
	t.Parallel()

	sources := namedSources(t)
	assert.NotContains(t, sources, "fireworks", "the host is not the publisher")

	for _, vendor := range []string{"openai", "zai", "minimax", "deepseek", "moonshotai"} {
		run, ok := sources[vendor]
		require.True(t, ok)
		assert.Contains(t, run.Providers, "fireworks")
		assert.Contains(t, run.Hosts, "docs.fireworks.ai")
		assert.Contains(t, run.Hosts, "fireworks.ai")
	}
}

func TestFireworksDocumentedChangesFollowOrdinaryRules(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, problems := reconcile.ParseRules(raw)
	require.Empty(t, problems)
	sources := namedSources(t)

	for _, tt := range []struct {
		name, field, old, newValue, rule string
	}{
		{"new model", "new_model", "", "accounts/fireworks/models/glm-5p3-flash", "new-model-add"},
		{"new host copy", "new_copy", "", "accounts/fireworks/models/glm-5p3-flash", "new-model-add"},
		{"capability", "capability", "structured output false", "structured output true", "capability-follow"},
		{"reasoning control", "capability", "", "reasoning effort low high max", "capability-follow"},
		{"price", "price", "0.15", "0.18", "price-follow"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			report := Report{Changes: []Change{{
				Offering: "fireworks:accounts/fireworks/models/glm-5p3-flash", Field: tt.field, Old: tt.old, New: tt.newValue,
				SourceURL: "https://fireworks.ai/models/fireworks/glm-5p3-flash", Rule: tt.rule,
			}}}
			assert.Empty(t, Validate(report, "zai", sources["zai"], rules))
		})
	}
}

func TestFireworksRetainsRuntimeAndRouterRules(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, problems := reconcile.ParseRules(raw)
	require.Empty(t, problems)

	for _, rule := range []string{"model-not-callable", "reasoning-unsendable"} {
		require.Contains(t, rules.Lines, rule)
		row := strings.Split(string(raw), "\n")[rules.Lines[rule]-1]
		assert.Contains(t, row, "selected-binding")
		assert.Contains(t, row, "native catalog metadata")
	}

	sources := namedSources(t)
	for _, tt := range []struct {
		name   string
		report Report
	}{
		{"unsendable controls", Report{NeedsHuman: []NeedsHuman{{
			Offering: "fireworks:accounts/fireworks/models/glm-5p3-flash", Field: "capability",
			SourceURLs: []string{"https://docs.fireworks.ai/api-reference/post-chatcompletions.md"}, Rule: "reasoning-unsendable",
		}}}},
		{"uncallable runtime", Report{Skipped: []Skipped{{
			Offering: "fireworks:accounts/fireworks/models/glm-5p3-flash", Reason: "not_callable", Rule: "model-not-callable",
		}}}},
		{"unscoped router", Report{Skipped: []Skipped{{
			Offering: "fireworks:accounts/fireworks/routers/glm-5p3-fast", Reason: "out_of_scope", Rule: "fireworks-router-unscoped",
		}}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Empty(t, Validate(tt.report, "zai", sources["zai"], rules))
		})
	}
}

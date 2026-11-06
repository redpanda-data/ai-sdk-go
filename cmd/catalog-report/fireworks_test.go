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

func TestFireworksContractExpansionNeedsQualification(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", reconcile.RulesPath))
	require.NoError(t, err)

	rules, problems := reconcile.ParseRules(raw)
	require.Empty(t, problems)

	const qualification = "fireworks-contract-qualification"
	require.Contains(t, rules.Lines, qualification)

	row := strings.Split(string(raw), "\n")[rules.Lines[qualification]-1]
	for _, text := range []string{"exact offering and feature", "fresh committed live-contract evidence", "CI has no Fireworks credentials", "wire API", "capability", "reasoning control"} {
		assert.Contains(t, row, text)
	}

	for _, rule := range []string{"model-not-callable", "reasoning-unsendable"} {
		row := strings.Split(string(raw), "\n")[rules.Lines[rule]-1]
		assert.Contains(t, row, "selected-binding")
		assert.Contains(t, row, qualification)
	}

	sources := namedSources(t)

	for _, field := range []string{"new_copy", "capability"} {
		report := Report{NeedsHuman: []NeedsHuman{{
			Offering: "fireworks:accounts/fireworks/models/gpt-oss-120b", Field: field,
			SourceURLs: []string{"https://docs.fireworks.ai/tools-sdks/anthropic-compatibility"}, Rule: qualification,
		}}}
		assert.Empty(t, Validate(report, "openai", sources["openai"], rules))
	}

	report := Report{Changes: []Change{{
		Offering: "fireworks:accounts/fireworks/models/gpt-oss-120b", Field: "price", Old: "0.15", New: "0.18",
		SourceURL: "https://docs.fireworks.ai/serverless/pricing", Rule: "price-follow",
	}}}
	assert.Empty(t, Validate(report, "openai", sources["openai"], rules), "qualified copies still get price maintenance")

	report.Changes[0].Rule = qualification
	assert.NotEmpty(t, Validate(report, "openai", sources["openai"], rules), "qualification is an Ask, not permission to expand the contract")
}

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
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock/bedrocktest"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat/openaicompattest"
)

// TestLiveTestModelsAreCurrent fails the pull request that records a
// deprecation, shutdown or replacement for a model the live tests share, and
// prints the swap, so the setting moves before the vendor turns the model off
// and main goes red. It reads committed data only, never the clock.
func TestLiveTestModelsAreCurrent(t *testing.T) {
	t.Parallel()

	// The reconcile bot's sandbox runs -short and drops a change that fails a
	// test, so failing there would drop the very date this check warns about.
	// The full Test job runs it once a person marks the bot's PR ready.
	if testing.Short() {
		t.Skip("skipping in -short mode, where reconcile runs record catalog dates")
	}

	for _, problem := range checkLiveTestModels(liveTestModels(), catalogLifecycles()) {
		t.Error(problem)
	}
}

// untrackedTestModels are shared model settings whose host has no catalog, so
// nothing records when their models shut down.
var untrackedTestModels = []string{
	"openaicompattest.DeepSeekDefaultStandardModel",
	"openaicompattest.DeepSeekDefaultReasoningModel",
}

// TestEveryTestModelSettingIsChecked keeps liveTestModels complete: every
// constant named like a model setting in providers/*/*test must be a row
// there or be listed as untracked. It reads source, not catalog data, so it
// also runs under -short.
func TestEveryTestModelSettingIsChecked(t *testing.T) {
	t.Parallel()

	paths, err := filepath.Glob(filepath.Join("..", "..", "providers", "*", "*test", "*.go"))
	require.NoError(t, err)
	require.NotEmpty(t, paths)

	var declared []string

	fset := token.NewFileSet()
	for _, path := range paths {
		file, err := parser.ParseFile(fset, path, nil, parser.SkipObjectResolution)
		require.NoError(t, err)

		for _, decl := range file.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.CONST {
				continue
			}

			for _, spec := range gen.Specs {
				value, ok := spec.(*ast.ValueSpec)
				if !ok {
					continue
				}

				for _, name := range value.Names {
					if strings.Contains(name.Name, "Model") {
						declared = append(declared, file.Name.Name+"."+name.Name)
					}
				}
			}
		}
	}

	models := liveTestModels()

	checked := make([]string, 0, len(models)+len(untrackedTestModels))
	for _, m := range models {
		checked = append(checked, m.setting)
	}

	assert.ElementsMatch(t, append(checked, untrackedTestModels...), declared)
}

// liveModel is a shared live-test model setting and the host its tests call.
type liveModel struct {
	setting string         // e.g. "openaitest.TestModelName"
	value   string         // the model ID the tests send
	host    llm.ProviderID // the catalog that dates the model for this host
	geo     string         // Bedrock only: the profile prefix NewModel adds to a bare ID
}

// liveTestModels is every shared model setting in providers/*/*test whose host
// has a catalog (DeepSeek's have none), with the catalog of the host its tests
// call.
func liveTestModels() []liveModel {
	bedrockGeo := bedrock.InferenceProfileRegion(bedrocktest.TestRegion)

	return []liveModel{
		{setting: "anthropictest.TestModelName", value: anthropictest.TestModelName, host: anthropic.ProviderName},
		{setting: "anthropictest.TestReasoningModelName", value: anthropictest.TestReasoningModelName, host: anthropic.ProviderName},
		{setting: "anthropictest.TestAdaptiveModelName", value: anthropictest.TestAdaptiveModelName, host: anthropic.ProviderName},
		{setting: "anthropictest.TestNoThinkingModelName", value: anthropictest.TestNoThinkingModelName, host: anthropic.ProviderName},
		{setting: "bedrocktest.TestModelName", value: bedrocktest.TestModelName, host: bedrock.ProviderName, geo: bedrockGeo},
		{setting: "googletest.TestModelName", value: googletest.TestModelName, host: google.ProviderName},
		{setting: "googletest.TestReasoningModelName", value: googletest.TestReasoningModelName, host: google.ProviderName},
		{setting: "openaitest.TestModelName", value: openaitest.TestModelName, host: openai.ProviderName},
		{setting: "openaitest.TestReasoningModelName", value: openaitest.TestReasoningModelName, host: openai.ProviderName},
		// openaicompat has no catalog of its own: each setting is dated by the
		// host its tests point the client at.
		{setting: "openaicompattest.TestModelName", value: openaicompattest.TestModelName, host: openai.ProviderName},
		{setting: "openaicompattest.AnthropicDefaultModel", value: openaicompattest.AnthropicDefaultModel, host: anthropic.ProviderName},
		{setting: "openaicompattest.GoogleDefaultModel", value: openaicompattest.GoogleDefaultModel, host: google.ProviderName},
	}
}

// catalogLifecycles maps each host's offering IDs to their lifecycles.
func catalogLifecycles() map[llm.ProviderID]map[string]catalog.Lifecycle {
	lifecycles := map[llm.ProviderID]map[string]catalog.Lifecycle{}

	for _, cat := range allCatalogs() {
		byID := map[string]catalog.Lifecycle{}
		for _, o := range cat.All() {
			byID[o.ID] = o.Life
		}

		lifecycles[cat.Provider()] = byID
	}

	return lifecycles
}

func TestCheckLiveTestModels(t *testing.T) {
	t.Parallel()

	openaiModel := func(value string) liveModel {
		return liveModel{setting: "openaitest.TestReasoningModelName", value: value, host: openai.ProviderName}
	}
	bedrockModel := func(value string) liveModel {
		return liveModel{setting: "bedrocktest.TestModelName", value: value, host: bedrock.ProviderName, geo: "us"}
	}

	lifecycles := map[llm.ProviderID]map[string]catalog.Lifecycle{
		openai.ProviderName: {
			"gpt-5.4-mini": {Stage: catalog.StageGA},
			"o3": {
				Stage:      catalog.StageGA,
				Deprecated: catalog.MustDate("2026-06-11"),
				Retires:    catalog.MustDate("2026-12-11"),
				ReplacedBy: "gpt-5.6-sol",
			},
			"gpt-5.4":         {Stage: catalog.StageGA, Retires: catalog.MustDate("2099-01-01"), ReplacedBy: "gpt-5.6-sol"},
			"gpt-5.5-preview": {Stage: catalog.StagePreview, ReplacedBy: "gpt-5.6-sol"},
			"gpt-5.4-nano":    {Stage: catalog.StageGA, Deprecated: catalog.MustDate("2026-10-01")},
		},
		bedrock.ProviderName: {
			"us.anthropic.claude-sonnet-4-5-20250929-v1:0": {
				Stage:      catalog.StageGA,
				Deprecated: catalog.MustDate("2026-10-08"),
				Retires:    catalog.MustDate("2027-04-08"),
				ReplacedBy: "global.anthropic.claude-sonnet-5-5",
			},
			"us.anthropic.claude-opus-4-5-20251101-v1:0": {Stage: catalog.StageGA},
		},
	}

	tests := []struct {
		name  string
		model liveModel
		want  string
	}{
		{"current offering", openaiModel("gpt-5.4-mini"), ""},
		{
			"not an offering", openaiModel("o1"),
			`openaitest.TestReasoningModelName = "o1": not an offering ID in the openai catalog`,
		},
		{
			"deprecated with a replacement", openaiModel("o3"),
			`openaitest.TestReasoningModelName = "o3": openai records Deprecated 2026-06-11, Retires 2026-12-11, ReplacedBy gpt-5.6-sol. ` +
				`In providers/openai/openaitest/constants.go set TestReasoningModelName = "gpt-5.6-sol"`,
		},
		{
			// A shutdown far in the future still fails today: the check reads
			// recorded data, never the clock.
			"shutdown date only", openaiModel("gpt-5.4"),
			`openaitest.TestReasoningModelName = "gpt-5.4": openai records Retires 2099-01-01, ReplacedBy gpt-5.6-sol. ` +
				`In providers/openai/openaitest/constants.go set TestReasoningModelName = "gpt-5.6-sol"`,
		},
		{
			"replacement only", openaiModel("gpt-5.5-preview"),
			`openaitest.TestReasoningModelName = "gpt-5.5-preview": openai records ReplacedBy gpt-5.6-sol. ` +
				`In providers/openai/openaitest/constants.go set TestReasoningModelName = "gpt-5.6-sol"`,
		},
		{
			"deprecated with no replacement", openaiModel("gpt-5.4-nano"),
			`openaitest.TestReasoningModelName = "gpt-5.4-nano": openai records Deprecated 2026-10-01 and no replacement. ` +
				`Pick one for TestReasoningModelName in providers/openai/openaitest/constants.go`,
		},
		{
			// The swap stays bare, so NewModel adds the test region's prefix
			// again: CI's account can't call global. profiles.
			"bedrock bare ID", bedrockModel("anthropic.claude-sonnet-4-5-20250929-v1:0"),
			`bedrocktest.TestModelName = "anthropic.claude-sonnet-4-5-20250929-v1:0": aws.bedrock records Deprecated 2026-10-08, ` +
				`Retires 2027-04-08, ReplacedBy global.anthropic.claude-sonnet-5-5 (as us.anthropic.claude-sonnet-4-5-20250929-v1:0). ` +
				`In providers/bedrock/bedrocktest/constants.go set TestModelName = "anthropic.claude-sonnet-5-5"`,
		},
		{
			// A prefixed setting keeps its own prefix in the swap, not the
			// global. one Bedrock records.
			"bedrock profile ID", bedrockModel("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
			`bedrocktest.TestModelName = "us.anthropic.claude-sonnet-4-5-20250929-v1:0": aws.bedrock records Deprecated 2026-10-08, ` +
				`Retires 2027-04-08, ReplacedBy global.anthropic.claude-sonnet-5-5. ` +
				`In providers/bedrock/bedrocktest/constants.go set TestModelName = "us.anthropic.claude-sonnet-5-5"`,
		},
		// NewModel calls an ID the catalog already knows as it is.
		{"bedrock offering ID", bedrockModel("us.anthropic.claude-opus-4-5-20251101-v1:0"), ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// One model per case, so at most one problem.
			problems := checkLiveTestModels([]liveModel{tt.model}, lifecycles)
			assert.Equal(t, tt.want, strings.Join(problems, "\n"))
		})
	}
}

// checkLiveTestModels lists each setting whose model isn't an exact offering ID
// in its host's catalog, or whose offering records a deprecation, shutdown or
// replacement.
func checkLiveTestModels(models []liveModel, lifecycles map[llm.ProviderID]map[string]catalog.Lifecycle) []string {
	var problems []string

	for _, m := range models {
		problem := fmt.Sprintf("%s = %q: ", m.setting, m.value)
		offerings := lifecycles[m.host]

		// The exact ID first, then the test region's profile ID: the default
		// route bedrock's NewModel takes for a bare name.
		id := m.value

		life, ok := offerings[id]
		if !ok && m.geo != "" {
			id = m.geo + "." + m.value
			life, ok = offerings[id]
		}

		if !ok {
			problems = append(problems, problem+fmt.Sprintf("not an offering ID in the %s catalog", m.host))
			continue
		}

		var recorded []string
		if !life.Deprecated.IsZero() {
			recorded = append(recorded, "Deprecated "+life.Deprecated.Format(time.DateOnly))
		}

		if !life.Retires.IsZero() {
			recorded = append(recorded, "Retires "+life.Retires.Format(time.DateOnly))
		}

		if life.ReplacedBy != "" {
			recorded = append(recorded, "ReplacedBy "+life.ReplacedBy)
		}

		if len(recorded) == 0 {
			continue
		}

		problem += fmt.Sprintf("%s records %s", m.host, strings.Join(recorded, ", "))

		swap := life.ReplacedBy

		if id != m.value {
			problem += fmt.Sprintf(" (as %s)", id)
		}

		// Bedrock records global. replacements, which CI's account can't call.
		// A bare setting's swap stays bare, so NewModel adds the test region's
		// prefix; a prefixed setting's swap keeps that prefix.
		if m.geo != "" && strings.HasPrefix(id, m.geo+".") {
			_, swap, _ = strings.Cut(swap, ".")
			if id == m.value {
				swap = m.geo + "." + swap
			}
		}

		// Every shared setting is declared in providers/<p>/<p>test/constants.go.
		pkg, name, _ := strings.Cut(m.setting, ".")
		file := "providers/" + strings.TrimSuffix(pkg, "test") + "/" + pkg + "/constants.go"

		if life.ReplacedBy == "" {
			problems = append(problems, problem+fmt.Sprintf(" and no replacement. Pick one for %s in %s", name, file))
			continue
		}

		problems = append(problems, problem+fmt.Sprintf(". In %s set %s = %q", file, name, swap))
	}

	return problems
}

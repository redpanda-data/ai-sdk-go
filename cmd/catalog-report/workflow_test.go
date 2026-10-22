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
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

// workflowStep is one step of a job in reconcile-models.yml.
type workflowStep struct {
	Uses string            `yaml:"uses"`
	Run  string            `yaml:"run"`
	With map[string]string `yaml:"with"`
}

// namedSource is one run's entry in named-sources.json.
type namedSource struct {
	Providers []string `json:"providers"`
	Hosts     []string `json:"hosts"`
}

// namedSources reads named-sources.json, keyed by run.
func namedSources(t *testing.T) map[string]namedSource {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	var runs map[string]namedSource
	require.NoError(t, json.Unmarshal(raw, &runs))
	require.NotEmpty(t, runs)

	return runs
}

// agentSteps reads the agent job's steps from the workflow file.
func agentSteps(t *testing.T) []workflowStep {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)

	var workflow struct {
		Jobs map[string]struct {
			Steps []workflowStep `yaml:"steps"`
		} `yaml:"jobs"`
	}
	require.NoError(t, yaml.Unmarshal(raw, &workflow))

	steps := workflow.Jobs["agent"].Steps
	require.NotEmpty(t, steps, "the workflow has no agent job steps")

	return steps
}

// TestAgentEgressAllowsNamedSources checks that Harden-Runner blocks the agent
// job's outbound traffic from its first step, and that every named-source host
// is on the allowlist, so a host added to named-sources.json can't become
// unreachable unnoticed.
func TestAgentEgressAllowsNamedSources(t *testing.T) {
	t.Parallel()

	first := agentSteps(t)[0]
	require.True(t, strings.HasPrefix(first.Uses, "step-security/harden-runner@"),
		"Harden-Runner must be the agent job's first step, got %q", first.Uses)
	assert.Equal(t, "block", first.With["egress-policy"])

	allowed := strings.Fields(first.With["allowed-endpoints"])

	for name, run := range namedSources(t) {
		require.NotEmpty(t, run.Hosts, "the %s run names no hosts", name)

		for _, host := range run.Hosts {
			assert.Contains(t, allowed, host+":443", "the %s run's named host %s isn't on the agent job's allowlist", name, host)
		}
	}
}

// TestAgentChecksEgressBeforeClaude checks that the agent job runs
// check-egress.sh before Claude starts.
func TestAgentChecksEgressBeforeClaude(t *testing.T) {
	t.Parallel()

	steps := agentSteps(t)
	claude := slices.IndexFunc(steps, func(s workflowStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action@")
	})
	require.NotEqual(t, -1, claude, "the agent job has no claude-code-action step")

	assert.True(t, slices.ContainsFunc(steps[:claude], func(s workflowStep) bool {
		return strings.Contains(s.Run, ".github/reconcile-models/check-egress.sh")
	}), "check-egress.sh must run before Claude starts")
}

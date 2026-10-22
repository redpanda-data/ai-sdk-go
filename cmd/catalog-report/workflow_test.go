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
	"regexp"
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

// workflowJob is one job in reconcile-models.yml.
type workflowJob struct {
	Env   map[string]string `yaml:"env"`
	Steps []workflowStep    `yaml:"steps"`
}

// agentJob reads the agent job from the workflow file.
func agentJob(t *testing.T) workflowJob {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)

	var workflow struct {
		Jobs map[string]workflowJob `yaml:"jobs"`
	}
	require.NoError(t, yaml.Unmarshal(raw, &workflow))

	job := workflow.Jobs["agent"]
	require.NotEmpty(t, job.Steps, "the workflow has no agent job steps")

	return job
}

// claudeStep is the index of the agent job's claude-code-action step.
func claudeStep(t *testing.T, steps []workflowStep) int {
	t.Helper()

	i := slices.IndexFunc(steps, func(s workflowStep) bool {
		return strings.HasPrefix(s.Uses, "anthropics/claude-code-action@")
	})
	require.NotEqual(t, -1, i, "the agent job has no claude-code-action step")

	return i
}

// TestAgentEgressAllowsNamedSources checks that Harden-Runner blocks the agent
// job's outbound traffic from its first step, and that every named-source host
// is on the allowlist, so a host added to named-sources.json can't become
// unreachable unnoticed.
func TestAgentEgressAllowsNamedSources(t *testing.T) {
	t.Parallel()

	first := agentJob(t).Steps[0]
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

	steps := agentJob(t).Steps
	claude := claudeStep(t, steps)

	assert.True(t, slices.ContainsFunc(steps[:claude], func(s workflowStep) bool {
		return strings.Contains(s.Run, ".github/reconcile-models/check-egress.sh")
	}), "check-egress.sh must run before Claude starts")
}

// TestAgentCannotReadProcessFiles checks that the agent's file tools are denied
// /proc and that its shell commands run with the job's secrets scrubbed.
func TestAgentCannotReadProcessFiles(t *testing.T) {
	t.Parallel()

	job := agentJob(t)
	assert.Equal(t, "1", job.Env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "the agent's shell must run without the job's secrets")

	args := job.Steps[claudeStep(t, job.Steps)].With["claude_args"]
	disallowed := regexp.MustCompile(`--disallowedTools "([^"]*)"`).FindStringSubmatch(args)
	require.NotNil(t, disallowed, "the Claude step passes no --disallowedTools list")
	assert.Contains(t, strings.Split(disallowed[1], ","), "Read(//proc/**)")
}

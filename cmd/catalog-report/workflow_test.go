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
	"os/exec"
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
	If    string            `yaml:"if"`
	Env   map[string]string `yaml:"env"`
	Steps []workflowStep    `yaml:"steps"`
}

// workflow is reconcile-models.yml: its triggers and its jobs.
type workflow struct {
	On struct {
		Schedule []struct {
			Cron string `yaml:"cron"`
		} `yaml:"schedule"`
	} `yaml:"on"`
	Jobs map[string]workflowJob `yaml:"jobs"`
}

// readWorkflow reads the workflow file.
func readWorkflow(t *testing.T) workflow {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "reconcile-models.yml"))
	require.NoError(t, err)

	var wf workflow
	require.NoError(t, yaml.Unmarshal(raw, &wf))

	return wf
}

// readJob reads one job from the workflow file.
func readJob(t *testing.T, name string) workflowJob {
	t.Helper()

	job := readWorkflow(t).Jobs[name]
	require.NotEmpty(t, job.Steps, "the workflow has no %s job steps", name)

	return job
}

// stepIndex is the index of the first step that matches; what names the step
// in the failure message.
func stepIndex(t *testing.T, steps []workflowStep, what string, match func(workflowStep) bool) int {
	t.Helper()

	i := slices.IndexFunc(steps, match)
	require.NotEqual(t, -1, i, "the job has no %s step", what)

	return i
}

// isClaude matches the agent job's claude-code-action step.
func isClaude(s workflowStep) bool {
	return strings.HasPrefix(s.Uses, "anthropics/claude-code-action@")
}

// TestAgentEgressAllowsNamedSources checks that Harden-Runner blocks the agent
// job's outbound traffic from its first step, and that every named-source host
// is on the allowlist, so a host added to named-sources.json can't become
// unreachable unnoticed.
func TestAgentEgressAllowsNamedSources(t *testing.T) {
	t.Parallel()

	first := readJob(t, "agent").Steps[0]
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

	steps := readJob(t, "agent").Steps
	claude := stepIndex(t, steps, "claude-code-action", isClaude)

	assert.True(t, slices.ContainsFunc(steps[:claude], func(s workflowStep) bool {
		return strings.Contains(s.Run, ".github/reconcile-models/check-egress.sh")
	}), "check-egress.sh must run before Claude starts")
}

// TestEgressCheckWaitsForTheBlock runs check-egress.sh with a stand-in
// Harden-Runner status file, curl and sleep.
func TestEgressCheckWaitsForTheBlock(t *testing.T) {
	t.Parallel()

	script := filepath.Join("..", "..", ".github", "reconcile-models", "check-egress.sh")
	tests := []struct {
		name     string
		on       string // when the status file says Initialized: "start", "late" or never
		curlExit string // 0 reaches the probe; 7 is a refused connection
		passes   bool
	}{
		{name: "on from the start", on: "start", curlExit: "7", passes: true},
		{name: "on after a slow start", on: "late", curlExit: "7", passes: true},
		{name: "never on", curlExit: "7", passes: false},
		{name: "on, but the probe gets through", on: "start", curlExit: "0", passes: false},
	}

	// The stand-ins are shared, since a new executable is slow to start the
	// first time on macOS. The sleep returns at once, writing the status if it's late.
	stubs := t.TempDir()
	for name, body := range map[string]string{
		"curl":  "#!/bin/sh\nexit \"$CURL_EXIT\"\n",
		"sleep": "#!/bin/sh\n[ -z \"$LATE_STATUS\" ] || printf Initialized >\"$LATE_STATUS\"\n",
	} {
		require.NoError(t, os.WriteFile(filepath.Join(stubs, name), []byte(body), 0o700)) //nolint:gosec // stand-ins must be executable
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			status := filepath.Join(t.TempDir(), "agent.status")

			if tt.on == "start" {
				require.NoError(t, os.WriteFile(status, []byte("Initialized"), 0o600))
			}

			late := ""
			if tt.on == "late" {
				late = status
			}

			cmd := exec.CommandContext(t.Context(), "bash", script, status)

			cmd.Env = append(os.Environ(), "PATH="+stubs+string(os.PathListSeparator)+os.Getenv("PATH"),
				"CURL_EXIT="+tt.curlExit, "LATE_STATUS="+late)

			out, err := cmd.CombinedOutput()
			assert.Equal(t, tt.passes, err == nil, "%s", out)
		})
	}
}

// TestAgentCannotReadProcessFiles checks that the agent's file tools are denied
// /proc and that its shell commands run with the job's secrets scrubbed.
func TestAgentCannotReadProcessFiles(t *testing.T) {
	t.Parallel()

	job := readJob(t, "agent")
	assert.Equal(t, "1", job.Env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "the agent's shell must run without the job's secrets")

	args := job.Steps[stepIndex(t, job.Steps, "claude-code-action", isClaude)].With["claude_args"]
	disallowed := regexp.MustCompile(`--disallowedTools "([^"]*)"`).FindStringSubmatch(args)
	require.NotNil(t, disallowed, "the Claude step passes no --disallowedTools list")
	assert.Contains(t, strings.Split(disallowed[1], ","), "Read(//proc/**)")
}

// TestOthersCommitsFindsEveryNonBotCommit runs others-commits.sh on trimmed
// answers of GitHub's list of a pull request's commits.
func TestOthersCommitsFindsEveryNonBotCommit(t *testing.T) {
	t.Parallel()

	script := filepath.Join("..", "..", ".github", "reconcile-models", "others-commits.sh")
	tests := []struct {
		sample string
		want   string
	}{
		{sample: "bot-only", want: ""},
		{sample: "person-commit", want: "b2c3d4e"},
		{sample: "update-branch", want: "c3d4e5f"},
		{sample: "amended", want: "d4e5f60"},
		{sample: "unlinked-author", want: "e5f6071, f607182"},
	}

	for _, tt := range tests {
		t.Run(tt.sample, func(t *testing.T) {
			t.Parallel()

			sample, err := os.Open(filepath.Join("testdata", "others-commits", tt.sample+".json"))
			require.NoError(t, err)

			defer sample.Close()

			cmd := exec.CommandContext(t.Context(), "bash", script, "vbotbuildovich")
			cmd.Stdin = sample

			out, err := cmd.Output()
			require.NoError(t, err)
			assert.Equal(t, tt.want, strings.TrimSpace(string(out)))
		})
	}
}

// TestPushChecksOthersCommitsFirst checks that the PR job looks for other
// people's commits on the bot PR before it pushes, and that the push is leased
// to the branch head it read, so a commit pushed in between isn't lost either.
func TestPushChecksOthersCommitsFirst(t *testing.T) {
	t.Parallel()

	steps := readJob(t, "pr").Steps
	run := steps[stepIndex(t, steps, "leased push", func(s workflowStep) bool {
		return strings.Contains(s.Run, "git push --force-with-lease=")
	})].Run

	assert.NotContains(t, run, "git push --force ", "the push must be leased, not a plain force-push")

	check := strings.Index(run, ".github/reconcile-models/others-commits.sh")
	require.NotEqual(t, -1, check, "the push step doesn't run others-commits.sh")
	assert.Less(t, check, strings.Index(run, "git push --force-with-lease="), "others-commits.sh must run before the push")
}

// TestScheduledRunsReachThePR checks that the workflow has a schedule and that
// the agent job doesn't skip scheduled runs.
func TestScheduledRunsReachThePR(t *testing.T) {
	t.Parallel()

	wf := readWorkflow(t)
	require.NotEmpty(t, wf.On.Schedule, "the workflow has no schedule")
	assert.NotContains(t, wf.Jobs["agent"].If, "schedule", "the agent job must run on a schedule too")
}

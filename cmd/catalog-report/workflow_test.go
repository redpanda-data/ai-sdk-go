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
	"fmt"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// workflowStep is one step of a job in reconcile-models.yml or
// reconcile-vendor.yml.
type workflowStep struct {
	ID   string            `yaml:"id"`
	Name string            `yaml:"name"`
	If   string            `yaml:"if"`
	Uses string            `yaml:"uses"`
	Run  string            `yaml:"run"`
	With map[string]string `yaml:"with"`
	// ContinueOnError is true, or an expression.
	ContinueOnError any `yaml:"continue-on-error"` //nolint:tagliatelle // GitHub's key
}

// namedSources reads named-sources.json, keyed by run.
func namedSources(t *testing.T) map[string]namedSource {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	runs, err := parseSources(raw)
	require.NoError(t, err)
	require.NotEmpty(t, runs)

	return runs
}

// vendorOrder lists the named-sources.json keys in the file's order, the order
// a run's chain takes the vendors in.
func vendorOrder(t *testing.T) []string {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	// YAML reads JSON and keeps a mapping's order.
	var doc yaml.Node
	require.NoError(t, yaml.Unmarshal(raw, &doc))

	var keys []string

	for i, node := range doc.Content[0].Content {
		if i%2 == 0 {
			keys = append(keys, node.Value)
		}
	}

	return keys
}

// needsOf lists the jobs a job needs.
func needsOf(job workflowJob) []string {
	switch needs := job.Needs.(type) {
	case string:
		return []string{needs}
	case []any:
		names := make([]string, 0, len(needs))
		for _, name := range needs {
			names = append(names, fmt.Sprint(name))
		}

		return names
	}

	return nil
}

// workflowJob is one job: its own steps, or a call of reconcile-vendor.yml
// (Uses, With and Secrets).
type workflowJob struct {
	If          string            `yaml:"if"`
	Env         map[string]string `yaml:"env"`
	Permissions map[string]string `yaml:"permissions"`
	Outputs     map[string]string `yaml:"outputs"`
	Steps       []workflowStep    `yaml:"steps"`
	Uses        string            `yaml:"uses"`
	With        map[string]any    `yaml:"with"`
	// Secrets is "inherit" or a map of the secrets passed; Needs is one job
	// or a list.
	Secrets any `yaml:"secrets"`
	Needs   any `yaml:"needs"`
}

// workflow is reconcile-models.yml or reconcile-vendor.yml: its triggers and
// its jobs.
type workflow struct {
	On struct {
		Schedule []struct {
			Cron string `yaml:"cron"`
		} `yaml:"schedule"`
		WorkflowCall struct {
			Inputs map[string]struct {
				Type     string `yaml:"type"`
				Required bool   `yaml:"required"`
			} `yaml:"inputs"`
			Outputs map[string]struct {
				Value string `yaml:"value"`
			} `yaml:"outputs"`
			Secrets map[string]any `yaml:"secrets"`
		} `yaml:"workflow_call"`
		WorkflowDispatch struct {
			Inputs map[string]struct {
				Options []string `yaml:"options"`
				Default string   `yaml:"default"`
			} `yaml:"inputs"`
		} `yaml:"workflow_dispatch"`
	} `yaml:"on"`
	Concurrency struct {
		Group            string `yaml:"group"`
		Queue            string `yaml:"queue"`
		CancelInProgress bool   `yaml:"cancel-in-progress"` //nolint:tagliatelle // GitHub's key
	} `yaml:"concurrency"`
	Env  map[string]string      `yaml:"env"`
	Jobs map[string]workflowJob `yaml:"jobs"`
}

// The run's workflow, and the one it calls for each vendor.
const (
	runWorkflow    = "reconcile-models.yml"
	vendorWorkflow = "reconcile-vendor.yml"
)

// readWorkflow reads one of the two workflow files.
func readWorkflow(t *testing.T, file string) workflow {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", file))
	require.NoError(t, err)

	var wf workflow
	require.NoError(t, yaml.Unmarshal(raw, &wf))

	return wf
}

// readJob reads one job with steps from one of the two workflow files.
func readJob(t *testing.T, file, name string) workflowJob {
	t.Helper()

	job := readWorkflow(t, file).Jobs[name]
	require.NotEmpty(t, job.Steps, "%s has no %s job steps", file, name)

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

	first := readJob(t, vendorWorkflow, "agent").Steps[0]
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

// TestAgentRunsChecksBeforeClaude checks that the agent job runs
// check-egress.sh and stop-docker.sh before Claude starts.
func TestAgentRunsChecksBeforeClaude(t *testing.T) {
	t.Parallel()

	steps := readJob(t, vendorWorkflow, "agent").Steps
	claude := stepIndex(t, steps, "claude-code-action", isClaude)

	for _, script := range []string{"check-egress.sh", "stop-docker.sh"} {
		assert.True(t, slices.ContainsFunc(steps[:claude], func(s workflowStep) bool {
			return strings.Contains(s.Run, ".github/reconcile-models/"+script)
		}), "%s must run before Claude starts", script)
	}
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

	// The sleep returns at once, writing the status if it's late.
	path := stubPath(t, map[string]string{
		"curl":  "#!/bin/sh\nexit \"$CURL_EXIT\"\n",
		"sleep": "#!/bin/sh\n[ -z \"$LATE_STATUS\" ] || printf Initialized >\"$LATE_STATUS\"\n",
	})

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

			cmd.Env = append(os.Environ(), path, "CURL_EXIT="+tt.curlExit, "LATE_STATUS="+late)

			out, err := cmd.CombinedOutput()
			assert.Equal(t, tt.passes, err == nil, "%s", out)
		})
	}
}

// stubPath writes stand-in executables to a new directory and returns a PATH
// entry that puts them first. Subtests share one, since a new executable is
// slow to start the first time on macOS.
func stubPath(t *testing.T, bodies map[string]string) string {
	t.Helper()

	dir := t.TempDir()
	for name, body := range bodies {
		require.NoError(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0o700)) //nolint:gosec // stand-ins must be executable
	}

	return "PATH=" + dir + string(os.PathListSeparator) + os.Getenv("PATH")
}

// TestStopDockerFailsWhileDockerAnswers runs stop-docker.sh with a stand-in
// sudo and curl.
func TestStopDockerFailsWhileDockerAnswers(t *testing.T) {
	t.Parallel()

	script := filepath.Join("..", "..", ".github", "reconcile-models", "stop-docker.sh")
	path := stubPath(t, map[string]string{
		"sudo": "#!/bin/sh\nexit \"$SUDO_EXIT\"\n",
		"curl": "#!/bin/sh\nexit \"$CURL_EXIT\"\n",
	})
	tests := []struct {
		name     string
		sudoExit string // 1 is a systemctl that found no Docker units
		curlExit string // 7 is a refused connection; 0 and 52 mean something answered
		passes   bool
	}{
		{name: "stopped", sudoExit: "0", curlExit: "7", passes: true},
		{name: "not installed", sudoExit: "1", curlExit: "7", passes: true},
		{name: "still answering", sudoExit: "0", curlExit: "0", passes: false},
		{name: "something else answering", sudoExit: "0", curlExit: "52", passes: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cmd := exec.CommandContext(t.Context(), "bash", script)

			cmd.Env = append(os.Environ(), path, "SUDO_EXIT="+tt.sudoExit, "CURL_EXIT="+tt.curlExit)

			out, err := cmd.CombinedOutput()
			assert.Equal(t, tt.passes, err == nil, "%s", out)
		})
	}
}

// TestPRJobSandboxHidesDocker checks that pr-job.sh's Linux sandbox hides
// /run, where Docker's socket lives, and that its self-check tries the socket.
func TestPRJobSandboxHidesDocker(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "pr-job.sh"))
	require.NoError(t, err)

	script := string(raw)
	assert.Contains(t, script, "--ro-bind / / --tmpfs /run", "the bwrap sandbox must hide /run")
	assert.Contains(t, script, "--unix-socket /var/run/docker.sock", "the sandbox self-check must try Docker's socket")
}

// TestPRJobRunsBaseCodeOnlyInTheSandbox checks that every `go run` in
// pr-job.sh goes through the sandbox, the base's too: a vendor's base holds
// the earlier vendors' unreviewed commits, and the job later gets the bot
// token.
func TestPRJobRunsBaseCodeOnlyInTheSandbox(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "pr-job.sh"))
	require.NoError(t, err)

	runs := 0

	for line := range strings.Lines(string(raw)) {
		if line = strings.TrimSpace(line); strings.HasPrefix(line, "#") || !strings.Contains(line, "go run") {
			continue
		}

		runs++

		assert.True(t, strings.HasPrefix(line, "sandbox go run "), "pr-job.sh runs code outside the sandbox: %s", line)
	}

	assert.Positive(t, runs, "pr-job.sh has no go run to check")
}

// TestPRJobToolsImportNoPatchableCode checks that the tools pr-job.sh runs
// outside the sandbox import no package a patch can change (the catalogue
// and the providers), so building them from a vendor's base runs none of
// the earlier vendors' code.
func TestPRJobToolsImportNoPatchableCode(t *testing.T) {
	t.Parallel()

	cmd := exec.CommandContext(t.Context(), "go", "list", "-deps", "./cmd/catalog-report", "./cmd/catalog-guard", "./cmd/catalog-replay")
	cmd.Dir = filepath.Join("..", "..")
	out, err := cmd.Output()
	require.NoError(t, err)

	module := "github.com/redpanda-data/ai-sdk-go/"

	for pkg := range strings.Lines(string(out)) {
		pkg = strings.TrimSpace(pkg)
		assert.False(t, pkg == module+"catalog" || strings.HasPrefix(pkg, module+"catalog/") || strings.HasPrefix(pkg, module+"providers/"),
			"a reconcile tool that runs outside the sandbox imports %s", pkg)
	}
}

// gitIn runs git in dir as a throwaway committer and returns its output.
func gitIn(t *testing.T, dir string, args ...string) string {
	t.Helper()

	cmd := exec.CommandContext(t.Context(), "git", append([]string{"-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"}, args...)...) //nolint:gosec // the tests' own git commands
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	require.NoError(t, err, string(out))

	return strings.TrimSpace(string(out))
}

// parseOutputs reads key=value lines, as a step writes them to $GITHUB_OUTPUT.
func parseOutputs(raw string) map[string]string {
	got := map[string]string{}

	for line := range strings.Lines(raw) {
		key, value, _ := strings.Cut(strings.TrimSpace(line), "=")
		got[key] = value
	}

	return got
}

// TestCombineBuildsThePRFromTheVendors runs combine.sh, the pr job's step that
// puts the vendors' parts together with no model: the last vendor commit is
// the PR's tip, the body covers every vendor that ran, a vendor counts once
// its check job handed on its commit, and a failed vendor with a commit
// waiting in the PR fails the job, or its change would drop out.
func TestCombineBuildsThePRFromTheVendors(t *testing.T) {
	t.Parallel()

	report := buildCatalogReport(t)
	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "combine.sh"))
	require.NoError(t, err)

	// Paths under the test's temp folders and the repo's own files.
	write := func(path, content string) {
		require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o700))     //nolint:gosec // see above
		require.NoError(t, os.WriteFile(path, []byte(content), 0o600)) //nolint:gosec // see above
	}
	copyFile := func(from, to string) {
		raw, err := os.ReadFile(from) //nolint:gosec // see above
		require.NoError(t, err)
		write(to, string(raw))
	}

	// The run's base: two vendors' named sources, the real rules and an empty catalogue.
	origin := t.TempDir()
	write(filepath.Join(origin, ".github", "reconcile-models", "named-sources.json"), `{"anthropic": {"providers": ["anthropic"], "hosts": ["platform.claude.com"]},
		"openai": {"providers": ["openai"], "hosts": ["developers.openai.com"]}}`)

	for _, path := range []string{reconcile.RulesPath, reconcile.DecisionsPath} {
		copyFile(filepath.Join("..", "..", path), filepath.Join(origin, path))
	}

	write(filepath.Join(origin, "catalog", "snapshot.json"), `{"providers": []}`+"\n")
	gitIn(t, origin, "init", "-q")
	gitIn(t, origin, "add", ".")
	gitIn(t, origin, "commit", "-q", "-m", "main")
	base := gitIn(t, origin, "rev-parse", "HEAD")

	// Each vendor commits on the one before and bundles its commit, as its
	// check job does, and leaves its folder.
	work, bundles, legs, quiet := t.TempDir(), t.TempDir(), t.TempDir(), t.TempDir()
	gitIn(t, work, "clone", "-q", "file://"+origin, ".")

	catalogue := func(ids ...string) string {
		offerings := make([]string, 0, len(ids))
		for _, id := range ids {
			offerings = append(offerings, `{"id": "`+id+`", "model": "x/`+id+`"}`)
		}

		return `{"providers": [{"provider": "openai", "offerings": [` + strings.Join(offerings, ", ") + `]}]}` + "\n"
	}
	const route = "us-east-1 anthropic.claude-sonnet-5-5 us.anthropic.claude-sonnet-5-5\n"
	vendor := func(name, offering, host, before, after string) string {
		write(filepath.Join(work, "catalog", "snapshot.json"), after)
		gitIn(t, work, "commit", "-q", "-am", "chore(catalog): reconcile "+name+" models")
		sha := gitIn(t, work, "rev-parse", "HEAD")
		require.NoError(t, os.MkdirAll(filepath.Join(bundles, "reconcile-bundle-"+sha), 0o700))
		gitIn(t, work, "bundle", "create", "-q", filepath.Join(bundles, "reconcile-bundle-"+sha, "reconcile.bundle"), "HEAD", "^"+base)

		vendorFolder(t, filepath.Join(legs, "reconcile-pr-"+name), priced(offering, "https://"+host+"/pricing"), map[string]string{
			"files.txt": "providers/" + name + "/models.go\ncatalog/snapshot.json\n", "code-files.txt": "",
			"base-snapshot.json": before, "snapshot.json": after, "base-routes.txt": route, "routes.txt": route,
		})
		vendorFolder(t, filepath.Join(quiet, "reconcile-pr-"+name), Report{Changes: []Change{}, NeedsHuman: []NeedsHuman{}, Skipped: []Skipped{}}, nil)

		return sha
	}
	anthropic := vendor("anthropic", "claude-opus-5-5", "platform.claude.com", catalogue(), catalogue("gpt-6-1"))
	openai := vendor("openai", "gpt-6-1", "developers.openai.com", catalogue("gpt-6-1"), catalogue("gpt-6-1", "gpt-6-2"))

	part := func(result, head, changed string) string {
		return `{"result": "` + result + `", "outputs": {"head": "` + head + `", "changed": "` + changed + `", "agent_run": "4100", "agent_at": "2026-10-28T07:00:00Z"}}`
	}
	failed := `{"result": "failure", "outputs": {}}`

	tests := []struct {
		name, legs, anthropic, openai, pending, wantErr string
		want                                            map[string]string
		body                                            []string
		files                                           string
		// record is each vendor's commit, its parent and its agent run in
		// the record.
		record map[string][3]string
	}{
		{
			name: "two vendors' commits", legs: legs, anthropic: part("success", anthropic, "true"), openai: part("success", openai, "true"), pending: "[]",
			want:   map[string]string{"tip": openai, "changed": "true"},
			body:   []string{"## Catalogue drift: anthropic, openai\n", "\n## anthropic\n", "\n## openai\n", "<details><summary>Snapshot diff</summary>", `+{"providers": [{"provider": "openai", "offerings": [{"id": "gpt-6-1", "model": "x/gpt-6-1"}, {"id": "gpt-6-2"`},
			files:  "catalog/snapshot.json\nproviders/anthropic/models.go\nproviders/openai/models.go\n",
			record: map[string][3]string{"anthropic": {anthropic, base, "4100"}, "openai": {openai, anthropic, "4100"}},
		},
		{
			name: "a failed vendor with nothing waiting is left out", legs: legs, anthropic: part("success", anthropic, "true"), openai: failed, pending: `["anthropic"]`,
			want:   map[string]string{"tip": anthropic, "changed": "true"},
			body:   []string{"\n## openai\n\nThis vendor's run failed"},
			files:  "catalog/snapshot.json\nproviders/anthropic/models.go\n",
			record: map[string][3]string{"anthropic": {anthropic, base, "4100"}},
		},
		{
			name: "a replayed vendor", legs: legs, anthropic: part("success", anthropic, "true"),
			openai: `{"result": "success", "outputs": {"head": "` + openai + `", "changed": "true", "replayed": "true", "agent_run": "40504204648", "agent_at": "2026-10-27T06:17:00Z"}}`, pending: `["openai"]`,
			want:   map[string]string{"tip": openai, "changed": "true"},
			body:   []string{"\n## openai\n\n> Replayed with no model: this is the change the agent made in run 40504204648 on 2026-10-27"},
			files:  "catalog/snapshot.json\nproviders/anthropic/models.go\nproviders/openai/models.go\n",
			record: map[string][3]string{"anthropic": {anthropic, base, "4100"}, "openai": {openai, anthropic, "40504204648"}},
		},
		{
			// Only the record job, after the check, failed: it keeps the
			// gate's baseline, so the checked commit still goes in.
			name: "a vendor whose check finished counts", legs: legs, anthropic: part("success", anthropic, "true"), openai: part("failure", openai, "true"), pending: "[]",
			want:   map[string]string{"tip": openai, "changed": "true"},
			files:  "catalog/snapshot.json\nproviders/anthropic/models.go\nproviders/openai/models.go\n",
			record: map[string][3]string{"anthropic": {anthropic, base, "4100"}, "openai": {openai, anthropic, "4100"}},
		},
		{
			name: "a failed vendor with a waiting commit fails the job", legs: legs, anthropic: part("success", anthropic, "true"), openai: failed, pending: `["openai"]`,
			wantErr: "openai failed and has a commit waiting in the PR",
		},
		{
			name: "nothing changed", legs: quiet, anthropic: part("success", base, "false"), openai: part("success", base, "false"), pending: "[]",
			want:  map[string]string{"tip": base, "changed": "false"},
			body:  []string{"## Catalogue drift: anthropic, openai\n\nNo drift found.\n"},
			files: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			checkout, out := t.TempDir(), t.TempDir()
			gitIn(t, checkout, "clone", "-q", "--depth", "1", "file://"+origin, ".")
			outputs := filepath.Join(t.TempDir(), "outputs")

			cmd := exec.CommandContext(t.Context(), "bash", script, report, tt.legs, bundles, out)
			cmd.Dir = checkout

			cmd.Env = append(os.Environ(), "GITHUB_OUTPUT="+outputs, "GITHUB_SHA="+base, `RUN=["anthropic","openai"]`, "PENDING="+tt.pending,
				`NEEDS={"plan": {"result": "success"}, "anthropic": `+tt.anthropic+`, "openai": `+tt.openai+`}`)
			combined, err := cmd.CombinedOutput()

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, string(combined), tt.wantErr)

				return
			}

			require.NoError(t, err, string(combined))

			raw, err := os.ReadFile(outputs)
			require.NoError(t, err)

			got := parseOutputs(string(raw))
			for key, want := range tt.want {
				assert.Equal(t, want, got[key], key)
			}

			assert.Equal(t, got["tip"], gitIn(t, checkout, "rev-parse", "HEAD"), "the checkout is at the tip the push sends")

			body, err := os.ReadFile(filepath.Join(out, "body.md"))
			require.NoError(t, err)

			for _, want := range tt.body {
				assert.Contains(t, string(body), want)
			}

			files, err := os.ReadFile(filepath.Join(out, "files.txt"))
			require.NoError(t, err)
			assert.Equal(t, tt.files, string(files))

			// The record a later run replays from: what this run pushes.
			raw, err = os.ReadFile(filepath.Join(out, "record.json"))
			if tt.record == nil {
				assert.ErrorIs(t, err, os.ErrNotExist, "a run that pushes nothing leaves no record")

				return
			}

			require.NoError(t, err)

			var record struct {
				Head    string `json:"head"`
				Vendors map[string]struct {
					Commit   string `json:"commit"`
					Parent   string `json:"parent"`
					AgentRun string `json:"agent_run"`
					AgentAt  string `json:"agent_at"`
					Report   Report `json:"report"`
				} `json:"vendors"`
			}
			require.NoError(t, json.Unmarshal(raw, &record))
			assert.Equal(t, got["tip"], record.Head)
			assert.ElementsMatch(t, slices.Collect(maps.Keys(tt.record)), slices.Collect(maps.Keys(record.Vendors)))

			for name, want := range tt.record {
				vendor := record.Vendors[name]
				assert.Equal(t, want[0], vendor.Commit, "%s's commit", name)
				assert.Equal(t, want[1], vendor.Parent, "%s's parent", name)
				assert.Equal(t, want[2], vendor.AgentRun, "%s's agent run", name)
				assert.NotEmpty(t, vendor.AgentAt, "%s's agent time", name)
				assert.NotEmpty(t, vendor.Report.Changes, "%s's report", name)
			}
		})
	}
}

// TestCheckoutBaseTakesTheBundlesCommit checks how a vendor's part of the run
// gets its base: the run's own commit as checked out, or the previous
// vendor's commit from its bundle, never anything else.
func TestCheckoutBaseTakesTheBundlesCommit(t *testing.T) {
	t.Parallel()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "checkout-base.sh"))
	require.NoError(t, err)

	// origin has the run's commit; a vendor's clone adds its commit and
	// bundles it, as the check job does.
	origin, leg := t.TempDir(), t.TempDir()
	gitIn(t, origin, "init", "-q")
	gitIn(t, origin, "commit", "-q", "--allow-empty", "-m", "main")
	runSHA := gitIn(t, origin, "rev-parse", "HEAD")
	gitIn(t, leg, "clone", "-q", "--depth", "1", "file://"+origin, ".")
	gitIn(t, leg, "commit", "-q", "--allow-empty", "-m", "vendor")
	legSHA := gitIn(t, leg, "rev-parse", "HEAD")
	bundle := filepath.Join(t.TempDir(), "reconcile.bundle")
	gitIn(t, leg, "bundle", "create", "-q", bundle, "HEAD", "^"+runSHA)

	tests := []struct {
		name, base, bundle, wantHead, wantErr string
	}{
		{name: "the run's own commit needs no bundle", base: runSHA, wantHead: runSHA},
		{name: "the previous vendor's commit from its bundle", base: legSHA, bundle: bundle, wantHead: legSHA},
		{name: "no bundle for another commit", base: legSHA, wantErr: "no bundle for " + legSHA},
		{name: "a bundle that holds a different commit", base: strings.Repeat("0", 40), bundle: bundle, wantErr: "the bundle holds " + legSHA},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			checkout := t.TempDir()
			gitIn(t, checkout, "clone", "-q", "--depth", "1", "file://"+origin, ".")

			cmd := exec.CommandContext(t.Context(), "bash", script, tt.base, tt.bundle)
			cmd.Dir = checkout
			out, err := cmd.CombinedOutput()

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, string(out), tt.wantErr)
				assert.Equal(t, runSHA, gitIn(t, checkout, "rev-parse", "HEAD"), "a refused base leaves the checkout as it was")

				return
			}

			require.NoError(t, err, string(out))
			assert.Equal(t, tt.wantHead, gitIn(t, checkout, "rev-parse", "HEAD"))
		})
	}
}

// TestReplayRebuildsTheWaitingCommit runs replay.sh, which rebuilds a
// vendor's waiting commit on a newer base with no model: a 3-way merge from
// the commit's own parent, the same checks as an agent's patch (here a
// stand-in pr-job.sh), and the exact replay check. It hands over the patch
// and the saved report only when all pass, and always leaves the checkout
// as it found it.
func TestReplayRebuildsTheWaitingCommit(t *testing.T) {
	t.Parallel()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "replay.sh"))
	require.NoError(t, err)

	replayBin := filepath.Join(t.TempDir(), "catalog-replay")
	build := exec.CommandContext(t.Context(), "go", "build", "-o", replayBin, "../catalog-replay")
	out, err := build.CombinedOutput()
	require.NoError(t, err, string(out))

	// A stand-in catalogue: one "id rate" line per offering, far enough
	// apart that a change to one is outside the other's diff context, and
	// a generator that writes the snapshot from it.
	models := func(opus, sonnet string) string {
		return "claude-opus-5-5 " + opus + "\n#\n#\n#\n#\n#\n#\nclaude-sonnet-5-5 " + sonnet + "\n"
	}
	const generate = `#!/bin/sh
printf '{"facts": {}, "providers": [{"provider": "anthropic", "offerings": ['
grep -v '^#' providers/anthropic/models.txt | awk 'NR>1{printf ", "} {printf "{\"id\": \"%s\", \"model\": \"anthropic/%s\", \"input\": %s}", $1, $1, $2}'
printf ']}]}\n'
`
	// pr-job.sh's part here: apply the patch, regenerate, list the files.
	prJob := filepath.Join(t.TempDir(), "pr-job.sh")
	require.NoError(t, os.WriteFile(prJob, []byte("#!/bin/sh\nset -e\ngit apply \"$2/patch.diff\"\n./gen.sh >catalog/snapshot.json\ngit diff --name-only >\"$2/files.txt\"\n"), 0o700)) //nolint:gosec // a stand-in must be executable

	write := func(dir, path, content string, mode os.FileMode) {
		require.NoError(t, os.MkdirAll(filepath.Join(dir, filepath.Dir(path)), 0o700))
		require.NoError(t, os.WriteFile(filepath.Join(dir, path), []byte(content), mode))
	}
	snapshot := func(dir string) {
		cmd := exec.CommandContext(t.Context(), "./gen.sh")
		cmd.Dir = dir
		raw, err := cmd.Output()
		require.NoError(t, err)
		write(dir, "catalog/snapshot.json", string(raw), 0o600)
	}
	commit := func(dir, opus, sonnet, message string) string {
		write(dir, "providers/anthropic/models.txt", models(opus, sonnet), 0o600)
		snapshot(dir)
		gitIn(t, dir, "add", ".")
		gitIn(t, dir, "commit", "-q", "-m", message)

		return gitIn(t, dir, "rev-parse", "HEAD")
	}

	// origin: main's commit P, the bot's commit B on it, and main's newer
	// commits, each a possible base.
	origin := t.TempDir()
	gitIn(t, origin, "init", "-q")
	write(origin, "gen.sh", generate, 0o700)
	parent := commit(origin, "5", "3", "main")
	bot := commit(origin, "6", "3", "bot: opus 5 to 6")
	gitIn(t, origin, "branch", "bot")
	gitIn(t, origin, "reset", "-q", "--hard", parent)

	bases := map[string]string{}

	for i, base := range []struct{ name, opus, sonnet string }{{"sonnet moved", "5", "4"}, {"opus moved", "7", "3"}} {
		bases[base.name] = commit(origin, base.opus, base.sonnet, "main: "+base.name)
		gitIn(t, origin, "branch", "base"+strconv.Itoa(i))
		gitIn(t, origin, "reset", "-q", "--hard", parent)
	}

	report := `{"changes": [{"offering": "claude-opus-5-5", "field": "price", "old": "5", "new": "6", "source_url": "https://platform.claude.com/pricing", "rule": "price-follow"}], "needs_human": [], "skipped": []}`
	asked := `{"changes": [{"offering": "claude-opus-5-5", "field": "price", "old": "5", "new": "6", "source_url": "https://platform.claude.com/pricing", "rule": "price-follow"}],
		"needs_human": [{"offering": "claude-sonnet-5-5", "field": "price", "catalogue": "", "page": "", "values": [], "source_urls": [], "rule": "no-rule"}], "skipped": []}`
	record := func(vendor, parent, report string) string {
		return `{"head": "` + bot + `", "vendors": {"` + vendor + `": {"commit": "` + bot + `", "parent": "` + parent + `", "agent_run": "40504204648", "agent_at": "2026-10-27T06:17:00Z", "report": ` + report + `}}}`
	}

	tests := []struct {
		name, base, record, wantErr string
	}{
		{name: "onto a base that changed another offering", base: "sonnet moved", record: record("anthropic", parent, report)},
		{name: "onto a base that changed the same line", base: "opus moved", record: record("anthropic", parent, report), wantErr: "doesn't merge"},
		{name: "a question about an offering the base changed", base: "sonnet moved", record: record("anthropic", parent, asked), wantErr: "a report row names it"},
		{name: "no commit for the vendor", base: "sonnet moved", record: record("openai", parent, report), wantErr: "no commit for anthropic"},
		{name: "a commit that isn't on its parent", base: "sonnet moved", record: record("anthropic", bases["opus moved"], report), wantErr: "isn't a commit on"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			checkout, out, outputs := t.TempDir(), t.TempDir(), filepath.Join(t.TempDir(), "outputs")
			gitIn(t, checkout, "clone", "-q", "file://"+origin, ".")
			gitIn(t, checkout, "checkout", "-q", "--detach", bases[tt.base])
			recordFile := filepath.Join(t.TempDir(), "record.json")
			write(filepath.Dir(recordFile), filepath.Base(recordFile), tt.record, 0o600)

			cmd := exec.CommandContext(t.Context(), "bash", script, "anthropic", recordFile, out, replayBin)
			cmd.Dir = checkout

			cmd.Env = append(os.Environ(), "PR_JOB="+prJob, "GITHUB_OUTPUT="+outputs)
			combined, err := cmd.CombinedOutput()

			assert.Equal(t, bases[tt.base], gitIn(t, checkout, "rev-parse", "HEAD"), "the checkout is back at its base")
			assert.Empty(t, gitIn(t, checkout, "status", "--porcelain"), "the checkout is clean")

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, string(combined), tt.wantErr)
				assert.NoFileExists(t, filepath.Join(out, "patch.diff"), "a failed replay hands over nothing")

				return
			}

			require.NoError(t, err, string(combined))

			patch, err := os.ReadFile(filepath.Join(out, "patch.diff"))
			require.NoError(t, err)
			assert.Contains(t, string(patch), "+claude-opus-5-5 6")
			assert.NotContains(t, string(patch), "snapshot.json", "pr-job.sh regenerates the snapshot")

			saved, err := os.ReadFile(filepath.Join(out, "report.json"))
			require.NoError(t, err)
			assert.JSONEq(t, report, string(saved))

			raw, err := os.ReadFile(outputs)
			require.NoError(t, err)
			assert.Equal(t, map[string]string{"agent_run": "40504204648", "agent_at": "2026-10-27T06:17:00Z"}, parseOutputs(string(raw)))
		})
	}
}

// TestPRJobRefusesPatchesOutsideTheCatalogue runs pr-job.sh on patches it
// must refuse before any network step or patched code runs: the checks that
// keep a hijacked agent's patch away from the bot token.
func TestPRJobRefusesPatchesOutsideTheCatalogue(t *testing.T) {
	t.Parallel()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "pr-job.sh"))
	require.NoError(t, err)

	report := `{"changes": [{"offering": "claude-opus-5-5"}], "needs_human": [], "skipped": []}`
	tests := []struct {
		name   string
		path   string
		change string // "edit", "create" or "delete"
		want   string
	}{
		{name: "a workflow", path: ".github/workflows/test.yaml", change: "edit", want: "the patch touches .github/workflows/test.yaml"},
		{name: "another vendor's provider", path: "providers/openai/models.go", change: "edit", want: "the patch touches providers/openai/models.go"},
		{name: "a deleted file", path: "providers/anthropic/models.go", change: "delete", want: "the patch deletes, renames or copies files"},
		{name: "a new file outside providers", path: "cmd/evil/main.go", change: "create", want: "the patch creates cmd/evil/main.go"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			out := t.TempDir()
			require.NoError(t, os.WriteFile(filepath.Join(out, "report.json"), []byte(report), 0o600))
			require.NoError(t, os.WriteFile(filepath.Join(out, "patch.diff"), []byte(onePatch(tt.path, tt.change)), 0o600))

			cmd := exec.CommandContext(t.Context(), "bash", script, "anthropic", out)
			cmd.Dir = filepath.Join("..", "..")

			stderr, err := cmd.CombinedOutput()
			require.Error(t, err, "pr-job.sh accepted the patch: %s", stderr)
			assert.Contains(t, string(stderr), "pr-job: "+tt.want)
		})
	}
}

// onePatch is a one-line git patch that edits, creates or deletes path.
func onePatch(path, change string) string {
	header := "diff --git a/" + path + " b/" + path + "\n"

	switch change {
	case "create":
		return header + "new file mode 100644\n--- /dev/null\n+++ b/" + path + "\n@@ -0,0 +1 @@\n+x\n"
	case "delete":
		return header + "deleted file mode 100644\n--- a/" + path + "\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
	default:
		return header + "--- a/" + path + "\n+++ b/" + path + "\n@@ -1 +1 @@\n-x\n+y\n"
	}
}

// TestAgentCannotReadProcessFiles checks that the agent's file tools are denied
// /proc and that its shell commands run with the job's secrets scrubbed.
func TestAgentCannotReadProcessFiles(t *testing.T) {
	t.Parallel()

	job := readJob(t, vendorWorkflow, "agent")
	assert.Equal(t, "1", job.Env["CLAUDE_CODE_SUBPROCESS_ENV_SCRUB"], "the agent's shell must run without the job's secrets")

	args := job.Steps[stepIndex(t, job.Steps, "claude-code-action", isClaude)].With["claude_args"]
	disallowed := regexp.MustCompile(`--disallowedTools "([^"]*)"`).FindStringSubmatch(args)
	require.NotNil(t, disallowed, "the Claude step passes no --disallowedTools list")
	assert.Contains(t, strings.Split(disallowed[1], ","), "Read(//proc/**)")
}

// TestPlanPicksTheVendors runs plan.sh, which decides before any agent starts
// which vendors run their agent and which replay their waiting commit, from
// the record the last push left, and whether someone else's commit on the
// PR stops the run.
func TestPlanPicksTheVendors(t *testing.T) {
	t.Parallel()

	script := filepath.Join("..", "..", ".github", "reconcile-models", "plan.sh")
	bot := func(sha, vendor string) string {
		return `{"sha": "` + sha + `", "author": {"login": "vbotbuildovich"}, "committer": {"login": "vbotbuildovich"},
			"commit": {"message": "chore(catalog): reconcile ` + vendor + ` models\n\nReconcile-Run: 40504204648/` + vendor + `"}}`
	}
	person := `{"sha": "b2c3d4e5f60718293a4b5c6d7e8f900112233445", "author": {"login": "adamwownysz-redpanda"}, "committer": {"login": "web-flow"},
		"commit": {"message": "test: swap the live test model"}}`
	// The bot's commits from an earlier run, for Anthropic and for OpenAI.
	const anthropic, openai = "a1b2c3d4e5f60718293a4b5c6d7e8f9001122334", "a2b3c4d5e6f70819203a4b5c6d7e8f9001122334"
	waiting := "[" + bot(anthropic, "anthropic") + "," + bot(openai, "openai") + "]"
	// The record that run left: the head it pushed, and each vendor's
	// commit and agent run, 26 hours before the plan runs.
	record := filepath.Join(t.TempDir(), "record.json")
	require.NoError(t, os.WriteFile(record, []byte(`{"head": "`+openai+`", "vendors": {
		"anthropic": {"commit": "`+anthropic+`", "parent": "9f8e7d6c5b4a39281706f5e4d3c2b1a098765432", "agent_run": "40504204648", "agent_at": "2026-10-27T06:17:00Z", "report": {}},
		"openai": {"commit": "`+openai+`", "parent": "`+anthropic+`", "agent_run": "40504204648", "agent_at": "2026-10-27T06:17:00Z", "report": {}}}}`), 0o600))
	undated := filepath.Join(t.TempDir(), "record.json")
	require.NoError(t, os.WriteFile(undated, []byte(`{"head": "`+openai+`", "vendors": {
		"anthropic": {"commit": "`+anthropic+`", "agent_at": "2026-10-27T06:17:00Z"}, "openai": {"commit": "`+openai+`"}}}`), 0o600))
	stranger := filepath.Join(t.TempDir(), "record.json")
	require.NoError(t, os.WriteFile(stranger, []byte(`{"head": "c0ffee0000000000000000000000000000000000", "vendors": {}}`), 0o600))

	const now = "2026-10-28T08:00:00Z"

	tests := []struct {
		name, selected, dryRun, commits, record, now string
		want                                         map[string]string
	}{
		{
			name: "a weekly run runs every vendor", selected: "all", dryRun: "false", commits: waiting,
			want: map[string]string{"run": `["anthropic","openai","google"]`, "agent": `["anthropic","openai","google"]`, "replay": "[]", "pending": `["anthropic","openai"]`, "stop": "false", "theirs": "", "record": "false"},
		},
		{
			name: "a weekly run with a record runs every agent", selected: "all", dryRun: "false", commits: waiting, record: record, now: now,
			want: map[string]string{"run": `["anthropic","openai","google"]`, "agent": `["anthropic","openai","google"]`, "replay": "[]", "carry": "[]", "stop": "false"},
		},
		{
			name: "a real run for one vendor replays the waiting ones", selected: "google", dryRun: "false", commits: waiting, record: record, now: now,
			want: map[string]string{"run": `["anthropic","openai","google"]`, "agent": `["google"]`, "replay": `["anthropic","openai"]`, "pending": `["anthropic","openai"]`, "stop": "false", "record": "true"},
		},
		{
			// A dry run previews what a real run would push.
			name: "a dry run for one vendor replays the others", selected: "openai", dryRun: "true", commits: waiting, record: record, now: now,
			want: map[string]string{"run": `["anthropic","openai"]`, "agent": `["openai"]`, "replay": `["anthropic"]`},
		},
		{
			// Past MAX_AGE a waiting change is stale, so its agent runs
			// again instead of carrying it forward.
			name: "a record older than a week runs the agents again", selected: "google", dryRun: "false", commits: waiting, record: record, now: "2026-11-05T07:00:00Z",
			want: map[string]string{"run": `["anthropic","openai","google"]`, "agent": `["anthropic","openai","google"]`, "replay": "[]"},
		},
		{
			// Until the carry job runs, a commit after the record's head
			// stops a real run, as without a record.
			name: "a person's commit after the record's head", selected: "google", dryRun: "false", commits: "[" + bot(anthropic, "anthropic") + "," + bot(openai, "openai") + "," + person + "]", record: record, now: now,
			want: map[string]string{"run": "[]", "stop": "true", "theirs": "b2c3d4e", "carry": `["b2c3d4e5f60718293a4b5c6d7e8f900112233445"]`},
		},
		{
			name: "a vendor the record gives no agent time runs its agent", selected: "google", dryRun: "false", commits: waiting, record: undated, now: now,
			want: map[string]string{"agent": `["openai","google"]`, "replay": `["anthropic"]`},
		},
		{
			name: "a record whose head isn't the PR's counts as none", selected: "google", dryRun: "false", commits: waiting, record: stranger, now: now,
			want: map[string]string{"run": `["anthropic","openai","google"]`, "agent": `["anthropic","openai","google"]`, "replay": "[]", "record": "false"},
		},
		{
			name: "a dry run for one vendor runs only that one", selected: "google", dryRun: "true", commits: waiting,
			want: map[string]string{"run": `["google"]`, "stop": "false"},
		},
		{
			// Every run rebuilds the PR, so a real run also runs each vendor
			// whose commit is waiting, or its change would drop out.
			name: "a real run for one vendor also runs the waiting ones", selected: "google", dryRun: "false", commits: waiting,
			want: map[string]string{"run": `["anthropic","openai","google"]`, "pending": `["anthropic","openai"]`},
		},
		{
			name: "no open PR", selected: "openai", dryRun: "false", commits: "[]",
			want: map[string]string{"run": `["openai"]`, "pending": "[]", "stop": "false"},
		},
		{
			name: "someone else's commit stops a real run before any agent", selected: "all", dryRun: "false", commits: "[" + bot("a1b2c3d4e5f60718293a4b5c6d7e8f9001122334", "anthropic") + "," + person + "]",
			want: map[string]string{"run": "[]", "stop": "true", "theirs": "b2c3d4e"},
		},
		{
			name: "a dry run goes on past someone else's commit", selected: "all", dryRun: "true", commits: "[" + person + "]",
			want: map[string]string{"run": `["anthropic","openai","google"]`, "stop": "false", "theirs": "b2c3d4e"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cmd := exec.CommandContext(t.Context(), "bash", script, "vbotbuildovich", tt.selected, tt.dryRun, `["anthropic","openai","google"]`, tt.record)
			cmd.Env = append(os.Environ(), "NOW="+tt.now)
			cmd.Stdin = strings.NewReader(tt.commits)
			out, err := cmd.Output()
			require.NoError(t, err)

			got := parseOutputs(string(out))
			for key, want := range tt.want {
				assert.Equal(t, want, got[key], key)
			}
		})
	}

	t.Run("a vendor named-sources.json doesn't list", func(t *testing.T) {
		t.Parallel()

		cmd := exec.CommandContext(t.Context(), "bash", script, "vbotbuildovich", "xai", "true", `["anthropic"]`)
		cmd.Stdin = strings.NewReader("[]")
		out, err := cmd.CombinedOutput()
		require.Error(t, err)
		assert.Contains(t, string(out), `no named sources for "xai"`)
	})
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

// TestChainHasOneLegPerVendor checks that every vendor in named-sources.json
// has its part in the run, and nothing else does, so adding a vendor's
// sources without its job (or the other way round) fails here.
func TestChainHasOneLegPerVendor(t *testing.T) {
	t.Parallel()

	found := legs(t)
	assert.ElementsMatch(t, vendorOrder(t), slices.Collect(maps.Keys(found)))

	for name, job := range found {
		assert.Equal(t, name, job.With["vendor"], "the %s job runs another vendor", name)
	}
}

// TestChainPassesTheBaseBySHA checks the chain: each vendor needs every vendor
// before it, runs even after one fails, and works on the newest commit an
// earlier vendor made, by SHA, else the run's own commit. A branch name
// would leave a vendor on last week's PR.
func TestChainPassesTheBaseBySHA(t *testing.T) {
	t.Parallel()

	jobs := readWorkflow(t, runWorkflow).Jobs
	order := vendorOrder(t)

	for i, vendor := range order {
		job := jobs[vendor]
		assert.Equal(t, append([]string{"plan"}, order[:i]...), needsOf(job), "%s's needs", vendor)

		heads := []string{"github.sha"}
		for _, earlier := range order[:i] {
			heads = slices.Insert(heads, 0, "needs."+earlier+".outputs.head")
		}

		assert.Equal(t, "${{ "+strings.Join(heads, " || ")+" }}", job.With["base_sha"], "%s's base", vendor)

		for _, part := range []string{"!cancelled()", "needs.plan.result == 'success'", "contains(fromJSON(needs.plan.outputs.run), '" + vendor + "')"} {
			assert.Contains(t, job.If, part, "%s's condition", vendor)
		}
	}

	assert.Equal(t, append([]string{"plan"}, order...), needsOf(jobs["pr"]), "the pr job waits for every vendor")
	assert.Contains(t, jobs["pr"].If, "!cancelled()", "the pr job runs after a vendor fails, to push the others")
}

// TestDispatchOptionsMatchNamedSources checks that a run started by hand can
// pick every vendor, or all of them, and nothing else.
func TestDispatchOptionsMatchNamedSources(t *testing.T) {
	t.Parallel()

	input := readWorkflow(t, runWorkflow).On.WorkflowDispatch.Inputs["vendor"]
	assert.Equal(t, append([]string{"all"}, vendorOrder(t)...), input.Options)
	assert.Equal(t, "all", input.Default)
}

// TestOnlyThePRJobGetsTheBotToken checks that the pr job, which runs nothing
// from the patches, is the only one in the run that can get an OIDC token
// and the bot token.
func TestOnlyThePRJobGetsTheBotToken(t *testing.T) {
	t.Parallel()

	for name, job := range readWorkflow(t, runWorkflow).Jobs {
		aws := slices.ContainsFunc(job.Steps, func(s workflowStep) bool { return strings.HasPrefix(s.Uses, "aws-actions/") })
		if name == "pr" {
			assert.Equal(t, "write", job.Permissions["id-token"])
			assert.True(t, aws, "the pr job reads the bot token")

			continue
		}

		assert.NotContains(t, job.Permissions, "id-token", "the %s job may not request an OIDC token", name)
		assert.False(t, aws, "the %s job may not reach the bot token", name)
	}
}

// TestPushIsLeasedOnThePlansHead checks that the run's one push is leased to
// the PR head the plan job read before any agent started, after checking the
// PR still has that head, so a commit pushed during the run is never lost
// and a PR closed during the run stays closed.
func TestPushIsLeasedOnThePlansHead(t *testing.T) {
	t.Parallel()

	job := readJob(t, runWorkflow, "pr")
	assert.Equal(t, "${{ needs.plan.outputs.h0 }}", job.Env["H0"])

	run := job.Steps[stepIndex(t, job.Steps, "push", func(s workflowStep) bool { return strings.Contains(s.Run, "git push") })].Run
	assert.Contains(t, run, `git push --force-with-lease="refs/heads/${BRANCH}:${H0}"`)
	assert.NotContains(t, run, "git push --force ", "the push must be leased, not a plain force-push")

	for _, check := range []string{"headRefOid", "--json state"} {
		at := strings.Index(run, check)
		assert.NotEqual(t, -1, at, "the push step doesn't check the PR's %s", check)
		assert.Less(t, at, strings.Index(run, "git push"), "the PR's %s must be checked before the push", check)
	}
}

// TestRerunFindsThePRItOpened checks that a re-run of the pr job, after its
// first attempt opened the PR, edits that PR instead of trying to open it
// again: the plan's PR output is from before the PR existed, so the push
// step looks the bot's PR up again before it decides.
func TestRerunFindsThePRItOpened(t *testing.T) {
	t.Parallel()

	steps := readJob(t, runWorkflow, "pr").Steps
	run := steps[stepIndex(t, steps, "push", func(s workflowStep) bool { return strings.Contains(s.Run, "git push") })].Run
	lookup := strings.Index(run, `PR="${PR:-$(gh pr list`)
	require.NotEqual(t, -1, lookup, "the push step must look the PR up when the plan found none")
	assert.Less(t, lookup, strings.Index(run, "gh pr view"))
	assert.Less(t, lookup, strings.Index(run, "gh pr create"))
}

// TestOthersCommitsRunsBeforeTheChain checks that the plan job looks for other
// people's commits on the bot PR before any vendor's agent starts, and that a
// PR left alone keeps one note, updated each run.
func TestOthersCommitsRunsBeforeTheChain(t *testing.T) {
	t.Parallel()

	plan := readJob(t, runWorkflow, "plan")
	assert.True(t, slices.ContainsFunc(plan.Steps, func(s workflowStep) bool { return strings.Contains(s.Run, ".github/reconcile-models/plan.sh") }),
		"the plan job must run plan.sh")

	steps := readJob(t, runWorkflow, "pr").Steps
	note := steps[stepIndex(t, steps, "note", func(s workflowStep) bool { return strings.Contains(s.Run, "gh pr comment") })]
	assert.Contains(t, note.Run, "--edit-last --create-if-none", "the note must edit its earlier one")
	assert.Contains(t, note.If, "env.STOP == 'true'")
}

// TestOutputsReadAreDeclared checks that every job output a workflow reads
// (needs.<job>.outputs.<name>, or jobs.<job>.outputs.<name> in a called
// workflow's outputs) is one that job declares. GitHub reads an undeclared
// output as empty without an error.
func TestOutputsReadAreDeclared(t *testing.T) {
	t.Parallel()

	read := regexp.MustCompile(`(?:needs|jobs)\.([a-z0-9_-]+)\.outputs\.([a-z0-9_]+)`)
	called := readWorkflow(t, vendorWorkflow).On.WorkflowCall.Outputs

	for _, file := range []string{runWorkflow, vendorWorkflow} {
		raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", file))
		require.NoError(t, err)

		jobs := readWorkflow(t, file).Jobs
		for _, match := range read.FindAllStringSubmatch(string(raw), -1) {
			job, found := jobs[match[1]]
			require.True(t, found, "%s reads outputs of a job it doesn't have: %s", file, match[0])

			if job.Uses == "./.github/workflows/"+vendorWorkflow {
				assert.Contains(t, called, match[2], "%s reads %s, which %s doesn't output", file, match[0], vendorWorkflow)

				continue
			}

			assert.Contains(t, job.Outputs, match[2], "%s reads %s, which the job doesn't declare", file, match[0])
		}
	}
}

// TestPlanReadsTheHeadItLeases checks that the plan job fails when the PR's
// commit list doesn't end at the head the push is leased to: GitHub updates
// the list after a push, so a commit pushed just before the run could be
// left out of the plan yet pass the lease.
func TestPlanReadsTheHeadItLeases(t *testing.T) {
	t.Parallel()

	steps := readJob(t, runWorkflow, "plan").Steps
	run := steps[stepIndex(t, steps, "plan.sh", func(s workflowStep) bool { return strings.Contains(s.Run, "plan.sh") })].Run
	check := strings.Index(run, `"$h0" != "$(jq -r 'last.sha // empty' <<<"$commits")"`)
	require.NotEqual(t, -1, check, "the plan must compare the PR's last commit with the head it leases")
	assert.Less(t, check, strings.Index(run, ".github/reconcile-models/plan.sh"))
}

// TestConcurrencyQueuesRuns checks that runs that push the same bot branch
// wait in line, up to GitHub's limit, instead of a newer one cancelling the
// one waiting: main's runs, and every other branch's, which share the test
// bot branch.
func TestConcurrencyQueuesRuns(t *testing.T) {
	t.Parallel()

	concurrency := readWorkflow(t, runWorkflow).Concurrency
	assert.Equal(t, "reconcile-models-${{ github.ref_name == 'main' && 'main' || 'test' }}", concurrency.Group)
	assert.Equal(t, "max", concurrency.Queue)
	assert.False(t, concurrency.CancelInProgress)
}

// TestScheduledRunsReachThePR checks that the workflow has a schedule and that
// neither the vendors' parts nor their agent jobs skip scheduled runs.
func TestScheduledRunsReachThePR(t *testing.T) {
	t.Parallel()

	wf := readWorkflow(t, runWorkflow)
	require.NotEmpty(t, wf.On.Schedule, "the workflow has no schedule")
	assert.NotContains(t, readJob(t, vendorWorkflow, "agent").If, "schedule", "the agent job must run on a schedule too")
	// A dry run stops before the PR, so only a run someone starts by hand may be one.
	const handStarted = "${{ github.event_name == 'workflow_dispatch' && "
	assert.True(t, strings.HasPrefix(wf.Env["DRY_RUN"], handStarted), "only a hand-started run may be a dry run, got %q", wf.Env["DRY_RUN"])

	for name, job := range legs(t) {
		assert.NotContains(t, job.If, "schedule", "%s must run on a schedule too", name)
		assert.True(t, strings.HasPrefix(fmt.Sprint(job.With["dry_run"]), handStarted), "only a hand-started run may make %s a dry run, got %v", name, job.With["dry_run"])
	}
}

// legs are the jobs of reconcile-models.yml that run one vendor's part, by
// calling reconcile-vendor.yml.
func legs(t *testing.T) map[string]workflowJob {
	t.Helper()

	found := map[string]workflowJob{}

	for name, job := range readWorkflow(t, runWorkflow).Jobs {
		if job.Uses == "./.github/workflows/"+vendorWorkflow {
			found[name] = job
		}
	}

	require.NotEmpty(t, found, "%s calls %s for no vendor", runWorkflow, vendorWorkflow)

	return found
}

// TestLegGetsNoBotToken checks that no job in a vendor's part of the run can
// reach the bot token: no OIDC token, no AWS step, no bot token. Only the
// run's pr job pushes, once, at the end.
func TestLegGetsNoBotToken(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", vendorWorkflow))
	require.NoError(t, err)
	assert.NotContains(t, string(raw), "ACTIONS_BOT_TOKEN")
	assert.NotContains(t, string(raw), "aws-actions/")

	for name, job := range readWorkflow(t, vendorWorkflow).Jobs {
		assert.NotContains(t, job.Permissions, "id-token", "the %s job may not request an OIDC token", name)
	}
}

// TestLegSecretsAreNamed checks that each vendor's part gets only the secrets
// it names: the model key, and the debug-logging flag the agent job's check
// reads. A called workflow sees no other secret, so inherit would hand it all.
// GitHub refuses a call whose secret names start with ACTIONS_ or GITHUB_, so
// the flag goes in as STEP_DEBUG.
func TestLegSecretsAreNamed(t *testing.T) {
	t.Parallel()

	for name, job := range legs(t) {
		secrets, named := job.Secrets.(map[string]any)
		require.True(t, named, "%s must name its secrets, not inherit them", name)
		assert.ElementsMatch(t, []string{"ANTHROPIC_API_KEY", "STEP_DEBUG"}, slices.Collect(maps.Keys(secrets)), "%s's secrets", name)
		assert.Equal(t, "${{ secrets.ACTIONS_STEP_DEBUG }}", secrets["STEP_DEBUG"], "%s must pass the repo's debug-logging flag", name)
	}

	for secret := range readWorkflow(t, vendorWorkflow).On.WorkflowCall.Secrets {
		assert.False(t, strings.HasPrefix(secret, "ACTIONS_") || strings.HasPrefix(secret, "GITHUB_"), "%s takes the reserved secret name %s", vendorWorkflow, secret)
	}
}

// TestRealRunStepsRequireDryRunFalse checks that whatever only a real run does
// checks for DRY_RUN 'false', so a missing or unexpected value is a dry run,
// and that reconcile-vendor.yml can't be called without dry_run.
func TestRealRunStepsRequireDryRunFalse(t *testing.T) {
	t.Parallel()

	guarded := 0

	for _, file := range []string{runWorkflow, vendorWorkflow} {
		for name, job := range readWorkflow(t, file).Jobs {
			conditions := []string{job.If}
			for _, step := range job.Steps {
				conditions = append(conditions, step.If)
			}

			for _, condition := range conditions {
				if !strings.Contains(condition, "DRY_RUN") {
					continue
				}

				guarded++

				assert.Contains(t, condition, "env.DRY_RUN == 'false'", "%s %s: a real-run step must check for 'false'", file, name)
				assert.NotContains(t, condition, "!=", "%s %s: %q lets an unexpected value through", file, name, condition)
			}
		}
	}

	assert.Positive(t, guarded, "no step checks DRY_RUN")

	input := readWorkflow(t, vendorWorkflow).On.WorkflowCall.Inputs["dry_run"]
	assert.True(t, input.Required, "%s must be called with dry_run", vendorWorkflow)
}

// TestLegCommitCarriesTheRunTrailer checks that each vendor's commit names the
// run and the vendor in a Reconcile-Run trailer, so a later run can tell
// which commit came from which run.
func TestLegCommitCarriesTheRunTrailer(t *testing.T) {
	t.Parallel()

	steps := readJob(t, vendorWorkflow, "check").Steps
	run := steps[stepIndex(t, steps, "commit", func(s workflowStep) bool { return strings.Contains(s.Run, " commit ") })].Run
	assert.Contains(t, run, `-m "Reconcile-Run: ${GITHUB_RUN_ID}/${PROVIDER}"`)
}

// TestLegHandsOnItsHeadLast checks that a vendor's part hands on its commit
// from the check job's last step, so the next vendor and the pr job get one
// only when every step before it passed, the uploads included.
func TestLegHandsOnItsHeadLast(t *testing.T) {
	t.Parallel()

	job := readJob(t, vendorWorkflow, "check")
	assert.Equal(t, "${{ steps.done.outputs.head }}", job.Outputs["head"])
	assert.Equal(t, "done", job.Steps[len(job.Steps)-1].ID, "the head must come from the job's last step")
	assert.Equal(t, "${{ jobs.check.outputs.head }}", readWorkflow(t, vendorWorkflow).On.WorkflowCall.Outputs["head"].Value)
}

// TestRecordIsSavedBeforeThePush checks that a real run saves, before its
// one push, the record a later run replays from, for 90 days, so a pushed
// PR always has one.
func TestRecordIsSavedBeforeThePush(t *testing.T) {
	t.Parallel()

	steps := readJob(t, runWorkflow, "pr").Steps
	save := stepIndex(t, steps, "record upload", func(s workflowStep) bool { return s.With["name"] == "reconcile-record" })
	assert.Less(t, save, stepIndex(t, steps, "push", func(s workflowStep) bool { return strings.Contains(s.Run, "git push") }))
	assert.Equal(t, "90", steps[save].With["retention-days"])
	assert.Contains(t, steps[save].If, "env.DRY_RUN == 'false'")
}

// TestPlanFindsTheRecordByTrailer checks that the plan job takes the record
// from the run the newest bot commit's trailer names, and hands it to
// plan.sh.
func TestPlanFindsTheRecordByTrailer(t *testing.T) {
	t.Parallel()

	job := readJob(t, runWorkflow, "plan")
	assert.Equal(t, "read", job.Permissions["actions"], "the plan job downloads another run's record")

	run := job.Steps[stepIndex(t, job.Steps, "plan.sh", func(s workflowStep) bool { return strings.Contains(s.Run, "plan.sh") })].Run
	for _, want := range []string{"Reconcile-Run: ", "gh run download", "-n reconcile-record"} {
		assert.Contains(t, run, want)
	}

	assert.Less(t, strings.Index(run, "gh run download"), strings.Index(run, "plan.sh"), "the record comes before the plan")
}

// TestLegHandsOnItsAgentRun checks that each vendor's part says which run's
// agent made its change and when, for the record.
func TestLegHandsOnItsAgentRun(t *testing.T) {
	t.Parallel()

	outputs := readWorkflow(t, vendorWorkflow).On.WorkflowCall.Outputs
	for _, name := range []string{"agent_run", "agent_at"} {
		assert.Equal(t, "${{ jobs.check.outputs."+name+" }}", outputs[name].Value)
		assert.Equal(t, "${{ steps.done.outputs."+name+" }}", readJob(t, vendorWorkflow, "check").Outputs[name])
	}
}

// TestReplayFallsBackToTheAgent checks the agent job of a vendor planned
// for replay: replay.sh runs first, Claude runs only when it fails, and a
// replay also stands in when Claude fails, so a vendor with a commit waiting
// in the PR doesn't block the push. What Claude left is collected only when
// no replay took its place.
func TestReplayFallsBackToTheAgent(t *testing.T) {
	t.Parallel()

	steps := readJob(t, vendorWorkflow, "agent").Steps
	replay := stepIndex(t, steps, "replay", func(s workflowStep) bool { return s.ID == "replay" })
	claude := stepIndex(t, steps, "Claude", isClaude)
	fallback := stepIndex(t, steps, "fallback", func(s workflowStep) bool { return s.ID == "fallback" })
	collect := stepIndex(t, steps, "collect", func(s workflowStep) bool {
		return strings.Contains(s.Run, "out/patch.diff") && strings.Contains(s.Run, "git diff")
	})

	assert.Less(t, replay, claude)
	assert.Less(t, claude, fallback)
	assert.Less(t, fallback, collect)

	for _, at := range []int{replay, fallback} {
		assert.Contains(t, steps[at].Run, ".github/reconcile-models/replay.sh")
	}

	assert.Equal(t, "inputs.mode == 'replay'", steps[replay].If)
	assert.Equal(t, true, steps[replay].ContinueOnError, "a failed replay runs the agent")
	assert.Equal(t, "steps.replay.outcome != 'success'", steps[claude].If)
	assert.Equal(t, true, steps[claude].ContinueOnError, "a failed agent tries the replay")
	assert.Equal(t, "steps.claude.outcome == 'failure'", steps[fallback].If)
	assert.Nil(t, steps[fallback].ContinueOnError, "with neither a change nor a replay, the job fails")
	assert.Equal(t, "always() && steps.replay.outcome != 'success' && steps.fallback.outcome != 'success'", steps[collect].If)
}

// TestReplayedLegSavesNoFingerprints checks that a replayed vendor leaves no
// gate baseline: its report is an older run's, not one of these pages.
func TestReplayedLegSavesNoFingerprints(t *testing.T) {
	t.Parallel()

	assert.Contains(t, readJob(t, vendorWorkflow, "record").If, "needs.agent.outputs.replayed != 'true'")
}

// TestLegsGetTheirMode checks that each vendor's part replays when the plan
// says so, and runs its agent otherwise.
func TestLegsGetTheirMode(t *testing.T) {
	t.Parallel()

	for name, job := range legs(t) {
		assert.Equal(t, "${{ contains(fromJSON(needs.plan.outputs.replay), '"+name+"') && 'replay' || 'agent' }}", job.With["mode"], "%s's mode", name)
	}
}

// TestLegChecksOutItsBase checks that the agent and the patch check both work
// on the commit the vendor's part was given, checked out before Claude or
// pr-job.sh starts.
func TestLegChecksOutItsBase(t *testing.T) {
	t.Parallel()

	for job, first := range map[string]func(workflowStep) bool{
		"agent": isClaude,
		"check": func(s workflowStep) bool { return strings.Contains(s.Run, ".github/reconcile-models/pr-job.sh") },
	} {
		steps := readJob(t, vendorWorkflow, job).Steps
		checkout := stepIndex(t, steps, "checkout-base.sh", func(s workflowStep) bool {
			return strings.Contains(s.Run, `.github/reconcile-models/checkout-base.sh "$BASE_SHA"`)
		})
		assert.Less(t, checkout, stepIndex(t, steps, "main", first), "the %s job must check out its base first", job)
	}
}

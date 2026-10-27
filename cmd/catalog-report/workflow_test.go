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
	"cmp"
	"fmt"
	"maps"
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

// workflowStep is one step of a job in reconcile-models.yml or
// reconcile-vendor.yml.
type workflowStep struct {
	If   string            `yaml:"if"`
	Uses string            `yaml:"uses"`
	Run  string            `yaml:"run"`
	With map[string]string `yaml:"with"`
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

// workflowJob is one job: its own steps, or a call of reconcile-vendor.yml
// (Uses, With and Secrets).
type workflowJob struct {
	If          string            `yaml:"if"`
	Env         map[string]string `yaml:"env"`
	Permissions map[string]string `yaml:"permissions"`
	Steps       []workflowStep    `yaml:"steps"`
	Uses        string            `yaml:"uses"`
	With        map[string]any    `yaml:"with"`
	// Secrets is "inherit" or a map of the secrets passed.
	Secrets any `yaml:"secrets"`
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
		} `yaml:"workflow_call"`
	} `yaml:"on"`
	Env  map[string]string      `yaml:"env"`
	Jobs map[string]workflowJob `yaml:"jobs"`
}

// The run's workflow, and the one it calls for each vendor.
const (
	runWorkflow    = "reconcile-models.yml"
	vendorWorkflow = "reconcile-vendor.yml"
)

// readWorkflow reads reconcile-models.yml, or the file named.
func readWorkflow(t *testing.T, file ...string) workflow {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", cmp.Or(strings.Join(file, ""), runWorkflow)))
	require.NoError(t, err)

	var wf workflow
	require.NoError(t, yaml.Unmarshal(raw, &wf))

	return wf
}

// readJob reads one job from reconcile-models.yml.
func readJob(t *testing.T, name string) workflowJob {
	t.Helper()

	job := readWorkflow(t).Jobs[name]
	require.NotEmpty(t, job.Steps, "the workflow has no %s job steps", name)

	return job
}

// vendorJob reads one job from reconcile-vendor.yml, one vendor's part of a run.
func vendorJob(t *testing.T, name string) workflowJob {
	t.Helper()

	job := readWorkflow(t, vendorWorkflow).Jobs[name]
	require.NotEmpty(t, job.Steps, "%s has no %s job steps", vendorWorkflow, name)

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

	first := vendorJob(t, "agent").Steps[0]
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

	steps := vendorJob(t, "agent").Steps
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

	cmd := exec.CommandContext(t.Context(), "go", "list", "-deps", "./cmd/catalog-report", "./cmd/catalog-guard")
	cmd.Dir = filepath.Join("..", "..")
	out, err := cmd.Output()
	require.NoError(t, err)

	module := "github.com/redpanda-data/ai-sdk-go/"

	for pkg := range strings.Lines(string(out)) {
		pkg = strings.TrimSpace(pkg)
		assert.False(t, pkg == module+"catalog" || strings.HasPrefix(pkg, module+"catalog/") || strings.HasPrefix(pkg, module+"providers/"),
			"a tool pr-job.sh runs outside the sandbox imports %s", pkg)
	}
}

// TestCheckoutBaseTakesTheBundlesCommit checks how a vendor's part of the run
// gets its base: the run's own commit as checked out, or the previous
// vendor's commit from its bundle, never anything else.
func TestCheckoutBaseTakesTheBundlesCommit(t *testing.T) {
	t.Parallel()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "checkout-base.sh"))
	require.NoError(t, err)

	git := func(dir string, args ...string) string {
		t.Helper()

		cmd := exec.CommandContext(t.Context(), "git", append([]string{"-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main"}, args...)...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		require.NoError(t, err, string(out))

		return strings.TrimSpace(string(out))
	}

	// origin has the run's commit; a vendor's clone adds its commit and
	// bundles it, as the check job does.
	origin, leg := t.TempDir(), t.TempDir()
	git(origin, "init", "-q")
	git(origin, "commit", "-q", "--allow-empty", "-m", "main")
	runSHA := git(origin, "rev-parse", "HEAD")
	git(leg, "clone", "-q", "--depth", "1", "file://"+origin, ".")
	git(leg, "commit", "-q", "--allow-empty", "-m", "vendor")
	legSHA := git(leg, "rev-parse", "HEAD")
	bundle := filepath.Join(t.TempDir(), "reconcile.bundle")
	git(leg, "bundle", "create", "-q", bundle, "HEAD", "^"+runSHA)

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
			git(checkout, "clone", "-q", "--depth", "1", "file://"+origin, ".")

			cmd := exec.CommandContext(t.Context(), "bash", script, tt.base, tt.bundle)
			cmd.Dir = checkout
			out, err := cmd.CombinedOutput()

			if tt.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, string(out), tt.wantErr)
				assert.Equal(t, runSHA, git(checkout, "rev-parse", "HEAD"), "a refused base leaves the checkout as it was")

				return
			}

			require.NoError(t, err, string(out))
			assert.Equal(t, tt.wantHead, git(checkout, "rev-parse", "HEAD"))
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

	job := vendorJob(t, "agent")
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

	// A PR left alone keeps one note, updated each run, not a new one every week.
	for line := range strings.Lines(run) {
		if strings.Contains(line, "gh pr comment") {
			assert.Contains(t, line, "--edit-last --create-if-none", "the push step must edit its earlier note")
		}
	}
}

// TestScheduledRunsReachThePR checks that the workflow has a schedule and that
// neither the vendors' parts nor their agent jobs skip scheduled runs.
func TestScheduledRunsReachThePR(t *testing.T) {
	t.Parallel()

	wf := readWorkflow(t)
	require.NotEmpty(t, wf.On.Schedule, "the workflow has no schedule")
	assert.NotContains(t, vendorJob(t, "agent").If, "schedule", "the agent job must run on a schedule too")
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

	for name, job := range readWorkflow(t).Jobs {
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

	for name, job := range legs(t) {
		assert.NotContains(t, job.Permissions, "id-token", "%s may not pass on OIDC rights", name)
	}
}

// TestLegSecretsAreNamed checks that each vendor's part gets only the secrets
// it names: the model key, and the debug-logging flag the agent job's check
// reads. A called workflow sees no other secret, so inherit would hand it all.
func TestLegSecretsAreNamed(t *testing.T) {
	t.Parallel()

	for name, job := range legs(t) {
		secrets, named := job.Secrets.(map[string]any)
		require.True(t, named, "%s must name its secrets, not inherit them", name)
		assert.ElementsMatch(t, []string{"ANTHROPIC_API_KEY", "ACTIONS_STEP_DEBUG"}, slices.Collect(maps.Keys(secrets)), "%s's secrets", name)
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

	steps := vendorJob(t, "check").Steps
	run := steps[stepIndex(t, steps, "commit", func(s workflowStep) bool { return strings.Contains(s.Run, " commit ") })].Run
	assert.Contains(t, run, `-m "Reconcile-Run: ${GITHUB_RUN_ID}/${PROVIDER}"`)
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
		steps := vendorJob(t, job).Steps
		checkout := stepIndex(t, steps, "checkout-base.sh", func(s workflowStep) bool {
			return strings.Contains(s.Run, `.github/reconcile-models/checkout-base.sh "$BASE_SHA"`)
		})
		assert.Less(t, checkout, stepIndex(t, steps, "main", first), "the %s job must check out its base first", job)
	}
}

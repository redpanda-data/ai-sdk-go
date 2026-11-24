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
	"encoding/json"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The notify job, and the only secret it may read.
const (
	notifyJob  = "notify"
	slackToken = "sdlc/prod/github/slack_vbot_token" //nolint:gosec // the secret's name in AWS, not a secret
)

// TestNotifyTellsTheChannel runs notify.sh, the notify job's step that writes
// the message for the team's channel: a PR the run pushed, the jobs that
// failed or were cancelled, a run that stopped before the push, and the
// questions of a run that pushed nothing. A run with none of these posts
// nothing, and a run from another branch posts only its failures unless it
// asks to post its test PR. Every message has the same shape: a bold heading
// that says what happened, what it means, the step to take, and the report's
// counts as a list.
func TestNotifyTellsTheChannel(t *testing.T) {
	t.Parallel()

	const (
		changes = "**28 changes** to the catalogue: 18 capabilities, 5 replacements. Not changed: 11 open questions, 148 skipped offerings."
		run     = "<https://github.com/o/r/actions/runs/42|GitHub Actions run>"
		summary = "<https://github.com/o/r/actions/runs/42/attempts/2|run summary>"
		newRun  = "<https://github.com/o/r/actions/workflows/reconcile-models.yml|start a new run>"
		pr336   = "<https://github.com/o/r/pull/336|#336>"
		branch  = "adam/try-it"

		// The parts several messages share: changes' counts, the bot PR with
		// its own heading (alone or below a failure), the retry after a push,
		// and the step for a stopped run.
		counts   = "• 28 changes to the catalog: 18 capabilities, 5 replacements\n• 11 open questions\n• 148 model offerings skipped under catalog rules\n"
		reviewer = "*Reconcile Bot - bot PR needs a reviewer*\n\nTo take " + pr336 + ", react with :eyes: and add yourself as a reviewer.\n\n" +
			"The PR has the report details.\n" + counts
		retryNew = "To retry the Anthropic model catalog check, " + newRun + " with \"Stop before pushing\" unticked. " +
			"Re-running this run's failed jobs can't update the PR again.\n"
		stopStep = "To let the next run update " + pr336 + ", merge or close it, or remove that commit if it was pushed after the bot's last update.\n"
	)

	// The report's own lines for a run with no changes, so a change to their
	// wording can't quietly stop the questions' posts, keep the report's
	// short words for its counts, or post the rates a page only rounds.
	questions, quiet := strings.TrimSpace(summaryLine(nil, 2)), strings.TrimSpace(summaryLine(nil, 0))
	everyCount := func(n int) string {
		part := vendorPart{held: make([]string, n), rounded: make([]NeedsHuman, n), shown: Report{Skipped: make([]Skipped, n)}}

		return strings.TrimSpace(summaryLine([]vendorPart{part}, n))
	}

	// needs builds the notify job's needs: every job succeeded unless results
	// names another result for it, the pr job has these outputs, and each
	// vendor in checked handed on a head before its job ended.
	needs := func(pr, results map[string]string, checked ...string) string {
		jobs := map[string]any{}

		for _, name := range []string{"plan", "anthropic", "openai", "carry", "pr"} {
			job := map[string]any{"result": cmp.Or(results[name], "success"), "outputs": map[string]string{}}
			if name == "plan" && job["result"] == "success" {
				job["outputs"] = map[string]string{"pr": "336"} // the open bot PR the plan read
			}

			if slices.Contains(checked, name) {
				job["outputs"] = map[string]string{"head": "abc1234"}
			}

			if name == "pr" && pr != nil {
				job["outputs"] = pr
			}

			jobs[name] = job
		}

		raw, err := json.Marshal(jobs)
		require.NoError(t, err)

		return string(raw)
	}
	pushed := map[string]string{"pushed": "true", "pr": "336"}

	tests := []struct {
		name  string
		needs string
		body  string // the PR body's summary line; empty for no body
		ref   string // the run's branch; empty for main
		optIn bool   // the dispatch asked to post its test PR
		want  string // the message; empty for no post
	}{
		{
			name:  "a pushed PR",
			needs: needs(pushed, nil),
			body:  changes,
			want:  reviewer,
		},
		{
			name:  "a pushed PR with no body",
			needs: needs(pushed, nil),
			want:  "*Reconcile Bot - bot PR needs a reviewer*\n\nTo take " + pr336 + ", react with :eyes: and add yourself as a reviewer.\n",
		},
		{
			name:  "a run that changed nothing",
			needs: needs(nil, nil),
			body:  quiet,
		},
		{
			name:  "questions but no push",
			needs: needs(nil, nil),
			body:  questions,
			want: "*Reconcile Bot - run has open questions*\n\nThe run pushed no catalog changes.\n\n" +
				"The " + summary + " has the report details.\n• 2 open questions\n",
		},
		{
			name:  "every count the report writes, one of each",
			needs: needs(nil, nil),
			body:  everyCount(1),
			want: "*Reconcile Bot - run has open questions*\n\nThe run pushed no catalog changes.\n\n" +
				"The " + summary + " has the report details.\n• 1 open question\n• 1 report entry covered by a recorded decision\n" +
				"• 1 model offering skipped under catalog rules\n",
		},
		{
			name:  "every count the report writes, two of each",
			needs: needs(nil, nil),
			body:  everyCount(2),
			want: "*Reconcile Bot - run has open questions*\n\nThe run pushed no catalog changes.\n\n" +
				"The " + summary + " has the report details.\n• 2 open questions\n• 2 report entries covered by recorded decisions\n" +
				"• 2 model offerings skipped under catalog rules\n",
		},
		{
			name:  "changes already on the PR",
			needs: needs(map[string]string{"changed": "true", "pushed": "false", "pr": "336"}, nil),
			body:  changes,
			want: "*Reconcile Bot - run has open questions*\n\nThis run's changes are already on " + pr336 + ".\n\n" +
				"The " + summary + " has the report details.\n" + counts,
		},
		{
			name:  "a vendor failed and the PR still updated",
			needs: needs(pushed, map[string]string{"anthropic": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date.\n\n" + retryNew + "\n" + reviewer,
		},
		{
			name:  "a vendor failed and the run pushed nothing",
			needs: needs(nil, map[string]string{"anthropic": "failure"}),
			body:  "**No changes** to the catalogue. Not changed: 6 open questions, 8 held by decisions, 141 skipped offerings.",
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry the Anthropic model catalog check, open the " + run + " and choose \"Re-run failed jobs\".\n\n" +
				"The " + summary + " has the report details.\n• 6 open questions\n• 8 report entries covered by recorded decisions\n" +
				"• 141 model offerings skipped under catalog rules\n",
		},
		{
			name:  "a vendor failed and the PR didn't update",
			needs: needs(nil, map[string]string{"anthropic": "failure", "pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. Also failed: updating the bot PR. The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry the Anthropic model catalog check, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "a vendor and the PR update failed after the push",
			needs: needs(pushed, map[string]string{"anthropic": "failure", "carry": "failure", "pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. Also failed: carrying people's commits and updating the bot PR.\n\n" + retryNew + "\n" + reviewer,
		},
		{
			name:  "a vendor failed after its check",
			needs: needs(pushed, map[string]string{"anthropic": "failure"}, "anthropic"),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's changes are in this run; a later step failed.\n\n" + retryNew + "\n" + reviewer,
		},
		{
			name:  "one vendor failed before its check and one after",
			needs: needs(nil, map[string]string{"anthropic": "failure", "openai": "cancelled"}, "openai"),
			body:  quiet,
			want: "*Reconcile Bot - Anthropic and OpenAI catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. OpenAI's changes are in this run; a later step failed. " +
				"The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry the Anthropic and OpenAI model catalog checks, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "two vendors failed after the push",
			needs: needs(pushed, map[string]string{"anthropic": "failure", "openai": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - Anthropic and OpenAI catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's and OpenAI's catalog entries may be out of date.\n\n" +
				"To retry the Anthropic and OpenAI model catalog checks, " + newRun + " for each, with \"Stop before pushing\" unticked. " +
				"Re-running this run's failed jobs can't update the PR again.\n\n" + reviewer,
		},
		{
			name:  "the push failed with changes waiting",
			needs: needs(map[string]string{"changed": "true", "pr": "336"}, map[string]string{"pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - catalog reconciliation failed* :red_circle:\n\n" +
				"Failed: updating the bot PR. The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "carrying people's commits failed",
			needs: needs(nil, map[string]string{"carry": "failure"}),
			want: "*Reconcile Bot - catalog reconciliation failed* :red_circle:\n\n" +
				"Failed: carrying people's commits. The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "someone pushed to the PR during the run",
			needs: needs(map[string]string{"changed": "true", "moved": "true", "pr": "336"}, map[string]string{"pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - catalog reconciliation failed* :red_circle:\n\n" +
				"Failed: updating the bot PR. Someone pushed to " + pr336 + " during the run, so this run didn't update it.\n\n" +
				"To retry, " + newRun + " with \"Stop before pushing\" unticked, or wait for the next scheduled run. " +
				"Re-running the failed jobs would stop the same way.\n",
		},
		{
			name:  "a vendor failed and someone pushed to the PR during the run",
			needs: needs(map[string]string{"changed": "true", "moved": "true", "pr": "336"}, map[string]string{"anthropic": "failure", "pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. Also failed: updating the bot PR. " +
				"Someone pushed to " + pr336 + " during the run, so this run didn't update it.\n\n" +
				"To retry, " + newRun + " with \"Stop before pushing\" unticked, or wait for the next scheduled run. " +
				"Re-running the failed jobs would stop the same way.\n",
		},
		{
			name:  "a vendor failed after its check and the PR already had the changes",
			needs: needs(map[string]string{"changed": "true", "pushed": "false", "pr": "336"}, map[string]string{"anthropic": "failure"}, "anthropic"),
			body:  changes,
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's changes are in this run; a later step failed. This run's changes are already on " + pr336 + ".\n\n" +
				"To retry the Anthropic model catalog check, open the " + run + " and choose \"Re-run failed jobs\".\n\n" +
				"The " + summary + " has the report details.\n" + counts,
		},
		{
			name:  "the PR was closed during the run",
			needs: needs(map[string]string{"changed": "true", "pr": "336"}, nil),
			body:  changes,
		},
		{
			name:  "a vendor's job was cancelled",
			needs: needs(nil, map[string]string{"openai": "cancelled"}),
			body:  quiet,
			want: "*Reconcile Bot - OpenAI catalog reconciliation failed* :red_circle:\n\n" +
				"OpenAI's catalog entries may be out of date. The run pushed no catalog changes, so " + pr336 + " wasn't updated.\n\n" +
				"To retry the OpenAI model catalog check, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "the plan failed",
			needs: needs(nil, map[string]string{"plan": "failure"}),
			want: "*Reconcile Bot - catalog reconciliation failed before checking any vendor* :red_circle:\n\n" +
				"To retry, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "the PR update failed after the push",
			needs: needs(pushed, map[string]string{"pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - catalog reconciliation failed* :red_circle:\n\n" +
				"Failed: updating the bot PR.\n\n" +
				"To retry, open the " + run + " and choose \"Re-run failed jobs\".\n\n" + reviewer,
		},
		{
			name:  "the new PR didn't open after the push",
			needs: needs(map[string]string{"pushed": "true"}, map[string]string{"pr": "failure"}),
			body:  changes,
			want: "*Reconcile Bot - catalog reconciliation failed* :red_circle:\n\n" +
				"Failed: updating the bot PR. The run pushed to the bot branch, but no PR shows it yet.\n\n" +
				"To retry, open the " + run + " and choose \"Re-run failed jobs\".\n",
		},
		{
			name:  "stopped before the push",
			needs: needs(map[string]string{"stop": "true", "reason": "a commit <x> & y can't be carried", "pr": "336"}, nil),
			want: "*Reconcile Bot - bot PR not updated*\n\n" +
				"The run didn't update " + pr336 + ", because a commit &lt;x&gt; &amp; y can't be carried.\n\n" + stopStep,
		},
		{
			name:  "a vendor failed and the run stopped before the push",
			needs: needs(map[string]string{"stop": "true", "reason": "a commit x can't be carried", "pr": "336"}, map[string]string{"anthropic": "failure"}),
			want: "*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date. The run didn't update " + pr336 + ", because a commit x can't be carried.\n\n" + stopStep,
		},
		{
			name:  "a test branch's PR",
			needs: needs(pushed, nil),
			body:  changes,
			ref:   branch,
		},
		{
			name:  "a test branch's questions",
			needs: needs(nil, nil),
			body:  questions,
			ref:   branch,
		},
		{
			name:  "a test branch that failed",
			needs: needs(pushed, map[string]string{"anthropic": "failure"}),
			body:  changes,
			ref:   branch,
			want: ":test_tube: *Test only:* this is a test run from `" + branch + "`, ignore this message.\n" +
				"*Reconcile Bot - Anthropic catalog reconciliation failed* :red_circle:\n\n" +
				"Anthropic's catalog entries may be out of date.\n\n" + retryNew,
		},
		{
			name:  "a test branch that asked to post its PR",
			needs: needs(pushed, nil),
			body:  changes,
			ref:   branch,
			optIn: true,
			want:  ":test_tube: *Test only:* this is a test run from `" + branch + "`, ignore this message.\n" + reviewer,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, notify(t, tt.needs, tt.body, tt.ref, tt.optIn))
		})
	}
}

// TestNotifyNamesEveryVendor fails every vendor in named-sources.json and
// checks that the heading names each one as people write it, not by its job's
// id, so a vendor added later gets a name in notify.sh too.
func TestNotifyNamesEveryVendor(t *testing.T) {
	t.Parallel()

	vendors := vendorOrder(t)
	jobs := map[string]any{"plan": map[string]string{"result": "success"}, "carry": map[string]string{"result": "success"}, "pr": map[string]string{"result": "success"}}

	for _, vendor := range vendors {
		jobs[vendor] = map[string]string{"result": "failure"}
	}

	raw, err := json.Marshal(jobs)
	require.NoError(t, err)

	heading, _, _ := strings.Cut(notify(t, string(raw), "", "", false), "\n")
	assert.Contains(t, heading, "catalog reconciliation failed")

	for _, vendor := range vendors {
		assert.NotContains(t, heading, vendor, "give %s a name in notify.sh's spoken()", vendor)
	}
}

// notify runs notify.sh as the notify job does: needs is the jobs' results,
// body the PR body's summary line (no body when empty), ref the run's branch
// (main when empty), and optIn whether the dispatch asked to post its test
// PR. It returns the message, or "" when the run posts nothing.
func notify(t *testing.T, needs, body, ref string, optIn bool) string {
	t.Helper()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "notify.sh"))
	require.NoError(t, err)

	dir := t.TempDir()
	bodyFile, message, outputs := filepath.Join(dir, "body.md"), filepath.Join(dir, "message.txt"), filepath.Join(dir, "outputs")

	if body != "" {
		require.NoError(t, os.WriteFile(bodyFile, []byte("## Catalogue drift: anthropic\n\n**Context:** x\n\n"+body+"\n\nmore\n"), 0o600))
	}

	cmd := exec.CommandContext(t.Context(), script, bodyFile, message)

	cmd.Env = append(os.Environ(), "NEEDS="+needs, "GITHUB_OUTPUT="+outputs, "GITHUB_REF_NAME="+cmp.Or(ref, "main"),
		"POST_TEST_PR="+strconv.FormatBool(optIn), "GITHUB_SERVER_URL=https://github.com", "GITHUB_REPOSITORY=o/r",
		"GITHUB_RUN_ID=42", "GITHUB_RUN_ATTEMPT=2")

	out, err := cmd.CombinedOutput()
	require.NoError(t, err, "%s", out)

	raw, err := os.ReadFile(outputs)
	require.NoError(t, err)

	if parseOutputs(string(raw))["post"] != "true" {
		assert.Equal(t, map[string]string{"post": "false"}, parseOutputs(string(raw)))
		assert.NoFileExists(t, message)

		return ""
	}

	text, err := os.ReadFile(message)
	require.NoError(t, err)

	return string(text)
}

// TestNotifyJobHearsFromEveryJob checks that the notify job waits for every
// other job, so a failed one is named and a vendor added later isn't missed,
// and that it runs after failures but not after a cancelled run.
func TestNotifyJobHearsFromEveryJob(t *testing.T) {
	t.Parallel()

	jobs := readWorkflow(t, runWorkflow).Jobs
	notify := readJob(t, runWorkflow, notifyJob)

	others := slices.DeleteFunc(slices.Collect(maps.Keys(jobs)), func(name string) bool { return name == notifyJob })
	assert.ElementsMatch(t, others, needsOf(notify))

	// notify.sh names every failed job a vendor except plan, carry and pr.
	steps := slices.DeleteFunc(slices.Clone(others), func(name string) bool {
		return strings.HasSuffix(jobs[name].Uses, "/reconcile-vendor.yml")
	})
	assert.ElementsMatch(t, []string{"plan", "carry", "pr"}, steps, "a new job that isn't a vendor's goes in notify.sh's steps too")
	assert.Equal(t, "${{ !cancelled() }}", notify.If)
	assert.Equal(t, "write", notify.Permissions["id-token"])
	assert.Len(t, notify.Permissions, 2, "the notify job needs only contents: read and id-token: write")

	// A run from another branch posts its test PR only when its dispatch asks to.
	optIn := readWorkflow(t, runWorkflow).On.WorkflowDispatch.Inputs["post_test_pr"]
	assert.Equal(t, "false", optIn.Default)

	message := notify.Steps[stepIndex(t, notify.Steps, "message", func(s workflowStep) bool { return strings.Contains(s.Run, "notify.sh") })]
	assert.Equal(t, "${{ github.event_name == 'workflow_dispatch' && inputs.post_test_pr }}", message.Env["POST_TEST_PR"])
}

// TestSlackPostChecksTheReply runs the notify job's post step with a stand-in
// curl: Slack answers 200 with ok false when it refuses a post (a channel the
// app isn't in, a bad token), so the step must read the reply, and the post
// carries the channel and the message without link previews.
func TestSlackPostChecksTheReply(t *testing.T) {
	t.Parallel()

	steps := readJob(t, runWorkflow, notifyJob).Steps
	run := steps[stepIndex(t, steps, "post", func(s workflowStep) bool { return strings.Contains(s.Run, "chat.postMessage") })].Run

	// curl saves the payload it was given and prints REPLY.
	path := stubPath(t, map[string]string{
		"curl": "#!/bin/sh\nfor a in \"$@\"; do case \"$a\" in @*) cp \"${a#@}\" \"$SENT\";; esac; done\necho \"$REPLY\"\n",
	})

	tests := []struct {
		name   string
		reply  string
		passes bool
	}{
		{name: "posted", reply: `{"ok": true}`, passes: true},
		{name: "refused", reply: `{"ok": false, "error": "not_in_channel"}`},
		{name: "not JSON", reply: `<html>`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()
			sent := filepath.Join(dir, "sent.json")
			require.NoError(t, os.WriteFile(filepath.Join(dir, "message.txt"), []byte("Catalog bot PR <u|#1> needs a reviewer.\n"), 0o600))

			// GitHub runs a step's script as bash -eo pipefail.
			cmd := exec.CommandContext(t.Context(), "bash", "-eo", "pipefail", "-c", run)

			cmd.Env = append(os.Environ(), path, "SENT="+sent, "REPLY="+tt.reply, "RUNNER_TEMP="+dir,
				"CHANNEL=C123", "SLACK_VBOT_TOKEN=xoxb-test")

			out, err := cmd.CombinedOutput()
			assert.Equal(t, tt.passes, err == nil, "%s", out)
			assert.NotContains(t, string(out), "xoxb-test", "the step must not print the token")

			if !tt.passes {
				return
			}

			raw, err := os.ReadFile(sent)
			require.NoError(t, err)

			var payload map[string]any
			require.NoError(t, json.Unmarshal(raw, &payload))
			assert.Equal(t, "C123", payload["channel"])
			assert.Equal(t, "Catalog bot PR <u|#1> needs a reviewer.\n", payload["text"])
			assert.Equal(t, false, payload["unfurl_links"])
		})
	}
}

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
// asks to post its test PR.
func TestNotifyTellsTheChannel(t *testing.T) {
	t.Parallel()

	script, err := filepath.Abs(filepath.Join("..", "..", ".github", "reconcile-models", "notify.sh"))
	require.NoError(t, err)

	const (
		changes = "**28 changes** to the catalogue: 18 capabilities, 5 replacements. Not changed: 11 open questions, 148 skipped offerings."
		run     = "<https://github.com/o/r/actions/runs/42|run>"
		pr336   = "<https://github.com/o/r/pull/336|#336>"
		branch  = "adam/try-it"
	)

	// The report's own lines for a run with no changes, so a change to their
	// wording can't quietly stop the questions' posts.
	questions, quiet := strings.TrimSpace(summaryLine(nil, 2)), strings.TrimSpace(summaryLine(nil, 0))

	// needs builds the notify job's needs: every job succeeded unless results
	// names another result for it, and the pr job has these outputs.
	needs := func(pr, results map[string]string) string {
		jobs := map[string]any{}

		for _, name := range []string{"plan", "anthropic", "openai", "carry", "pr"} {
			job := map[string]any{"result": cmp.Or(results[name], "success"), "outputs": map[string]string{}}
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
		post  bool
		has   []string
		hasnt []string
	}{
		{
			name:  "a pushed PR",
			needs: needs(pushed, nil),
			body:  changes,
			post:  true,
			has: []string{
				"Reconcile bot PR needs a reviewer: " + pr336 + "\n\n" +
					"• 28 changes to the catalogue: 18 capabilities, 5 replacements\n• 11 open questions\n• 148 skipped offerings\n\n" +
					"To take it, react with :eyes:",
				"two working days, the fallback reviewer",
			},
			hasnt: []string{"**", "failed", "Test", "Not changed"},
		},
		{
			name:  "a pushed PR with no body",
			needs: needs(pushed, nil),
			post:  true,
			has:   []string{"needs a reviewer: " + pr336 + "\n\nTo take it"},
			hasnt: []string{"•"},
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
			post:  true,
			has:   []string{run, "pushed nothing", "2 open questions", "run's summary"},
			hasnt: []string{":eyes:"},
		},
		{
			name:  "changes already on the PR",
			needs: needs(map[string]string{"pushed": "false", "pr": "336"}, nil),
			body:  changes,
			post:  true,
			has:   []string{run, "already on " + pr336, "11 open questions"},
			hasnt: []string{"pushed nothing", "needs a reviewer"},
		},
		{
			name:  "a vendor failed and the PR still updated",
			needs: needs(pushed, map[string]string{"anthropic": "failure"}),
			body:  changes,
			post:  true,
			has:   []string{run, "failed in: anthropic", pr336, ":eyes:"},
			hasnt: []string{"openai"},
		},
		{
			name:  "a vendor's job was cancelled",
			needs: needs(nil, map[string]string{"openai": "cancelled"}),
			body:  quiet,
			post:  true,
			has:   []string{run, "failed in: openai"},
		},
		{
			name:  "the plan failed",
			needs: needs(nil, map[string]string{"plan": "failure"}),
			post:  true,
			has:   []string{run, "failed in: plan"},
		},
		{
			name:  "the PR update failed after the push",
			needs: needs(pushed, map[string]string{"pr": "failure"}),
			body:  changes,
			post:  true,
			has:   []string{"failed in: pr", pr336, "needs a reviewer"},
			hasnt: []string{"pushed nothing"},
		},
		{
			name:  "the new PR didn't open after the push",
			needs: needs(map[string]string{"pushed": "true"}, map[string]string{"pr": "failure"}),
			body:  changes,
			post:  true,
			has:   []string{"failed in: pr", "pushed to the bot branch", "no PR"},
			hasnt: []string{"needs a reviewer", "/pull/"},
		},
		{
			name:  "stopped before the push",
			needs: needs(map[string]string{"stop": "true", "reason": "a commit <x> & y can't be carried", "pr": "336"}, nil),
			body:  questions,
			post:  true,
			has:   []string{run, "didn't update " + pr336 + ", because a commit &lt;x&gt; &amp; y can't be carried."},
			hasnt: []string{"<x>", "pushed nothing"},
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
			post:  true,
			has:   []string{run, "failed in: anthropic"},
			hasnt: []string{pr336, "needs a reviewer"},
		},
		{
			name:  "a test branch that asked to post its PR",
			needs: needs(pushed, nil),
			body:  changes,
			ref:   branch,
			optIn: true,
			post:  true,
			has:   []string{"Reconcile bot PR needs a reviewer: " + pr336, "• 28 changes to the catalogue"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			dir := t.TempDir()
			body, message, outputs := filepath.Join(dir, "body.md"), filepath.Join(dir, "message.txt"), filepath.Join(dir, "outputs")

			if tt.body != "" {
				require.NoError(t, os.WriteFile(body, []byte("## Catalogue drift: anthropic\n\n**Context:** x\n\n"+tt.body+"\n\nmore\n"), 0o600))
			}

			cmd := exec.CommandContext(t.Context(), script, body, message)

			cmd.Env = append(os.Environ(), "NEEDS="+tt.needs, "GITHUB_OUTPUT="+outputs, "GITHUB_REF_NAME="+cmp.Or(tt.ref, "main"),
				"POST_TEST_PR="+strconv.FormatBool(tt.optIn), "GITHUB_SERVER_URL=https://github.com", "GITHUB_REPOSITORY=o/r", "GITHUB_RUN_ID=42")

			out, err := cmd.CombinedOutput()
			require.NoError(t, err, "%s", out)

			raw, err := os.ReadFile(outputs)
			require.NoError(t, err)
			assert.Equal(t, map[string]string{"post": strconv.FormatBool(tt.post)}, parseOutputs(string(raw)))

			if !tt.post {
				assert.NoFileExists(t, message)

				return
			}

			text, err := os.ReadFile(message)
			require.NoError(t, err)

			// A test branch's post opens with the note; main's never has it.
			if tt.ref != "" {
				assert.True(t, strings.HasPrefix(string(text), ":test_tube: *Test only:* a run from the `"+tt.ref+"` branch; nothing on main changes.\n"), "%s", text)
			} else {
				assert.NotContains(t, string(text), ":test_tube:")
			}

			for _, want := range tt.has {
				assert.Contains(t, string(text), want)
			}

			for _, unwanted := range tt.hasnt {
				assert.NotContains(t, string(text), unwanted)
			}
		})
	}
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

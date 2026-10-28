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
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestReplayCommand runs the command on commits in a scratch repo: a bot
// commit that changes a rate, the snapshot and a test, replayed onto a base
// with an unrelated change. Only an exact replay passes; the command names
// each difference and fails otherwise.
func TestReplayCommand(t *testing.T) {
	t.Parallel()

	bin := filepath.Join(t.TempDir(), "catalog-replay")
	build := exec.CommandContext(t.Context(), "go", "build", "-o", bin, ".")
	out, err := build.CombinedOutput()
	require.NoError(t, err, string(out))

	repo := t.TempDir()
	git := func(args ...string) string {
		t.Helper()

		cmd := exec.CommandContext(t.Context(), "git", append([]string{"-c", "user.name=t", "-c", "user.email=t@t"}, args...)...) //nolint:gosec // the test's own git commands
		cmd.Dir = repo
		out, err := cmd.CombinedOutput()
		require.NoError(t, err, string(out))

		return strings.TrimSpace(string(out))
	}
	// commit writes the files and commits them, returning the commit.
	commit := func(message string, files map[string]string) string {
		t.Helper()

		for path, content := range files {
			require.NoError(t, os.MkdirAll(filepath.Join(repo, filepath.Dir(path)), 0o700))
			require.NoError(t, os.WriteFile(filepath.Join(repo, path), []byte(content), 0o600))
		}

		git("add", ".")
		git("commit", "-q", "-m", message)

		return git("rev-parse", "HEAD")
	}
	snapshot := func(opus, sonnet string) string {
		return `{"facts": {}, "providers": [{"provider": "anthropic", "offerings": [{"id": "claude-opus-5-5", "model": "anthropic/claude-opus-5-5", "input": ` + opus + `},
			{"id": "claude-sonnet-5-5", "model": "anthropic/claude-sonnet-5-5", "input": ` + sonnet + `}]}]}`
	}
	const models = "package anthropic\n\n// Opus\nvar opus = 5\n\n// gap\n// gap\n// gap\n// gap\n\n// Sonnet\nvar sonnet = 3\n"
	const test = "package anthropic\n\n// pins 5\n"

	git("init", "-q")

	parent := commit("main", map[string]string{"catalog/snapshot.json": snapshot("5", "3"), "providers/anthropic/models.go": models, "providers/anthropic/models_test.go": test})
	bot := commit("bot", map[string]string{
		"catalog/snapshot.json": snapshot("6", "3"), "providers/anthropic/models.go": strings.Replace(models, "opus = 5", "opus = 6", 1),
		"providers/anthropic/models_test.go": strings.Replace(test, "pins 5", "pins 6", 1),
	})

	git("checkout", "-q", parent)

	base := commit("main moves on", map[string]string{"catalog/snapshot.json": snapshot("5", "4"), "providers/anthropic/models.go": strings.Replace(models, "sonnet = 3", "sonnet = 4", 1)})
	moved := strings.Replace(strings.Replace(models, "sonnet = 3", "sonnet = 4", 1), "opus = 5", "opus = 6", 1)
	exact := commit("replay", map[string]string{
		"catalog/snapshot.json": snapshot("6", "4"), "providers/anthropic/models.go": moved,
		"providers/anthropic/models_test.go": strings.Replace(test, "pins 5", "pins 6", 1),
	})

	git("checkout", "-q", base)

	noTest := commit("replay without the test", map[string]string{"catalog/snapshot.json": snapshot("6", "4"), "providers/anthropic/models.go": moved})

	git("checkout", "-q", base)

	wrong := commit("replay on sonnet", map[string]string{
		"catalog/snapshot.json": snapshot("5", "5"), "providers/anthropic/models.go": strings.Replace(models, "sonnet = 3", "sonnet = 5", 1),
		"providers/anthropic/models_test.go": strings.Replace(test, "pins 5", "pins 6", 1),
	})

	report := filepath.Join(t.TempDir(), "report.json")
	require.NoError(t, os.WriteFile(report, []byte(`{"changes": [{"offering": "claude-opus-5-5", "field": "price", "old": "5", "new": "6",
		"source_url": "https://platform.claude.com/pricing", "rule": "price-follow"}], "needs_human": [], "skipped": []}`), 0o600))

	tests := []struct {
		name, replay string
		want         []string
	}{
		{name: "an exact replay", replay: exact},
		{name: "a replay that loses a file", replay: noTest, want: []string{"providers/anthropic/models_test.go: the replay's change to it differs from the bot's"}},
		{
			name: "a replay on the wrong offering", replay: wrong,
			want: []string{
				"claude-opus-5-5: the replay doesn't make the bot's change",
				"claude-sonnet-5-5: the replay changes it, and the bot's commit didn't",
				"providers/anthropic/models.go: the replay's change to it differs from the bot's",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			cmd := exec.CommandContext(t.Context(), bin, "-parent", parent, "-bot", bot, "-base", base, "-replay", tt.replay, "-report", report, "-vendor", "anthropic")
			cmd.Dir = repo
			out, err := cmd.CombinedOutput()

			if tt.want == nil {
				require.NoError(t, err, string(out))
				assert.Empty(t, string(out))

				return
			}

			require.Error(t, err)

			for _, want := range tt.want {
				assert.Contains(t, string(out), want)
			}
		})
	}

	t.Run("a missing flag", func(t *testing.T) {
		t.Parallel()

		cmd := exec.CommandContext(t.Context(), bin, "-parent", parent, "-bot", bot, "-base", base, "-report", report, "-vendor", "anthropic")
		cmd.Dir = repo
		out, err := cmd.CombinedOutput()
		require.Error(t, err)
		assert.Contains(t, string(out), "-replay")
	})
}

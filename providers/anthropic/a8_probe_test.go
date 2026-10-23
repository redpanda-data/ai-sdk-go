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

package anthropic

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestA8Probe exists only on the adam/ai-2527-a8-probe test branch. It tries
// harmless writes to files that later steps of the agent job use, and fails
// on purpose with one "a8:" line per write, so the agent can report them; the
// "A8 probe check" step reports which writes later steps can see.
func TestA8Probe(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("runner only")
	}

	short := func(err error) string {
		var pathErr *os.PathError
		if errors.As(err, &pathErr) {
			return pathErr.Err.Error()
		}

		if err == nil {
			return "ok"
		}

		return err.Error()
	}
	appendTo := func(label, path, text string) {
		f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		if err == nil {
			_, err = f.WriteString(text)
			_ = f.Close()
		}

		t.Errorf("a8: %s: %s", label, short(err))
	}

	_, err := os.Stat("/home/runner/work/_temp/_runner_file_commands")
	t.Errorf("a8: file-commands dir: %s", short(err))

	envFiles, _ := filepath.Glob("/home/runner/work/_temp/_runner_file_commands/set_env_*")
	t.Errorf("a8: set_env files found: %d", len(envFiles))
	t.Errorf("a8: GITHUB_ENV set: %t", os.Getenv("GITHUB_ENV") != "")

	if p := os.Getenv("GITHUB_ENV"); p != "" {
		envFiles = append(envFiles, p)
	}

	for i, f := range envFiles {
		appendTo("env file "+string(rune('a'+i)), f, "\nA8_MARKER=set-by-code-the-agent-ran\n")
	}

	appendTo("run-stats", "../../.github/reconcile-models/run-stats.sh", "\necho 'A8 marker: run-stats.sh was changed by code the agent ran'\n")
	appendTo("git config", "../../.git/config", "\n[a8]\n\tmarker = set-by-code-the-agent-ran\n")

	err = os.WriteFile("/home/runner/work/_actions/a8-probe.txt", []byte("a8\n"), 0o600)
	t.Errorf("a8: _actions: %s", short(err))
}

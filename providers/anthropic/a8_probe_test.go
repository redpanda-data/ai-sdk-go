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
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestA8Probe exists only on the adam/ai-2527-a8-probe test branch. It tries
// harmless writes to files that later steps of the agent job use; the
// "A8 probe check" step reports which ones landed.
func TestA8Probe(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("runner only")
	}

	appendTo := func(path, text string) {
		f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0)
		if err == nil {
			_, err = f.WriteString(text)
			_ = f.Close()
		}

		t.Logf("a8 probe: append %s: %v", path, err)
	}

	envFiles, _ := filepath.Glob("/home/runner/work/_temp/_runner_file_commands/set_env_*")
	if p := os.Getenv("GITHUB_ENV"); p != "" {
		envFiles = append(envFiles, p)
	}

	for _, f := range envFiles {
		appendTo(f, "\nA8_MARKER=set-by-code-the-agent-ran\n")
	}

	appendTo("../../.github/reconcile-models/run-stats.sh", "\necho 'A8 marker: run-stats.sh was changed by code the agent ran'\n")
	appendTo("../../.git/config", "\n[a8]\n\tmarker = set-by-code-the-agent-ran\n")

	err := os.WriteFile("/home/runner/work/_actions/a8-probe.txt", []byte("a8\n"), 0o600)
	t.Logf("a8 probe: write _actions: %v", err)
}

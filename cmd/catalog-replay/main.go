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

// Command catalog-replay checks that a vendor's commit from an earlier
// reconcile run, replayed onto a newer base with no model, is exactly the
// original change. Run in the checkout with four commits: the bot commit's
// parent, the bot commit, the replay's base and the replay. The committed
// snapshots must agree item by item (reconcile.ReplayMismatches), and every
// other file the commits change must change by the same hunks, context
// lines included (git patch-id). It prints each difference and exits 1, so
// the run falls back to the vendor's agent; it runs nothing from the commits.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// snapshotPath is regenerated, never replayed, so only its items count.
const snapshotPath = "catalog/snapshot.json"

func main() {
	parent := flag.String("parent", "", "the commit the bot's change was made on")
	bot := flag.String("bot", "", "the bot's commit")
	base := flag.String("base", "", "the replay's base")
	replay := flag.String("replay", "", "the replay's commit")
	reportPath := flag.String("report", "", "the bot commit's report.json")
	vendor := flag.String("vendor", "", "the run's vendor, a named-sources.json key")

	flag.Parse()

	if *parent == "" || *bot == "" || *base == "" || *replay == "" || *reportPath == "" || *vendor == "" {
		fmt.Fprintln(os.Stderr, "usage: catalog-replay -parent SHA -bot SHA -base SHA -replay SHA -report FILE -vendor NAME")
		os.Exit(2)
	}

	ctx := context.Background()

	raw, err := os.ReadFile(*reportPath)
	exitOn(err)

	report, err := reconcile.ParseReport(raw)
	exitOn(err)

	snapshots := make([]reconcile.Snapshot, 0, 4)

	for _, commit := range []string{*parent, *bot, *base, *replay} {
		snap, err := reconcile.ParseSnapshot([]byte(git(ctx, "show", commit+":"+snapshotPath)))
		exitOn(err)

		snapshots = append(snapshots, snap)
	}

	mismatches := reconcile.ReplayMismatches(snapshots[0], snapshots[1], snapshots[2], snapshots[3], report, *vendor)

	files := strings.Fields(git(ctx, "diff", "--name-only", *parent, *bot) + "\n" + git(ctx, "diff", "--name-only", *base, *replay))
	slices.Sort(files)

	for _, file := range slices.Compact(files) {
		if file != snapshotPath && patchID(ctx, *parent, *bot, file) != patchID(ctx, *base, *replay, file) {
			mismatches = append(mismatches, file+": the replay's change to it differs from the bot's")
		}
	}

	for _, mismatch := range mismatches {
		fmt.Fprintln(os.Stdout, mismatch)
	}

	if len(mismatches) > 0 {
		os.Exit(1)
	}
}

// patchID identifies one file's change from one commit to another by its
// hunks, line numbers aside; empty when the file doesn't change.
func patchID(ctx context.Context, from, to, file string) string {
	cmd := exec.CommandContext(ctx, "git", "patch-id", "--stable")
	cmd.Stdin = strings.NewReader(git(ctx, "diff", from, to, "--", file))

	out, err := cmd.Output()
	exitOn(err)

	id, _, _ := strings.Cut(string(out), " ")

	return id
}

// git runs a read-only git command in the checkout and returns its output.
func git(ctx context.Context, args ...string) string {
	out, err := exec.CommandContext(ctx, "git", args...).Output()
	if err != nil {
		exitOn(fmt.Errorf("git %s: %w", strings.Join(args, " "), err))
	}

	return string(out)
}

func exitOn(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-replay:", err)
		os.Exit(2)
	}
}

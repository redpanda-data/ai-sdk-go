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

// Command catalog-probe calls each offering a reconcile run's vendor
// commits add or change once on its host, with a tiny request, before the
// bot PR opens. It reads only data from the vendor legs' folders (their
// snapshots before and after the patch), sends plain HTTP requests rather
// than going through the SDK's catalog, and writes probe-results.json and
// probes.md, the PR body's "Live checks" block, to OUT_DIR. A call that
// didn't work is sent once more. No outcome fails the command: it exits 2
// only when it can't read its inputs or write its results.
//
// Usage: catalog-probe -out OUT_DIR VENDOR_DIR...
// Env: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY and AWS credentials;
// a host without them is listed as skipped.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
)

func main() {
	out := flag.String("out", "", "the folder for probe-results.json and probes.md")

	flag.Parse()

	if *out == "" {
		fmt.Fprintln(os.Stderr, "usage: catalog-probe -out OUT_DIR VENDOR_DIR...")
		os.Exit(2)
	}

	ctx := context.Background()
	h := realHosts()
	h.anthropicKey, h.openaiKey, h.geminiKey = os.Getenv("ANTHROPIC_API_KEY"), os.Getenv("OPENAI_API_KEY"), os.Getenv("GOOGLE_API_KEY")

	// Without credentials every Bedrock probe is skipped, not failed.
	if cfg, err := config.LoadDefaultConfig(ctx); err == nil {
		if _, err := cfg.Credentials.Retrieve(ctx); err == nil {
			h.awsCredentials = cfg.Credentials
		}
	}

	if err := run(ctx, h, flag.Args(), *out, 10*time.Second); err != nil {
		fmt.Fprintln(os.Stderr, "catalog-probe:", err)
		os.Exit(2)
	}
}

// realHosts are the providers' own endpoints, without credentials.
func realHosts() *hosts {
	return &hosts{
		client:       &http.Client{Timeout: time.Minute},
		anthropicURL: "https://api.anthropic.com",
		openaiURL:    "https://api.openai.com",
		geminiURL:    "https://generativelanguage.googleapis.com",
		awsURL:       func(service, region string) string { return "https://" + service + "." + region + ".amazonaws.com" },
	}
}

// run plans the probes for the vendor folders, calls each, calls once more
// each one that didn't work, after one wait for them all, and writes the
// results to out.
func run(ctx context.Context, h *hosts, dirs []string, out string, wait time.Duration) error {
	probes, err := plan(dirs)
	if err != nil {
		return err
	}

	// Not yet called, or called without working.
	pending := func(p Probe) bool { return p.Outcome != outcomeCallable && p.Outcome != outcomeSkipped }
	callPending := func() {
		for i, p := range probes {
			if pending(p) {
				probes[i] = h.probe(ctx, p)
			}
		}
	}

	callPending()

	if slices.ContainsFunc(probes, pending) {
		time.Sleep(wait)
		callPending()
	}

	for _, p := range probes {
		fmt.Fprintf(os.Stderr, "catalog-probe: %s %s %s: %s %s\n", p.Provider, p.ID, p.Region, p.Outcome, p.Detail)
	}

	raw, err := json.MarshalIndent(map[string]any{"probes": probes}, "", "  ")
	if err != nil {
		return err
	}

	if err := os.MkdirAll(out, 0o755); err != nil {
		return err
	}

	if err := os.WriteFile(filepath.Join(out, "probe-results.json"), append(raw, '\n'), 0o644); err != nil { //nolint:gosec // a CI artifact, public anyway
		return err
	}

	return os.WriteFile(filepath.Join(out, "probes.md"), []byte(markdown(probes)), 0o644) //nolint:gosec // a CI artifact, public anyway
}

// markdown is the PR body's "Live checks" block: what didn't work first,
// then what worked, then what wasn't called. Every value in it is an ID the
// offering pattern accepts or a few words shown() kept.
func markdown(probes []Probe) string {
	rank := func(p Probe) int {
		switch p.Outcome {
		case outcomeCallable:
			return 1
		case outcomeSkipped:
			return 2
		}

		return 0
	}

	probes = slices.Clone(probes)
	slices.SortStableFunc(probes, func(a, b Probe) int { return rank(a) - rank(b) })

	checked, failed := 0, 0

	for _, p := range probes {
		if p.Outcome != outcomeSkipped {
			checked++
		}

		if rank(p) == 0 {
			failed++
		}
	}

	var b strings.Builder

	b.WriteString("## Live checks\n\n")
	b.WriteString("Before this PR opened, the run checked each offering the PR adds or changes on its host, " +
		"using the workflow's own code, not this PR's. A check is one tiny request; on Bedrock it first asks, read-only, " +
		"whether the account subscribes to the model, and calls nothing if not. " +
		"A check that didn't work was tried a second time, and skipped rows weren't checked. " +
		"Read any row that didn't work before approving, because the PR opens either way.\n\n")

	switch {
	case checked == 0:
		b.WriteString("**Nothing was checked.**\n\n")
	case failed == 0 && checked == 1:
		b.WriteString("**The 1 checked offering worked.**\n\n")
	case failed == 0:
		fmt.Fprintf(&b, "**All %d checked offerings worked.**\n\n", checked)
	default:
		fmt.Fprintf(&b, "**%d of %d checked offerings didn't work.**\n\n", failed, checked)
	}

	b.WriteString("| Offering | Host | Result | Detail | Request ID |\n|---|---|---|---|---|\n")

	for _, p := range probes {
		offering, host, requestID := "(not shown)", p.Provider, ""
		if p.ID != "" {
			offering = "`" + p.ID + "`"
		}

		if p.Region != "" {
			host += " (" + p.Region + ")"
		}

		if p.RequestID != "" {
			requestID = "`" + p.RequestID + "`"
		}

		fmt.Fprintf(&b, "| %s | %s | %s | %s | %s |\n", offering, host, strings.ReplaceAll(p.Outcome, "_", " "), p.Detail, requestID)
	}

	return b.String()
}

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

// Command catalog-gate is the reconcile workflow's first job, with no model.
// It fingerprints a vendor's source pages and the run's inputs, and decides
// whether the agent needs to run: only when something changed since the last
// successful real run. With -compare it is the record job's check instead:
// what the agent cited that the gate doesn't fingerprint, and whether a skip
// would have lost findings.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// fetchDeadline caps all fetching well inside the gate job's 10 minutes: a
// page still unfetched by then is unreadable, which runs the agent.
const fetchDeadline = 6 * time.Minute

func main() {
	provider := flag.String("provider", "", "vendor: a named-sources.json key, e.g. anthropic")
	mode := flag.String("mode", modeShadow, "shadow (report only, the agent always runs) or enforce")
	force := flag.Bool("force", false, "run the agent whatever the gate finds")
	maxAge := flag.Duration("max-age", 60*time.Hour, "the oldest baseline the gate relies on")
	baselinePath := flag.String("baseline", "", "fingerprints.json of the last successful real run, if any")
	baselineRun := flag.String("baseline-run", "", "that run's ID, for the summary's link")
	outDir := flag.String("out", "", "where to write fingerprints.json, verdict.json, summary.md and text/")
	compare := flag.Bool("compare", false, "instead, compare this run's agent report with the baseline run's")
	gateDir := flag.String("gate", "", "with -compare: the gate's output directory")
	reportPath := flag.String("report", "", "with -compare: this run's report.json")
	baselineReport := flag.String("baseline-report", "", "with -compare: the baseline run's report.json, if any")

	flag.Parse()

	var err error

	switch {
	case *compare && *gateDir != "" && *reportPath != "":
		err = runCompare(*gateDir, *reportPath, *baselineReport)
	case !*compare && *provider != "" && *outDir != "" && (*mode == modeShadow || *mode == modeEnforce):
		err = runGate(*provider, *outDir, *baselinePath, *baselineRun,
			options{Mode: *mode, Force: *force, MaxAge: *maxAge, Now: time.Now().UTC()})
	default:
		fmt.Fprintln(os.Stderr, "usage: catalog-gate -provider P -out DIR [-mode shadow|enforce] [-force] [-max-age D] [-baseline FILE -baseline-run ID]")
		fmt.Fprintln(os.Stderr, "       catalog-gate -compare -gate DIR -report FILE [-baseline-report FILE]")
		os.Exit(2)
	}

	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-gate:", err)
		os.Exit(1)
	}
}

// runGate fingerprints the pages and inputs, decides, writes its files and
// prints run_agent and verdict for $GITHUB_OUTPUT. It runs in the repo root.
func runGate(provider, outDir, baselinePath, baselineRun string, opts options) error {
	sources, err := os.ReadFile(sourcesPath)
	if err != nil {
		return err
	}

	snapshot, err := os.ReadFile(filepath.Join("catalog", "snapshot.json"))
	if err != nil {
		return err
	}

	pages, err := loadPages(sources, snapshot, provider, opts.Now.Format(time.DateOnly))
	if err != nil {
		return err
	}

	if err := os.MkdirAll(filepath.Join(outDir, "text"), 0o750); err != nil {
		return err
	}

	current := fingerprints{
		Version: 1, Provider: provider, Commit: os.Getenv("GITHUB_SHA"), FetchedAt: opts.Now,
		Sources: map[string]source{}, Inputs: hashInputs(inputPaths),
	}

	ctx, cancel := context.WithTimeout(context.Background(), fetchDeadline)
	defer cancel()

	fetch := newFetcher()

	for _, p := range pages {
		s, text := fingerprintPage(ctx, p, fetch)
		current.Sources[p.ID] = s

		if text != "" {
			if err := os.WriteFile(filepath.Join(outDir, "text", p.ID+".txt"), []byte(text), 0o600); err != nil {
				return err
			}
		}
	}

	var baseline *fingerprints

	if raw, err := os.ReadFile(baselinePath); err == nil {
		var saved fingerprints
		if json.Unmarshal(raw, &saved) == nil {
			baseline = &saved
		}
	}

	d := decide(current, baseline, opts)

	for name, value := range map[string]any{"fingerprints.json": current, "verdict.json": d} {
		raw, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			return err
		}

		if err := os.WriteFile(filepath.Join(outDir, name), append(raw, '\n'), 0o600); err != nil {
			return err
		}
	}

	baselineURL := ""
	if server, repo := os.Getenv("GITHUB_SERVER_URL"), os.Getenv("GITHUB_REPOSITORY"); baselineRun != "" && server != "" && repo != "" {
		baselineURL = server + "/" + repo + "/actions/runs/" + baselineRun
	}

	if err := os.WriteFile(filepath.Join(outDir, "summary.md"), []byte(summary(d, baselineURL)), 0o600); err != nil {
		return err
	}

	_, err = fmt.Fprintf(os.Stdout, "run_agent=%t\nverdict=%s\n", d.RunAgent, d.Verdict)

	return err
}

// runCompare prints the record job's comparison for the job summary.
func runCompare(gateDir, reportPath, baselineReportPath string) error {
	report, err := os.ReadFile(reportPath)
	if err != nil {
		return err
	}

	var pages fingerprints

	var d decision

	for name, into := range map[string]any{"fingerprints.json": &pages, "verdict.json": &d} {
		raw, err := os.ReadFile(filepath.Join(gateDir, name))
		if err != nil {
			return err
		}

		if err := json.Unmarshal(raw, into); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
	}

	baseline, _ := os.ReadFile(baselineReportPath)

	text, err := compareRuns(report, baseline, pages, d.Same)
	if err != nil {
		return err
	}

	_, err = fmt.Fprint(os.Stdout, text)

	return err
}

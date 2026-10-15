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

// Command catalog-report validates the reconcile-models agent's structured
// report and renders the PR body from it. It writes nothing when the report
// is invalid, so the workflow fails closed.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// maxBodyBytes keeps the PR body under GitHub's limit of 65,536 characters.
const maxBodyBytes = 64000

func main() {
	reportPath := flag.String("report", "", "the agent's report.json")
	provider := flag.String("provider", "", "provider name, e.g. anthropic")
	hostList := flag.String("hosts", "", "comma-separated named-source hosts")
	diffPath := flag.String("diff", "", "optional snapshot diff to include")
	outPath := flag.String("out", "", "where to write the PR body")
	codeFilesPath := flag.String("code-files", "", "optional file listing the non-data files the patch changes, one per line")
	rulesPath := flag.String("rules", "", "RULES.md, for the rule each row names")
	decisionsPath := flag.String("decisions", "", "optional decisions.yaml, for the held table")
	baseURL := flag.String("base-url", "", "optional link prefix for repo files at the base commit, e.g. https://github.com/OWNER/REPO/blob/SHA")
	beforePath := flag.String("base-snapshot", "", "optional catalog/snapshot.json before the patch; with -snapshot, each change gets an effect note")
	afterPath := flag.String("snapshot", "", "optional catalog/snapshot.json after the patch")
	routesBeforePath := flag.String("base-routes", "", "optional catalog-routes output before the patch; with -routes, the body names routing changes")
	routesAfterPath := flag.String("routes", "", "optional catalog-routes output after the patch")

	flag.Parse()

	if *reportPath == "" || *provider == "" || *hostList == "" || *rulesPath == "" || *outPath == "" || (*beforePath == "") != (*afterPath == "") || (*routesBeforePath == "") != (*routesAfterPath == "") {
		fmt.Fprintln(os.Stderr, "usage: catalog-report -report FILE -provider P -hosts h1,h2 -rules FILE [-diff FILE] [-code-files FILE] [-decisions FILE] [-base-url URL] [-base-snapshot FILE -snapshot FILE] [-base-routes FILE -routes FILE] -out FILE")
		os.Exit(2)
	}

	data, err := os.ReadFile(*reportPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report: no report from the agent job:", err)
		os.Exit(1)
	}

	report, err := reconcile.ParseReport(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}

	rawRules, err := os.ReadFile(*rulesPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}

	rules, problems := reconcile.ParseRules(rawRules)
	if len(problems) > 0 {
		for _, problem := range problems {
			fmt.Fprintln(os.Stderr, "catalog-report: RULES.md:", problem)
		}

		os.Exit(1)
	}

	if problems := Validate(report, strings.Split(*hostList, ","), rules); len(problems) > 0 {
		for _, problem := range problems {
			fmt.Fprintln(os.Stderr, "catalog-report:", problem)
		}

		os.Exit(1)
	}

	diff := ""

	if *diffPath != "" {
		raw, err := os.ReadFile(*diffPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "catalog-report:", err)
			os.Exit(1)
		}

		diff = string(raw)
	}

	var codeFiles []string

	if *codeFilesPath != "" {
		raw, err := os.ReadFile(*codeFilesPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "catalog-report:", err)
			os.Exit(1)
		}

		// One path per line; spaces stay in the path, so ValidateCodeFiles rejects them.
		codeFiles = strings.FieldsFunc(string(raw), func(r rune) bool { return r == '\n' })
		if problems := ValidateCodeFiles(codeFiles); len(problems) > 0 {
			for _, problem := range problems {
				fmt.Fprintln(os.Stderr, "catalog-report:", problem)
			}

			os.Exit(1)
		}
	}

	base := Base{URL: *baseURL, Rules: rules, Today: time.Now().UTC().Format(time.DateOnly)}

	if *decisionsPath != "" {
		raw, err := os.ReadFile(*decisionsPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "catalog-report:", err)
			os.Exit(1)
		}

		base.Decisions, err = reconcile.ParseDecisions(raw)
		if err != nil {
			fmt.Fprintln(os.Stderr, "catalog-report: decisions.yaml:", err)
			os.Exit(1)
		}
	}

	// The patched snapshot and routes come from code the agent wrote: one
	// that doesn't parse fails the run rather than rendering without it. The
	// usage check above allows both of a pair or neither.
	if *beforePath != "" {
		before, after := mustLoad(*beforePath, reconcile.ParseSnapshot), mustLoad(*afterPath, reconcile.ParseSnapshot)
		base.Before, base.After = &before, &after
	}

	if *routesBeforePath != "" {
		base.RoutesBefore, base.RoutesAfter = mustLoad(*routesBeforePath, ParseRoutes), mustLoad(*routesAfterPath, ParseRoutes)
	}

	body := RenderCapped(report, *provider, diff, codeFiles, base, maxBodyBytes)
	if err := os.WriteFile(*outPath, []byte(body), 0o600); err != nil { //nolint:gosec,nolintlint // -out is set by pr-job.sh, not by the report; nolintlint: older linters don't know this gosec rule
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}
}

// mustLoad reads and parses one input file, or exits.
func mustLoad[T any](path string, parse func([]byte) (T, error)) T {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}

	parsed, err := parse(raw)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", path+":", err)
		os.Exit(1)
	}

	return parsed
}

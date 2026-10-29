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
	"cmp"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

// maxBodyBytes keeps the PR body under GitHub's limit of 65,536 characters.
const maxBodyBytes = 64000

// namedSource is one vendor's entry in named-sources.json: the packages its
// run owns and the hosts its sources may cite.
type namedSource struct {
	Providers []string `json:"providers"`
	Hosts     []string `json:"hosts"`
}

func parseSources(raw []byte) (map[string]namedSource, error) {
	var vendors map[string]namedSource
	if err := json.Unmarshal(raw, &vendors); err != nil {
		return nil, fmt.Errorf("named sources: %w", err)
	}

	return vendors, nil
}

func main() {
	var vendorDirs, failed []string

	flag.Func("vendor", "NAME=DIR, once per vendor in the order the run applied them: DIR holds its report.json and, when its patch changed the catalogue, base-snapshot.json and snapshot.json", func(v string) error {
		vendorDirs = append(vendorDirs, v)
		return nil
	})
	flag.Func("failed", "NAME of a vendor whose run failed, once per vendor", func(v string) error {
		failed = append(failed, v)
		return nil
	})

	sourcesPath := flag.String("sources", "", "named-sources.json, for each vendor's hosts")
	diffPath := flag.String("diff", "", "optional snapshot diff to include")
	outPath := flag.String("out", "", "where to write the PR body")
	codeFilesPath := flag.String("code-files", "", "optional file listing the non-data files the patch changes, one per line")
	rulesPath := flag.String("rules", "", "RULES.md, for the rule each row names")
	decisionsPath := flag.String("decisions", "", "optional decisions.yaml, for the held table")
	baseURL := flag.String("base-url", "", "optional link prefix for repo files at the base commit, e.g. https://github.com/OWNER/REPO/blob/SHA")
	routesBeforePath := flag.String("base-routes", "", "optional catalog-routes output before the patch; with -routes, the body names routing changes")
	routesAfterPath := flag.String("routes", "", "optional catalog-routes output after the patch")
	baseBranch := flag.String("base-branch", "main", "the branch each run rebuilds the bot branch from, named in the body")

	flag.Parse()

	if len(vendorDirs)+len(failed) == 0 || *sourcesPath == "" || *rulesPath == "" || *outPath == "" || (*routesBeforePath == "") != (*routesAfterPath == "") {
		fmt.Fprintln(os.Stderr, "usage: catalog-report -vendor NAME=DIR [-vendor ...] [-failed NAME ...] -sources FILE -rules FILE [-diff FILE] [-code-files FILE] [-decisions FILE] [-base-url URL] [-base-routes FILE -routes FILE] [-base-branch B] -out FILE")
		os.Exit(2)
	}

	// The branch goes into the body as plain text.
	if *baseBranch == "" || !safeValue(*baseBranch) {
		fmt.Fprintf(os.Stderr, "catalog-report: bad base branch %q\n", *baseBranch)
		os.Exit(1)
	}

	sources := mustLoad(*sourcesPath, parseSources)

	rawRules, err := os.ReadFile(*rulesPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}

	rules, problems := reconcile.ParseRules(rawRules)
	exitOn("catalog-report: RULES.md:", problems)

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
		exitOn("catalog-report:", ValidateCodeFiles(codeFiles))
	}

	base := Base{URL: *baseURL, Rules: rules, Today: time.Now().UTC().Format(time.DateOnly), Branch: *baseBranch}

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

	if *routesBeforePath != "" {
		base.RoutesBefore, base.RoutesAfter = mustLoad(*routesBeforePath, ParseRoutes), mustLoad(*routesAfterPath, ParseRoutes)
	}

	vendors := make([]Vendor, 0, len(vendorDirs)+len(failed))
	named := func(name string) {
		if _, ok := sources[name]; !ok {
			failf("no named sources for vendor %q", name)
		}

		if slices.ContainsFunc(vendors, func(v Vendor) bool { return v.Name == name }) {
			failf("vendor %q named twice", name)
		}
	}

	for _, arg := range vendorDirs {
		name, dir, found := strings.Cut(arg, "=")
		if !found || dir == "" {
			failf("-vendor %q isn't NAME=DIR", arg)
		}

		named(name)

		vendor := loadVendor(name, dir)
		// Each vendor's sources must be on its own hosts, and its patched
		// catalogue must agree with its new_model and new_copy rows.
		exitOn("catalog-report: "+name+":", Validate(vendor.Report, sources[name].Hosts, rules))
		exitOn("catalog-report: "+name+":", Base{Before: vendor.Before, After: vendor.After}.ValidateAdded(vendor.Report, name))
		vendors = append(vendors, vendor)
	}

	for _, name := range failed {
		named(name)
		vendors = append(vendors, Vendor{Name: name, Failed: true})
	}

	// The whole PR's catalogue, for the routing changes' prices: before the
	// first vendor's patch, and after the last one's.
	for _, vendor := range vendors {
		if vendor.Before != nil {
			base.Before = cmp.Or(base.Before, vendor.Before)
			base.After = vendor.After
		}
	}

	body := RenderCapped(vendors, diff, codeFiles, base, maxBodyBytes)
	if err := os.WriteFile(*outPath, []byte(body), 0o600); err != nil { //nolint:gosec,nolintlint // -out is set by pr-job.sh, not by the report; nolintlint: older linters don't know this gosec rule
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}
}

// loadVendor reads a vendor's report from dir and, when its patch changed
// the catalogue, the catalogue before and after it; or exits. The patched
// catalogue comes from code the agent wrote: one that doesn't parse fails
// the run rather than rendering without it. A report with changes needs
// both, so its effect notes and "Changed with no row" can't go missing.
func loadVendor(name, dir string) Vendor {
	vendor := Vendor{Name: name, Report: mustLoad(filepath.Join(dir, "report.json"), reconcile.ParseReport)}

	before, after := filepath.Join(dir, "base-snapshot.json"), filepath.Join(dir, "snapshot.json")
	exists := func(path string) bool {
		_, err := os.Stat(path)
		return !errors.Is(err, os.ErrNotExist)
	}

	switch {
	case exists(before) != exists(after):
		failf("%s: %s needs both base-snapshot.json and snapshot.json, or neither", name, dir)
	case exists(before):
		b, a := mustLoad(before, reconcile.ParseSnapshot), mustLoad(after, reconcile.ParseSnapshot)
		vendor.Before, vendor.After = &b, &a
	case len(vendor.Report.Changes) > 0:
		failf("%s: %s has changes but no base-snapshot.json and snapshot.json", name, dir)
	}

	return vendor
}

// failf prints one problem and exits.
func failf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "catalog-report: "+format+"\n", args...)
	os.Exit(1)
}

// exitOn prints each problem after prefix and exits when there are any.
func exitOn(prefix string, problems []string) {
	if len(problems) == 0 {
		return
	}

	for _, problem := range problems {
		fmt.Fprintln(os.Stderr, prefix, problem)
	}

	os.Exit(1)
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

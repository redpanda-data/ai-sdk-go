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

	flag.Parse()

	if *reportPath == "" || *provider == "" || *hostList == "" || *outPath == "" {
		fmt.Fprintln(os.Stderr, "usage: catalog-report -report FILE -provider P -hosts h1,h2 [-diff FILE] [-code-files FILE] -out FILE")
		os.Exit(2)
	}

	data, err := os.ReadFile(*reportPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report: no report from the agent job:", err)
		os.Exit(1)
	}

	report, err := Parse(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}

	if problems := Validate(report, strings.Split(*hostList, ",")); len(problems) > 0 {
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

	body := RenderCapped(report, *provider, diff, codeFiles, maxBodyBytes)
	if err := os.WriteFile(*outPath, []byte(body), 0o600); err != nil { //nolint:gosec,nolintlint // -out is set by pr-job.sh, not by the report; nolintlint: older linters don't know this gosec rule
		fmt.Fprintln(os.Stderr, "catalog-report:", err)
		os.Exit(1)
	}
}

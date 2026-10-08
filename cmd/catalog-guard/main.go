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

// Command catalog-guard checks that a patched catalogue file differs from
// its base only in data: literal values, new constants and catalogue
// entries. The reconcile-models workflow runs it on every patch before
// regenerating the snapshot.
package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	basePath := flag.String("base", "", "the file before the patch")
	patchedPath := flag.String("patched", "", "the file after the patch")

	flag.Parse()

	if *basePath == "" || *patchedPath == "" {
		fmt.Fprintln(os.Stderr, "usage: catalog-guard -base FILE -patched FILE")
		os.Exit(2)
	}

	base, err := os.ReadFile(*basePath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-guard:", err)
		os.Exit(2)
	}

	patched, err := os.ReadFile(*patchedPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-guard:", err)
		os.Exit(2)
	}

	problems := Check(base, patched)
	for _, problem := range problems {
		fmt.Fprintf(os.Stderr, "catalog-guard: %s: %s\n", *patchedPath, problem)
	}

	if len(problems) > 0 {
		os.Exit(1)
	}
}

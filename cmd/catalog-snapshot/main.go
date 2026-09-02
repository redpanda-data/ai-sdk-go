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

// Command catalog-snapshot renders every provider's model catalog into the
// committed snapshot artifact (catalog/snapshot.json).
//
// The snapshot is the review surface for catalog changes — every authored
// edit and its time-independent derivations (price tier, replacement)
// show up as a plain JSON diff — and the read format for non-Go consumers
// such as the AI Gateway console. TestCommittedSnapshotIsFresh fails the
// unit-test job when the committed artifact is stale; -check does the same
// from the command line.
//
// This command imports every provider package; the catalog library itself
// deliberately does not (enforced by catalog's architecture test).
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"

	"github.com/redpanda-data/ai-sdk-go/catalog/snapshot"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func main() {
	out := flag.String("out", "catalog/snapshot.json", "path to write the snapshot to")
	check := flag.Bool("check", false, "verify the file at -out matches the generated snapshot instead of writing")

	flag.Parse()

	var buf bytes.Buffer

	err := snapshot.Encode(&buf,
		anthropic.Catalog(),
		bedrock.Catalog(),
		google.Catalog(),
		openai.Catalog(),
	)
	if err != nil {
		fmt.Fprintln(os.Stderr, "catalog-snapshot:", err)
		os.Exit(1)
	}

	if *check {
		existing, err := os.ReadFile(*out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "catalog-snapshot: read %s: %v\n", *out, err)
			os.Exit(1)
		}

		if !bytes.Equal(existing, buf.Bytes()) {
			fmt.Fprintf(os.Stderr, "catalog-snapshot: %s is stale — run `task catalog:snapshot` and commit the result\n", *out)
			os.Exit(1)
		}

		return
	}

	if err := os.WriteFile(*out, buf.Bytes(), 0o600); err != nil {
		fmt.Fprintln(os.Stderr, "catalog-snapshot:", err)
		os.Exit(1)
	}
}

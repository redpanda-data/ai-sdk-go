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

package main

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

const baseSrc = `package anthropic

import (
	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

const (
	ModelA = "model-a"
)

var caps = func() int { return 1 }()

func Catalog() *catalog.Catalog { return catalog.MustNew(entries()) }

func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:      ModelA,
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-01-01")},
			Pricing: pricing.FlatInfoFromRates(pricing.NewRates(1.00, 2.00, 0.10).WithCacheCreation(1.25, 2.00, 0)),
		},
	}
}
`

const newEntry = `
		{
			ID:      ModelB,
			Life:    catalog.Lifecycle{Available: catalog.MustDate("2026-02-01")},
			Pricing: pricing.FlatInfoFromRates(pricing.NewRates(3.00, 6.00, 0.30)),
		},
	}
}
`

type replacement struct{ old, new string }

func apply(t *testing.T, src string, edits []replacement, appendSrc string) string {
	t.Helper()

	for _, edit := range edits {
		if !strings.Contains(src, edit.old) {
			t.Fatalf("fixture does not contain %q", edit.old)
		}

		src = strings.Replace(src, edit.old, edit.new, 1)
	}

	return src + appendSrc
}

func TestCheck(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		edits     []replacement
		appendSrc string
		wantErr   string // empty means the patch must be allowed
	}{
		{name: "unchanged"},
		{name: "price change", edits: []replacement{{"1.00, 2.00, 0.10", "1.50, 2.00, 0.10"}}},
		{name: "date change", edits: []replacement{{`"2026-01-01"`, `"2026-01-15"`}}},
		{name: "comment change", edits: []replacement{{"ID:      ModelA,", "ID:      ModelA, // source: vendor page"}}},
		{name: "new entry and const", edits: []replacement{
			{`ModelA = "model-a"`, "ModelA = \"model-a\"\n\tModelB = \"model-b\""},
			{"\n\t}\n}\n", newEntry},
		}},
		{name: "new import", edits: []replacement{{"import (", "import (\n\t\"os/exec\""}}, wantErr: "imports changed"},
		{name: "new init", appendSrc: "\nfunc init() { _ = 1 }\n", wantErr: "new function init"},
		{name: "body change", edits: []replacement{{"return catalog.MustNew(entries())", "return nil"}}, wantErr: "body changed: Catalog"},
		{name: "entries gains code", edits: []replacement{{"\treturn []catalog.Entry{", "\t_ = caps\n\treturn []catalog.Entry{"}}, wantErr: "body changed: entries"},
		{name: "new function literal", edits: []replacement{{"ID:      ModelA,", "ID:      func() string { return ModelA }(),"}}, wantErr: "function literals changed"},
		{name: "new call", edits: []replacement{{"ID:      ModelA,", "ID:      catalog.Download(ModelA),"}}, wantErr: "new call: catalog.Download"},
		{name: "new var", appendSrc: "\nvar extra = 1\n", wantErr: "vars changed"},
		{name: "const with call", edits: []replacement{{`ModelA = "model-a"`, `ModelA = catalog.Make("a")`}}, wantErr: "non-literal value"},
		{name: "invalid go", appendSrc: "\nfunc {", wantErr: "parse patched"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			patched := apply(t, baseSrc, tt.edits, tt.appendSrc)

			problems := Check([]byte(baseSrc), []byte(patched))
			if tt.wantErr == "" {
				assert.Empty(t, problems)
				return
			}

			assert.NotEmpty(t, problems)
			assert.Contains(t, strings.Join(problems, "\n"), tt.wantErr)
		})
	}
}

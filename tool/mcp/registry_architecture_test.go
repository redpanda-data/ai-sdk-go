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

package mcp

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestContextRegistryHasNoCatalogTimers(t *testing.T) {
	t.Parallel()

	files, err := filepath.Glob("registry*.go")
	require.NoError(t, err)
	require.NotEmpty(t, files)

	checked := 0

	for _, filename := range files {
		if strings.HasSuffix(filename, "_test.go") {
			continue
		}

		checked++

		t.Run(filename, func(t *testing.T) {
			t.Parallel()

			positions := token.NewFileSet()
			file, err := parser.ParseFile(positions, filename, nil, 0)
			require.NoError(t, err)

			aliases := make(map[string]bool)

			for _, imported := range file.Imports {
				path, err := strconv.Unquote(imported.Path.Value)
				require.NoError(t, err)

				if path != "time" {
					continue
				}

				name := "time"
				if imported.Name != nil {
					name = imported.Name.Name
				}

				aliases[name] = true
			}

			ast.Inspect(file, func(node ast.Node) bool {
				var name string

				switch expression := node.(type) {
				case *ast.SelectorExpr:
					if pkg, ok := expression.X.(*ast.Ident); ok && aliases[pkg.Name] {
						name = expression.Sel.Name
					}
				case *ast.Ident:
					if aliases["."] {
						name = expression.Name
					}
				}

				switch name {
				case "NewTicker", "Tick", "NewTimer", "After", "AfterFunc", "Sleep":
					assert.Fail(t, "contextual registry scheduling is forbidden", "%s: time.%s could schedule background catalog discovery; expiry must stay on demand", positions.Position(node.Pos()), name)
				}

				return true
			})
		})
	}

	assert.Positive(t, checked, "the guard must inspect contextual registry production files, not the legacy client poller")
}

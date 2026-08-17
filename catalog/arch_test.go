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

package catalog

import (
	"go/build"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPackageDependencies pins the architecture: the catalog must be
// linkable by a gateway without dragging in any provider SDK. Only llm,
// pricing, and the stdlib are allowed imports.
func TestPackageDependencies(t *testing.T) {
	t.Parallel()

	pkg, err := build.ImportDir(".", 0)
	require.NoError(t, err)

	allowed := map[string]bool{
		"github.com/redpanda-data/ai-sdk-go/llm":     true,
		"github.com/redpanda-data/ai-sdk-go/pricing": true,
	}

	for _, imp := range pkg.Imports {
		if !strings.Contains(imp, ".") {
			continue // stdlib
		}

		assert.True(t, allowed[imp], "catalog must not import %q — it has to stay linkable without provider SDKs", imp)
	}
}

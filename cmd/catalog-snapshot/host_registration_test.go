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
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/reconcile"
)

func TestEveryOfferingHostIsTracked(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	var tracked map[string]struct {
		Providers []string `json:"providers"`
	}
	require.NoError(t, json.Unmarshal(raw, &tracked))

	packages := make(map[string]string, len(reconcile.PackageProviders))
	for pkg, provider := range reconcile.PackageProviders {
		packages[provider] = pkg
	}

	for _, cat := range allCatalogs() {
		pkg, ok := packages[string(cat.Provider())]
		require.Truef(t, ok, "catalog provider %s has no reconciliation package", cat.Provider())

		for _, offering := range cat.All() {
			publisher := string(offering.Facts().Publisher)
			assert.Containsf(t, tracked[publisher].Providers, pkg,
				"%s/%s: the %s reconciliation run does not own host package %s", cat.Provider(), offering.ID, publisher, pkg)
		}
	}
}

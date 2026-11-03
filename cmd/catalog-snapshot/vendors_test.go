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
)

// TestEveryPublisherIsTracked fails on a catalogue model whose vendor isn't a
// key of named-sources.json, the vendors the team agreed to track (RULES.md,
// vendor-excluded). Adding a vendor is a reviewed change to that file, made
// before its models.
func TestEveryPublisherIsTracked(t *testing.T) {
	t.Parallel()

	raw, err := os.ReadFile(filepath.Join("..", "..", ".github", "reconcile-models", "named-sources.json"))
	require.NoError(t, err)

	var tracked map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &tracked))

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			publisher := string(o.Facts().Publisher)
			assert.Containsf(t, tracked, publisher,
				"%s/%s: its vendor %q isn't tracked; add it to .github/reconcile-models/named-sources.json first", cat.Provider(), o.ID, publisher)
		}
	}
}

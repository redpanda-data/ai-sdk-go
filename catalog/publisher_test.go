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

package catalog_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

// TestEveryFactsRecordDeclaresItsPublisher pins each authored publisher
// against the vendor segment of the ModelID it sits under. The two are
// authored separately, so nothing in the build path stops a new model
// naming one vendor in its ID and another in its Publisher.
func TestEveryFactsRecordDeclaresItsPublisher(t *testing.T) {
	t.Parallel()

	registry := catalog.DefaultRegistry()
	require.NotEmpty(t, registry)

	for id, facts := range registry {
		vendor, _, ok := strings.Cut(string(id), "/")
		require.Truef(t, ok, "%s carries no vendor segment", id)
		assert.Equalf(t, vendor, string(facts.Publisher), "%s publisher", id)
	}
}

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

package vertex_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func TestCatalogTokenCounting(t *testing.T) {
	t.Parallel()

	claude := []string{
		"claude-fable-5-2", "claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
		"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5@20251101",
		"claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5@20250929",
		"claude-haiku-5-5", "claude-haiku-4-5",
	}

	for _, id := range claude {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			o, ok := vertex.Catalog().Lookup(id)
			require.True(t, ok)
			assert.ElementsMatch(t, []catalog.WireAPI{
				{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportVertex},
			}, o.TokenCounting)
		})
	}

	var counted []string

	for _, o := range vertex.Catalog().All() {
		if len(o.TokenCounting) == 0 {
			assert.True(t, strings.HasPrefix(o.ID, "gemini-"), "%s: only Gemini is left unknown in this change", o.ID)

			continue
		}

		counted = append(counted, o.ID)
	}

	assert.ElementsMatch(t, claude, counted)
}

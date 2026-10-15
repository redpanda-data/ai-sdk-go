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

package anthropic_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	ids := []string{
		"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
		"claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-5-5", "claude-sonnet-5",
		"claude-sonnet-4-6", "claude-sonnet-4-5", "claude-haiku-5-5", "claude-haiku-4-5",
		"claude-opus-4-6", "claude-opus-4-5", "claude-opus-4-1",
	}

	gotIDs := make([]string, 0, len(ids))
	for _, o := range anthropic.Catalog().All() {
		gotIDs = append(gotIDs, o.ID)
	}

	assert.ElementsMatch(t, ids, gotIDs)

	for _, id := range ids {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			o, ok := anthropic.Catalog().Lookup(id)
			require.True(t, ok)
			assert.ElementsMatch(t, []catalog.WireAPI{
				{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect},
			}, o.WireAPIs)
		})
	}
}

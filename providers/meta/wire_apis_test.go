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

package meta_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/meta"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	all := meta.Catalog().All()
	require.Len(t, all, 2)

	for _, id := range []string{"muse-spark-1.3", "muse-wonder-1.0"} {
		o, ok := meta.Catalog().Lookup(id)
		require.True(t, ok)
		assert.Equal(t, id, o.ID)
		assert.Equal(t, catalog.ModelID("meta/"+id), o.Model)
		assert.ElementsMatch(t, []catalog.WireAPI{
			{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
			{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		}, o.WireAPIs)
	}
}

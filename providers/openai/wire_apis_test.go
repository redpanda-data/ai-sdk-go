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

package openai_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		ids  []string
		apis []catalog.WireAPI
	}{
		{
			ids: []string{
				"gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.1", "gpt-5.2",
				"gpt-5.2-chat-latest", "gpt-5.3-chat-latest", "gpt-6-astra",
				"gpt-6-sol", "gpt-6-luna", "gpt-6.1-sol", "gpt-6.1-terra",
				"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.5",
				"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-4.1", "gpt-4.1-mini",
				"o3", "o4-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
			},
			apis: []catalog.WireAPI{
				{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
				{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
			},
		},
		{
			ids: []string{"gpt-5.2-pro", "o1-pro", "o3-pro"},
			apis: []catalog.WireAPI{
				{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
			},
		},
	}

	var wantIDs []string

	for _, tc := range cases {
		for _, id := range tc.ids {
			wantIDs = append(wantIDs, id)
			t.Run(id, func(t *testing.T) {
				t.Parallel()

				o, ok := openai.Catalog().Lookup(id)
				require.True(t, ok)
				assert.Equal(t, id, o.ID)
				assert.ElementsMatch(t, tc.apis, o.WireAPIs)
			})
		}
	}

	gotIDs := make([]string, 0, len(wantIDs))
	for _, o := range openai.Catalog().All() {
		gotIDs = append(gotIDs, o.ID)
	}

	assert.ElementsMatch(t, wantIDs, gotIDs)
}

func TestWireAPIsResolveAliases(t *testing.T) {
	t.Parallel()

	o, ok := openai.Catalog().Resolve("gpt-5.6")
	require.True(t, ok)
	assert.Equal(t, "gpt-5.6-sol", o.ID)
	assert.ElementsMatch(t, []catalog.WireAPI{
		{Protocol: catalog.WireProtocolOpenAIChatCompletions, Transport: catalog.WireTransportDirect},
		{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
	}, o.WireAPIs)
}

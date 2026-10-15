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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		ids      []string
		protocol catalog.WireProtocol
	}{
		{
			ids: []string{
				"gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
				"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
				"gemini-3.1-pro-preview", "gemini-3-flash-preview",
				"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
			},
			protocol: catalog.WireProtocolGeminiGenerateContent,
		},
		{
			ids: []string{
				"claude-fable-5-1", "claude-fable-5", "claude-opus-5-5", "claude-opus-5",
				"claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-opus-4-5@20251101",
				"claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5@20250929",
				"claude-haiku-5-5", "claude-haiku-4-5",
			},
			protocol: catalog.WireProtocolAnthropicMessages,
		},
	}

	var wantIDs []string

	for _, tc := range cases {
		for _, id := range tc.ids {
			wantIDs = append(wantIDs, id)
			t.Run(id, func(t *testing.T) {
				t.Parallel()

				o, ok := vertex.Catalog().Lookup(id)
				require.True(t, ok)
				assert.Equal(t, id, o.ID)
				assert.ElementsMatch(t, []catalog.WireAPI{
					{Protocol: tc.protocol, Transport: catalog.WireTransportVertex},
				}, o.WireAPIs)
			})
		}
	}

	gotIDs := make([]string, 0, len(wantIDs))
	for _, o := range vertex.Catalog().All() {
		gotIDs = append(gotIDs, o.ID)
	}

	assert.ElementsMatch(t, wantIDs, gotIDs)
}

func TestWireAPIsKeepSharedFactsAndProviderIDs(t *testing.T) {
	t.Parallel()

	cases := []struct {
		vertexID string
		directID string
		direct   *catalog.Catalog
		protocol catalog.WireProtocol
	}{
		{"gemini-3.8-flash", "gemini-3.8-flash", google.Catalog(), catalog.WireProtocolGeminiGenerateContent},
		{"claude-opus-4-5@20251101", "claude-opus-4-5", anthropic.Catalog(), catalog.WireProtocolAnthropicMessages},
		{"claude-sonnet-4-5@20250929", "claude-sonnet-4-5", anthropic.Catalog(), catalog.WireProtocolAnthropicMessages},
	}
	for _, tc := range cases {
		t.Run(tc.vertexID, func(t *testing.T) {
			t.Parallel()

			v, ok := vertex.Catalog().Lookup(tc.vertexID)
			require.True(t, ok)
			d, ok := tc.direct.Lookup(tc.directID)
			require.True(t, ok)
			assert.Equal(t, tc.vertexID, v.ID)
			assert.Equal(t, tc.directID, d.ID)
			assert.Equal(t, d.Model, v.Model)
			assert.Equal(t, d.Facts(), v.Facts())
			assert.ElementsMatch(t, []catalog.WireAPI{
				{Protocol: tc.protocol, Transport: catalog.WireTransportVertex},
			}, v.WireAPIs)
			assert.ElementsMatch(t, []catalog.WireAPI{
				{Protocol: tc.protocol, Transport: catalog.WireTransportDirect},
			}, d.WireAPIs)

			resolved, ok := vertex.Catalog().Resolve(tc.directID)
			require.True(t, ok)
			assert.Equal(t, v.ID, resolved.ID)
			assert.ElementsMatch(t, v.WireAPIs, resolved.WireAPIs)
		})
	}
}

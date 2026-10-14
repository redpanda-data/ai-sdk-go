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

package google_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
)

func TestCatalogWireAPIs(t *testing.T) {
	t.Parallel()

	ids := []string{
		"gemini-4-argon", "gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
		"gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash",
		"gemini-3.1-pro-preview", "gemini-3-pro-preview", "gemini-3-flash-preview",
		"gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
	}

	gotIDs := make([]string, 0, len(ids))
	for _, o := range google.Catalog().All() {
		gotIDs = append(gotIDs, o.ID)
	}

	assert.ElementsMatch(t, ids, gotIDs)

	for _, id := range ids {
		t.Run(id, func(t *testing.T) {
			t.Parallel()

			o, ok := google.Catalog().Lookup(id)
			require.True(t, ok)
			assert.ElementsMatch(t, []catalog.WireAPI{
				{Protocol: catalog.WireProtocolGeminiGenerateContent, Transport: catalog.WireTransportDirect},
			}, o.WireAPIs)
		})
	}
}

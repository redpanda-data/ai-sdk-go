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

package fireworks_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestFireworksSDKReplayAllowsUncapturedCatalogOfferings(t *testing.T) {
	t.Parallel()

	report, _ := contractReadReport(t, "testdata/sdk-runtime/messages-live.json")
	captured := sdkRuntimeCapturedCases(t, fireworks.Catalog(), report.Evidence)
	require.NotEmpty(t, captured)

	offerings := fireworks.Catalog().All()
	entries := make([]catalog.Entry, 0, len(offerings)+1)

	for _, offering := range offerings {
		entries = append(entries, offering.Entry)
	}

	extra := entries[0]
	extra.ID = "accounts/fireworks/models/docs-only"
	extra.Capabilities = llm.ModelCapabilities{}
	extra.Reasoning = catalog.ReasoningSupport{}
	extra.Modalities = catalog.Modalities{Input: []catalog.Modality{catalog.ModalityText}, Output: []catalog.Modality{catalog.ModalityText}}
	extra.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect}}
	extra.WireAPISupport = []catalog.WireAPISupport{{API: extra.WireAPIs[0]}}
	entries = append(entries, extra)

	extended, err := catalog.New(fireworks.ProviderName, entries, catalog.WithRequiredWireAPIs())
	require.NoError(t, err)
	assert.Equal(t, captured, sdkRuntimeCapturedCases(t, extended, report.Evidence))
}

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
	"bytes"
	"reflect"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/catalog/snapshot"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func contractCompatibleCatalog(t *testing.T, report contractManifest) string {
	t.Helper()

	_, current := contractInventory(t)
	if report.CatalogFingerprint == current {
		return current
	}
	var previous []catalog.Entry

	for _, o := range fireworks.Catalog().All() {
		require.Empty(t, o.Constraints.SupportedParams, "only the reviewed SDK-admission withdrawal is compatible without new native calls")
		e := o.Entry
		// Reconstruct only the withdrawn v2 admission tuple. Matching the entire
		// captured snapshot hash proves no API, fact, limit, capability or price changed.
		e.Constraints.SupportedParams = []string{"temperature", "top_p", "max_tokens", "stop", "tools", "tool_choice", "response_format", "reasoning_effort"}
		previous = append(previous, e)
	}

	old, err := catalog.New(fireworks.ProviderName, previous)
	require.NoError(t, err)
	var encoded bytes.Buffer
	require.NoError(t, snapshot.Encode(&encoded, old))
	require.Equal(t, report.CatalogFingerprint, contractHash(encoded.Bytes()), "catalog drift exceeds the explicitly reviewed SupportedParams withdrawal")

	return current
}

func contractUnknownSupport(value any, known []string) []string {
	v := reflect.ValueOf(value)
	var missing []string

	for n := range v.NumField() {
		if !slices.Contains(known, v.Type().Field(n).Name) && !v.Field(n).IsZero() {
			missing = append(missing, v.Type().Field(n).Name)
		}
	}

	return missing
}

func contractRequireHandledSupport(t *testing.T, caps llm.ModelCapabilities, reasoning catalog.ReasoningSupport) {
	t.Helper()
	require.Empty(t, contractUnknownSupport(caps, []string{"ToolSearchProtocol", "Streaming", "Tools", "JSONMode", "StructuredOutput", "Vision", "Audio", "MultiTurn", "SystemPrompts", "Reasoning"}), "new nonzero capability requires explicit executable cases")
	require.Empty(t, contractUnknownSupport(reasoning, []string{"Efforts", "Adaptive", "Budget", "ThinkingPrefixCheck"}), "new nonzero reasoning feature requires explicit executable cases")
}

func TestFireworksContractSyntheticFutureFeatureGuard(t *testing.T) {
	t.Parallel()

	for _, value := range []bool{false, true} {
		unknown := contractUnknownSupport(struct {
			Existing      bool
			FutureFeature bool
		}{Existing: true, FutureFeature: value}, []string{"Existing"})
		if value {
			require.Equal(t, []string{"FutureFeature"}, unknown)
		} else {
			require.Empty(t, unknown)
		}
	}
}

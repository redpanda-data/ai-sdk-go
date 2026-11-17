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

	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func TestEveryDiscoveredFireworksIDHasADisposition(t *testing.T) {
	t.Parallel()

	dir := filepath.Join("..", "..", "providers", "fireworks", "testdata", "discovery")
	raw, err := os.ReadFile(filepath.Join(dir, "inference-models.json"))
	require.NoError(t, err)

	type discoveredModel struct {
		ID            string `json:"id"`
		ContextLength *int   `json:"context_length"`
		SupportsChat  bool   `json:"supports_chat"`
	}

	var inventory struct {
		Source   string `json:"source"`
		Scope    string `json:"scope"`
		Response struct {
			Data []discoveredModel `json:"data"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal(raw, &inventory))
	assert.Equal(t, "https://api.fireworks.ai/inference/v1/models", inventory.Source)
	assert.Equal(t, "credential_and_inventory_only_not_invocation_qualification", inventory.Scope)
	require.Len(t, inventory.Response.Data, 30)

	raw, err = os.ReadFile(filepath.Join(dir, "reconciliation.json"))
	require.NoError(t, err)

	var selection struct {
		Inventory string `json:"inventory"`
		Scope     string `json:"scope"`
		Offerings []struct {
			ID          string `json:"id"`
			Disposition string `json:"disposition"`
			Reason      string `json:"reason"`
		} `json:"offerings"`
	}
	require.NoError(t, json.Unmarshal(raw, &selection))
	assert.Equal(t, "inference-models.json", selection.Inventory)
	assert.Equal(t, "candidate_selection_not_invocation_or_protocol_qualification", selection.Scope)

	discovered := make(map[string]bool, len(inventory.Response.Data))

	for _, model := range inventory.Response.Data {
		assert.NotContains(t, discovered, model.ID)
		discovered[model.ID] = true

		if model.ID == "accounts/fireworks/models/qwen3p8-max" {
			assert.Nil(t, model.ContextLength)
			assert.True(t, model.SupportsChat, "an unknown limit does not establish unsupported APIs")
		}
	}

	accounted := make(map[string]bool, len(selection.Offerings))
	candidates := make(map[string]bool)

	for _, model := range selection.Offerings {
		assert.Contains(t, discovered, model.ID)
		assert.NotContains(t, accounted, model.ID)
		accounted[model.ID] = true

		switch model.Disposition {
		case "candidate":
			assert.Contains(t, []string{"current_coding", "inexpensive_background", "historical_protocol_regression"}, model.Reason)
			candidates[model.ID] = true
		case "withheld":
			assert.Contains(t, []string{
				"unpriced_router_contract", "separate_router_contract", "retiring_unqualified", "non_generation_surface",
				"separate_speed_contract", "unknown_host_limit", "outside_qualification_batch", "unknown_exact_token_rates",
				"us_only_capacity", "decision_only_surface", "dynamic_routing_contract",
			}, model.Reason)
		default:
			t.Errorf("%s: unknown disposition %q", model.ID, model.Disposition)
		}

		if model.ID == "accounts/fireworks/models/qwen3p8-max" {
			assert.Equal(t, "unknown_host_limit", model.Reason)
		}

		if model.ID == "accounts/fireworks/models/gpt-oss-120b" {
			assert.Equal(t, "historical_protocol_regression", model.Reason)
		}
	}

	assert.Equal(t, discovered, accounted)
	require.Len(t, candidates, 7)

	for id := range candidates {
		offering, ok := fireworks.Catalog().Lookup(id)
		require.True(t, ok, "historical offering must remain in the catalog: %s", id)
		assert.NotEqual(t, "fireworks", string(offering.Facts().Publisher))
	}
}

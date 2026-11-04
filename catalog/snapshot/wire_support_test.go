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

package snapshot

import (
	"bytes"
	"encoding/json"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func wireSupportFixture(t *testing.T, reverse bool) *catalog.Catalog {
	t.Helper()

	e := validEntry("robin-2", "acme/robin-2")
	e.ResolutionPolicy = catalog.ResolutionPolicyExact
	e.Capabilities.JSONMode = true
	e.Capabilities.Reasoning = true
	e.Capabilities.ToolSearchProtocol = llm.ToolSearchProtocolOpenAI
	e.Reasoning = catalog.ReasoningSupport{
		Efforts: []llm.ReasoningEffort{"low", "high"}, Adaptive: true, Budget: true, ThinkingPrefixCheck: true,
	}
	e.WireAPIs = []catalog.WireAPI{
		{Protocol: catalog.WireProtocolOpenAIResponses, Transport: catalog.WireTransportDirect},
		{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportVertex},
		{Protocol: catalog.WireProtocolAnthropicMessages, Transport: catalog.WireTransportDirect},
	}

	e.WireAPISupport = []catalog.WireAPISupport{
		{API: e.WireAPIs[0], Capabilities: e.Capabilities, Reasoning: e.Reasoning},
		{API: e.WireAPIs[1]},
		{
			API: e.WireAPIs[2], Capabilities: llm.ModelCapabilities{Streaming: true, Tools: true, Reasoning: true},
			Reasoning: catalog.ReasoningSupport{Efforts: []llm.ReasoningEffort{"low"}},
		},
	}
	if reverse {
		slices.Reverse(e.WireAPIs)
		slices.Reverse(e.WireAPISupport)
	}

	c, err := catalog.New("acme", []catalog.Entry{e}, catalog.WithRegistry(testRegistry()))
	require.NoError(t, err)

	return c
}

func TestEncodeWireAPISupportDeterministic(t *testing.T) {
	t.Parallel()

	var a, b bytes.Buffer
	require.NoError(t, Encode(&a, wireSupportFixture(t, false)))
	require.NoError(t, Encode(&b, wireSupportFixture(t, true)))
	assert.Equal(t, a.String(), b.String(), "API support authoring order must not affect the snapshot")

	var snap snapshotDTO
	require.NoError(t, json.Unmarshal(a.Bytes(), &snap))
	assert.Equal(t, 3, snap.SchemaVersion, "additive optional metadata does not change the schema version")
	require.Len(t, snap.Providers, 1)
	require.Len(t, snap.Providers[0].Offerings, 1)
	o := snap.Providers[0].Offerings[0]
	assert.Equal(t, "exact", o.ResolutionPolicy)
	require.Len(t, o.WireAPISupport, 3)
	assert.Equal(t, "anthropic_messages", o.WireAPISupport[0].Protocol)
	assert.Equal(t, "direct", o.WireAPISupport[0].Transport)
	assert.True(t, o.Capabilities["json_mode"])
	assert.False(t, o.WireAPISupport[0].Capabilities["json_mode"])
	assert.Equal(t, []llm.ReasoningEffort{"low"}, o.WireAPISupport[0].Reasoning.Efforts)
	assert.Equal(t, "vertex", o.WireAPISupport[1].Transport)
	assert.False(t, o.WireAPISupport[1].Capabilities["tools"], "explicitly withheld features stay false")
	assert.Equal(t, "openai_responses", o.WireAPISupport[2].Protocol)
	assert.True(t, o.WireAPISupport[2].Capabilities["json_mode"])
	assert.True(t, o.WireAPISupport[2].Capabilities["tool_search"])
	assert.Equal(t, "openai", o.WireAPISupport[2].ToolSearchProtocol)
	assert.True(t, o.WireAPISupport[2].Reasoning.Adaptive)
	assert.True(t, o.WireAPISupport[2].Reasoning.Budget)
	assert.True(t, o.WireAPISupport[2].Reasoning.ThinkingPrefixCheck)
}

func TestWireAPISupportSnapshotCompatibility(t *testing.T) {
	t.Parallel()

	var legacy bytes.Buffer
	require.NoError(t, Encode(&legacy, snapshotFixture(t)))
	assert.NotContains(t, legacy.String(), `"wire_api_support"`)
	assert.NotContains(t, legacy.String(), `"resolution_policy"`)

	var old snapshotDTO
	require.NoError(t, json.Unmarshal([]byte(`{"schema_version":3,"facts":{},"providers":[{"provider":"acme","offerings":[{"id":"robin-2","model":"acme/robin-2","wire_apis":[{"protocol":"openai_responses","transport":"direct"}]}]}]}`), &old))
	oldOffering := old.Providers[0].Offerings[0]
	assert.Nil(t, oldOffering.WireAPISupport, "missing per-API features remain unknown")
	assert.Empty(t, oldOffering.ResolutionPolicy, "old snapshots retain legacy version-stamp resolution")

	var current bytes.Buffer
	require.NoError(t, Encode(&current, wireSupportFixture(t, false)))
	var oldReader struct {
		SchemaVersion int `json:"schema_version"`
		Providers     []struct {
			Provider  string `json:"provider"`
			Offerings []struct {
				ID           string          `json:"id"`
				Model        string          `json:"model"`
				Capabilities map[string]bool `json:"capabilities"`
			} `json:"offerings"`
		} `json:"providers"`
	}
	require.NoError(t, json.Unmarshal(current.Bytes(), &oldReader), "tolerant existing readers ignore optional additions")
	assert.Equal(t, 3, oldReader.SchemaVersion)
	assert.Equal(t, "robin-2", oldReader.Providers[0].Offerings[0].ID)
	assert.Equal(t, "acme/robin-2", oldReader.Providers[0].Offerings[0].Model)
	assert.True(t, oldReader.Providers[0].Offerings[0].Capabilities["json_mode"], "existing field semantics must not change")
}

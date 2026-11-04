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

package catalog

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func wireSupportEntry() Entry {
	e := validEntry("robin-2", "acme/robin-2")
	e.Aliases = []string{"robin-latest"}
	e.Capabilities.JSONMode = true
	e.Capabilities.Reasoning = true
	e.Capabilities.ToolSearchProtocol = llm.ToolSearchProtocolOpenAI
	e.Reasoning = ReasoningSupport{
		Efforts: []llm.ReasoningEffort{"low", "high"}, Adaptive: true, Budget: true, ThinkingPrefixCheck: true,
	}
	e.WireAPIs = []WireAPI{
		{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportVertex},
	}
	e.WireAPISupport = []WireAPISupport{
		{
			API: e.WireAPIs[0], Capabilities: e.Capabilities, Reasoning: e.Reasoning,
		},
		{
			API: e.WireAPIs[1], Capabilities: llm.ModelCapabilities{Tools: true, Streaming: true, Reasoning: true},
			Reasoning: ReasoningSupport{Efforts: []llm.ReasoningEffort{"low"}},
		},
	}

	return e
}

func TestWithRequiredWireAPIs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		required bool
		apis     []WireAPI
		wantErr  bool
	}{
		{name: "legacy nil"},
		{name: "legacy empty", apis: []WireAPI{}},
		{name: "required nil", required: true, wantErr: true},
		{name: "required empty", required: true, apis: []WireAPI{}, wantErr: true},
		{
			name: "required explicit", required: true,
			apis: []WireAPI{{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			first := validEntry("robin-1", "acme/robin-1")
			first.WireAPIs = []WireAPI{{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect}}
			second := validEntry("robin-2", "acme/robin-2")
			second.WireAPIs = tt.apis

			opts := []Option{WithRegistry(testRegistry())}
			if tt.required {
				opts = append(opts, WithRequiredWireAPIs())
			}

			c, err := New("acme", []Entry{first, second}, opts...)
			if tt.wantErr {
				require.ErrorContains(t, err, "WireAPIs must declare at least one binding")
				require.ErrorContains(t, err, `catalog: acme: entries[1] "robin-2":`)
				assert.Nil(t, c)

				return
			}

			require.NoError(t, err)

			o, ok := c.Lookup(second.ID)
			require.True(t, ok)
			assert.Equal(t, tt.apis, o.WireAPIs, "empty legacy declarations must not acquire inferred defaults")
		})
	}
}

func TestWireAPISupportJSONRoundTrip(t *testing.T) {
	t.Parallel()

	e := wireSupportEntry()
	data, err := json.Marshal(e) //nolint:musttag // Entry fingerprints use Go field names, not snapshot DTO tags.
	require.NoError(t, err)
	var decoded Entry
	require.NoError(t, json.Unmarshal(data, &decoded)) //nolint:musttag // Entry fingerprints must also decode with Go field names.
	assert.Equal(t, e, decoded, "entry fingerprints must support structured API bindings without map-key encoding")
}

func TestWireAPISupportValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		mutate func(*Entry)
		want   string
	}{
		{"valid subset", func(*Entry) {}, ""},
		{"legacy unknown", func(e *Entry) { e.WireAPISupport = nil }, ""},
		{"withheld features", func(e *Entry) {
			e.WireAPISupport[0] = WireAPISupport{API: e.WireAPIs[0]}
		}, ""},
		{"duplicate binding", func(e *Entry) {
			e.WireAPISupport = append(e.WireAPISupport, e.WireAPISupport[0])
		}, "duplicates protocol"},
		{"undeclared binding", func(e *Entry) {
			e.WireAPISupport[0].API.Transport = WireTransportBedrockRuntime
		}, ".API is not declared in WireAPIs"},
		{"unknown protocol", func(e *Entry) {
			e.WireAPISupport[0].API.Protocol = "unknown"
		}, ".API is not declared in WireAPIs"},
		{"unknown transport", func(e *Entry) {
			e.WireAPISupport[0].API.Transport = "unknown"
		}, ".API is not declared in WireAPIs"},
		{"API capability required", func(e *Entry) {
			e.WireAPISupport[0].Capabilities.Reasoning = false
		}, ".Reasoning requires API Capabilities.Reasoning"},
		{"effort exceeds model", func(e *Entry) {
			e.WireAPISupport[0].Reasoning.Efforts = []llm.ReasoningEffort{"max"}
		}, `.Reasoning.Efforts contains "max" outside offering reasoning support`},
		{"adaptive exceeds model", func(e *Entry) { e.Reasoning.Adaptive = false }, ".Reasoning.Adaptive exceeds offering reasoning support"},
		{"budget exceeds model", func(e *Entry) { e.Reasoning.Budget = false }, ".Reasoning.Budget exceeds offering reasoning support"},
		{"prefix check exceeds model", func(e *Entry) { e.Reasoning.ThinkingPrefixCheck = false }, ".Reasoning.ThinkingPrefixCheck exceeds offering reasoning support"},
		{"prefix check requires adaptive", func(e *Entry) {
			e.WireAPISupport[0].Reasoning.Adaptive = false
		}, ".Reasoning.ThinkingPrefixCheck requires API Reasoning.Adaptive"},
		{"tool search protocol mismatch", func(e *Entry) {
			e.WireAPISupport[0].Capabilities.ToolSearchProtocol = llm.ToolSearchProtocolAnthropic
		}, ".Capabilities.ToolSearchProtocol exceeds offering capabilities"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			e := wireSupportEntry()
			tt.mutate(&e)

			c, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
			if tt.want != "" {
				require.ErrorContains(t, err, tt.want)
				require.ErrorContains(t, err, `catalog: acme: entries[0] "robin-2":`)
				assert.Nil(t, c)

				return
			}

			require.NoError(t, err)
			assert.NotNil(t, c)
		})
	}
}

func TestWireAPISupportCapabilitiesAreSubsets(t *testing.T) {
	t.Parallel()

	for name, caps := range map[string]llm.ModelCapabilities{
		"ToolSearchProtocol": {ToolSearchProtocol: llm.ToolSearchProtocolOpenAI},
		"Streaming":          {Streaming: true}, "Tools": {Tools: true}, "JSONMode": {JSONMode: true},
		"StructuredOutput": {StructuredOutput: true}, "Vision": {Vision: true}, "Audio": {Audio: true},
		"MultiTurn": {MultiTurn: true}, "SystemPrompts": {SystemPrompts: true}, "Reasoning": {Reasoning: true},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			e := validEntry("robin-2", "acme/robin-2")
			e.Capabilities = llm.ModelCapabilities{}
			e.WireAPIs = []WireAPI{{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect}}
			e.WireAPISupport = []WireAPISupport{{API: e.WireAPIs[0], Capabilities: caps}}
			_, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
			require.ErrorContains(t, err, ".Capabilities."+name+" exceeds offering capabilities")
		})
	}
}

func TestWireAPISupportUnknownDoesNotInherit(t *testing.T) {
	t.Parallel()

	e := wireSupportEntry()
	c := mustCatalog(t, e)
	o, ok := c.Resolve("robin-latest")
	require.True(t, ok)

	messages, ok := o.SupportForWireAPI(e.WireAPIs[1])
	require.True(t, ok)
	assert.True(t, o.Capabilities.JSONMode)
	assert.False(t, messages.Capabilities.JSONMode)
	assert.Equal(t, []llm.ReasoningEffort{"low"}, messages.Reasoning.Efforts)

	_, ok = o.SupportForWireAPI(e.WireAPIs[2])
	assert.False(t, ok, "same protocol through a different transport must remain unknown")

	e.WireAPISupport = []WireAPISupport{{API: e.WireAPIs[1]}}
	o, ok = mustCatalog(t, e).Lookup(e.ID)
	require.True(t, ok)
	support, ok := o.SupportForWireAPI(e.WireAPIs[1])
	require.True(t, ok, "explicitly withholding all features differs from unknown")
	assert.Equal(t, llm.ModelCapabilities{}, support.Capabilities)

	_, ok = o.SupportForWireAPI(e.WireAPIs[0])
	assert.False(t, ok)
	_, ok = (Offering{}).SupportForWireAPI(WireAPI{})
	assert.False(t, ok)
}

func TestWireAPISupportSortedAndCopied(t *testing.T) {
	t.Parallel()

	reads := map[string]func(*testing.T, *Catalog) Offering{
		"lookup": func(t *testing.T, c *Catalog) Offering {
			t.Helper()

			o, ok := c.Lookup("robin-2")
			require.True(t, ok)

			return o
		},
		"alias": func(t *testing.T, c *Catalog) Offering {
			t.Helper()

			o, ok := c.Resolve("robin-latest")
			require.True(t, ok)

			return o
		},
		"all":       func(_ *testing.T, c *Catalog) Offering { return c.All()[0] },
		"offerings": func(_ *testing.T, c *Catalog) Offering { return c.Offerings("acme/robin-2")[0] },
		"view":      func(_ *testing.T, c *Catalog) Offering { return c.At(MustDate("2026-11-04")).Current()[0] },
	}

	for name, read := range reads {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			e := wireSupportEntry()
			authoredAPIs := []WireAPI{e.WireAPISupport[0].API, e.WireAPISupport[1].API}
			c := mustCatalog(t, e)
			assert.Equal(t, authoredAPIs[0], e.WireAPISupport[0].API, "construction must not reorder authored records")
			e.WireAPISupport[0].API.Protocol = mutated
			e.WireAPISupport[0].Reasoning.Efforts[0] = mutated

			o := read(t, c)
			require.Len(t, o.WireAPISupport, 2)
			assert.Equal(t, authoredAPIs[1], o.WireAPISupport[0].API)
			assert.Equal(t, authoredAPIs[0], o.WireAPISupport[1].API)
			assert.Equal(t, []llm.ReasoningEffort{"low", "high"}, o.WireAPISupport[1].Reasoning.Efforts)

			support, ok := o.SupportForWireAPI(authoredAPIs[0])
			require.True(t, ok)

			support.Reasoning.Efforts[0] = mutated
			assert.Equal(t, llm.ReasoningEffort("low"), o.WireAPISupport[1].Reasoning.Efforts[0])

			o.WireAPISupport[0].API.Protocol = mutated
			o.WireAPISupport[1].Reasoning.Efforts[0] = mutated
			again := read(t, c)
			assert.Equal(t, authoredAPIs[1], again.WireAPISupport[0].API)
			assert.Equal(t, []llm.ReasoningEffort{"low", "high"}, again.WireAPISupport[1].Reasoning.Efforts)
			assert.True(t, slices.IsSortedFunc(again.WireAPISupport, func(a, b WireAPISupport) int {
				return compareWireAPIs(a.API, b.API)
			}))
		})
	}
}

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
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWireAPIsValidation(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		apis []WireAPI
		want string
	}{
		{
			name: "empty protocol",
			apis: []WireAPI{{Transport: WireTransportDirect}},
			want: `WireAPIs[0].Protocol "" is not a valid wire protocol`,
		},
		{
			name: "unknown protocol",
			apis: []WireAPI{{Protocol: "unknown", Transport: WireTransportDirect}},
			want: `WireAPIs[0].Protocol "unknown" is not a valid wire protocol`,
		},
		{
			name: "empty transport",
			apis: []WireAPI{{Protocol: WireProtocolAnthropicMessages}},
			want: `WireAPIs[0].Transport "" is not a valid wire transport`,
		},
		{
			name: "unknown transport",
			apis: []WireAPI{{Protocol: WireProtocolAnthropicMessages, Transport: "unknown"}},
			want: `WireAPIs[0].Transport "unknown" is not a valid wire transport`,
		},
		{
			name: "duplicate binding",
			apis: []WireAPI{
				{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect},
				{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect},
				{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect},
			},
			want: `duplicates protocol "openai_responses" and transport "direct"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			e := validEntry("robin-2", "acme/robin-2")
			e.WireAPIs = tt.apis
			_, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
			require.ErrorContains(t, err, tt.want)
			require.ErrorContains(t, err, `catalog: acme: entries[0] "robin-2":`)
		})
	}
}

func TestWireAPIsNormalization(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.WireAPIs = []WireAPI{
		{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportVertex},
		{Protocol: WireProtocolOpenAIChatCompletions, Transport: WireTransportDirect},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect},
		{Protocol: WireProtocolBedrockConverse, Transport: WireTransportBedrockRuntime},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockInvokeModel},
		{Protocol: WireProtocolGeminiGenerateContent, Transport: WireTransportDirect},
		{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportBedrockMantle},
	}
	authored := slices.Clone(e.WireAPIs)
	c := mustCatalog(t, e)
	assert.Equal(t, authored, e.WireAPIs, "normalization must not reorder the author's slice")

	want := []WireAPI{
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockInvokeModel},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportVertex},
		{Protocol: WireProtocolBedrockConverse, Transport: WireTransportBedrockRuntime},
		{Protocol: WireProtocolGeminiGenerateContent, Transport: WireTransportDirect},
		{Protocol: WireProtocolOpenAIChatCompletions, Transport: WireTransportDirect},
		{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportBedrockMantle},
		{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect},
	}
	o, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, want, o.WireAPIs)

	e.WireAPIs[0].Protocol = mutated
	e.WireAPIs[0].Transport = mutated
	again, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, want, again.WireAPIs, "the catalog must not alias authored records")
}

func TestWireAPIsUnspecified(t *testing.T) {
	t.Parallel()

	for _, apis := range [][]WireAPI{nil, {}} {
		e := validEntry("robin-2", "acme/robin-2")
		e.WireAPIs = apis
		c := mustCatalog(t, e)
		o, ok := c.Resolve(e.ID)
		require.True(t, ok)
		assert.Empty(t, o.WireAPIs, "unknown bindings must not grow provider defaults")
	}
}

func TestWireAPIsDoNotRestrictProtocolTransportPairs(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.WireAPIs = []WireAPI{{Protocol: WireProtocolGeminiGenerateContent, Transport: WireTransportBedrockInvokeModel}}
	c := mustCatalog(t, e)
	o, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, e.WireAPIs, o.WireAPIs, "catalog validation checks vocabulary, not vendor support matrices")
}

func TestWireAPIsReadCopies(t *testing.T) {
	t.Parallel()

	lookup := func(t *testing.T, c *Catalog) Offering {
		t.Helper()

		o, ok := c.Lookup("robin-2")
		require.True(t, ok)

		return o
	}
	resolve := func(id string) func(*testing.T, *Catalog) Offering {
		return func(t *testing.T, c *Catalog) Offering {
			t.Helper()

			o, ok := c.Resolve(id)
			require.True(t, ok)

			return o
		}
	}
	first := func(get func(*Catalog) []Offering) func(*testing.T, *Catalog) Offering {
		return func(t *testing.T, c *Catalog) Offering {
			t.Helper()

			os := get(c)
			require.NotEmpty(t, os)

			return os[0]
		}
	}
	tests := []struct {
		name string
		read func(*testing.T, *Catalog) Offering
	}{
		{"lookup", lookup},
		{"resolve", resolve("robin-2")},
		{"alias", resolve("robin-latest")},
		{"snapshot", resolve("robin-2-20260101")},
		{"all", first((*Catalog).All)},
		{"offerings", first(func(c *Catalog) []Offering { return c.Offerings("acme/robin-2") })},
		{"current", first(func(c *Catalog) []Offering { return c.At(MustDate("2026-06-01")).Current() })},
		{"previous", first(func(c *Catalog) []Offering { return c.At(MustDate("2026-06-01")).Previous() })},
		{"deprecated", first(func(c *Catalog) []Offering { return c.At(MustDate("2026-06-01")).Deprecated() })},
		{"retired", first(func(c *Catalog) []Offering { return c.At(MustDate("2028-06-01")).Retired() })},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			apis := []WireAPI{{Protocol: WireProtocolOpenAIResponses, Transport: WireTransportDirect}}
			e := validEntry("robin-2", "acme/robin-2")
			e.Aliases = []string{"robin-latest"}
			e.WireAPIs = apis
			e.Life.Deprecated = MustDate("2026-01-01")
			e.Life.Retires = MustDate("2027-01-01")
			newer := validEntry("robin-3", "acme/robin-3")
			newer.WireAPIs = apis
			c := mustCatalog(t, e, newer)

			o := tt.read(t, c)
			require.Len(t, o.WireAPIs, 1)
			o.WireAPIs[0].Protocol = mutated
			o.WireAPIs[0].Transport = mutated

			again, ok := c.Lookup(o.ID)
			require.True(t, ok)
			assert.Equal(t, apis, again.WireAPIs)
		})
	}
}

func TestWireAPIsDoNotResolveUnregisteredProfiles(t *testing.T) {
	t.Parallel()

	e := validEntry("us.robin-2", "acme/robin-2")
	e.WireAPIs = []WireAPI{{Protocol: WireProtocolBedrockConverse, Transport: WireTransportBedrockRuntime}}
	c := mustCatalog(t, e)

	for _, id := range []string{"robin-2", "eu.robin-2", "global.robin-2", "us.robin-3"} {
		o, ok := c.Resolve(id)
		assert.False(t, ok, "%s must remain unknown", id)
		assert.Empty(t, o.WireAPIs)
	}
}

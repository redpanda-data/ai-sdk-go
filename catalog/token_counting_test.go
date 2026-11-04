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

func TestTokenCountingNormalization(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.TokenCounting = []WireAPI{
		{Protocol: WireProtocolBedrockConverse, Transport: WireTransportBedrockRuntime},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockMantle},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockInvokeModel},
	}
	authored := slices.Clone(e.TokenCounting)
	c := mustCatalog(t, e)
	assert.Equal(t, authored, e.TokenCounting, "normalization must not reorder the author's slice")

	want := []WireAPI{
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockInvokeModel},
		{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockMantle},
		{Protocol: WireProtocolBedrockConverse, Transport: WireTransportBedrockRuntime},
	}
	o, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, want, o.TokenCounting)

	e.TokenCounting[0].Protocol = mutated
	again, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, want, again.TokenCounting, "the catalog must not alias authored records")

	o.TokenCounting[0].Transport = mutated
	again, ok = c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, want, again.TokenCounting, "reads must deep-copy")
}

func TestTokenCountingUnspecified(t *testing.T) {
	t.Parallel()

	for _, apis := range [][]WireAPI{nil, {}} {
		e := validEntry("robin-2", "acme/robin-2")
		e.WireAPIs = []WireAPI{{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportDirect}}
		e.TokenCounting = apis
		c := mustCatalog(t, e)
		o, ok := c.Resolve(e.ID)
		require.True(t, ok)
		assert.Empty(t, o.TokenCounting, "unknown token counting must not default to the wire APIs")
	}
}

func TestTokenCountingIsIndependentOfWireAPIs(t *testing.T) {
	t.Parallel()

	// Bedrock counts on Mantle for profile offerings that serve inference
	// only through bedrock-runtime, so the two sets must not be tied.
	e := validEntry("us.robin-2", "acme/robin-2")
	e.WireAPIs = []WireAPI{{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockRuntime}}
	e.TokenCounting = []WireAPI{{Protocol: WireProtocolAnthropicMessages, Transport: WireTransportBedrockMantle}}
	c := mustCatalog(t, e)
	o, ok := c.Lookup(e.ID)
	require.True(t, ok)
	assert.Equal(t, e.WireAPIs, o.WireAPIs)
	assert.Equal(t, e.TokenCounting, o.TokenCounting)
}

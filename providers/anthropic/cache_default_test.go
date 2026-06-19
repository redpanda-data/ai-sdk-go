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

package anthropic

import (
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestDefaultProviderEnablesCaching pins the default: a provider built the way
// every consumer builds it (no caching option) must have caching ON. Prompt
// caching being opt-in means everyone forgets to opt in, which is exactly the
// regression this guards against.
func TestDefaultProviderEnablesCaching(t *testing.T) {
	t.Parallel()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	assert.True(t, p.EnableCaching,
		"caching must default to ON; the SDK's consumers never call WithCaching()")
}

// TestDefaultCachingEmitsCacheControlMarkers proves the default flows all the
// way through: a model built from a default provider must emit cache_control
// markers on the cacheable prefix (last system block) and on the last message,
// without anyone calling WithCaching().
func TestDefaultCachingEmitsCacheControlMarkers(t *testing.T) {
	t.Parallel()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet45)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleSystem,
				Content: []llm.Part{
					// Realistic system prompt; size is irrelevant to marker
					// emission (Anthropic ignores markers below its token
					// minimum server-side, the SDK always writes them).
					llm.NewTextPart("You are a helpful assistant. " + strings.Repeat("context ", 200)),
				},
			},
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart("Hello")},
			},
		},
	}

	apiReq, err := m.requestMapper.ToProvider(req)
	require.NoError(t, err)

	require.NotEmpty(t, apiReq.System, "system blocks must be present")
	lastSystem := apiReq.System[len(apiReq.System)-1]
	assert.True(t, hasCacheMarker(lastSystem.CacheControl),
		"last system block must carry a cache_control marker by default")

	require.NotEmpty(t, apiReq.Messages, "messages must be present")
	assert.True(t, lastMessageHasCacheMarker(apiReq.Messages[len(apiReq.Messages)-1]),
		"last message must carry a cache_control marker by default")
}

// hasCacheMarker reports whether a cache_control value has been set. The zero
// value leaves Type empty; NewBetaCacheControlEphemeralParam sets it to
// "ephemeral".
func hasCacheMarker(cc anthropic.BetaCacheControlEphemeralParam) bool {
	return cc.Type != ""
}

// lastMessageHasCacheMarker reports whether any text block in the message
// carries a cache_control marker.
func lastMessageHasCacheMarker(msg anthropic.BetaMessageParam) bool {
	for _, block := range msg.Content {
		if block.OfText != nil && hasCacheMarker(block.OfText.CacheControl) {
			return true
		}
	}

	return false
}

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
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseMapper_Metadata(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelClaudeOpus46])

	resp, err := mapper.FromProvider(&anthropic.BetaMessage{
		ID:    "msg_123",
		Model: anthropic.Model("claude-opus-4-6-20260401"),
		Content: []anthropic.BetaContentBlockUnion{{
			Type: blockTypeText,
			Text: "Hello",
		}},
		StopReason: anthropic.BetaStopReasonEndTurn,
		Usage: anthropic.BetaUsage{
			InputTokens:  10,
			OutputTokens: 5,
			ServiceTier:  anthropic.BetaUsageServiceTierStandard,
			Speed:        anthropic.BetaUsageSpeedFast,
			InferenceGeo: "us-east-1",
		},
	})
	require.NoError(t, err)

	assert.Equal(t, llm.ServiceTierDefault, resp.ServiceTier)
	assert.Equal(t, llm.SpeedFast, resp.Speed)
	assert.Equal(t, "us-east-1", resp.InferenceRegion)
	assert.Equal(t, ModelClaudeOpus46, resp.InvokedModelID)
}

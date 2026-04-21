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

package openai

import (
	"testing"

	"github.com/openai/openai-go/v3/responses"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestResponseMapper_Metadata(t *testing.T) {
	t.Parallel()

	mapper := NewResponseMapper(supportedModels[ModelGPT5Mini])

	resp, err := mapper.FromProvider(&responses.Response{
		ID:          "resp_123",
		Model:       ModelGPT5Mini,
		Status:      responses.ResponseStatusCompleted,
		ServiceTier: responses.ResponseServiceTierDefault,
		Output: []responses.ResponseOutputItemUnion{{
			Type: outputTypeMessage,
			Content: []responses.ResponseOutputMessageContentUnion{{
				Type: contentTypeOutputText,
				Text: "Hello",
			}},
		}},
		Usage: responses.ResponseUsage{
			InputTokens:  10,
			OutputTokens: 5,
			TotalTokens:  15,
		},
	})
	require.NoError(t, err)

	assert.Equal(t, llm.ServiceTierDefault, resp.ServiceTier)
	assert.Equal(t, ModelGPT5Mini, resp.InvokedModelID)
}

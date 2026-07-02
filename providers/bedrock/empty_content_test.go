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

package bedrock

import (
	"context"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TestMapAssistantMessage_EmptyContentIsRepaired is the Bedrock parity
// regression for the empty-content replay bug. A max_tokens cut can persist an
// assistant turn whose only block was a partial tool_use the streaming
// finalizer dropped, leaving Content empty. Bedrock's Converse API is
// Anthropic-shaped and rejects a content-less assistant message, so replaying
// that history fails the whole request. The mapper must substitute a single
// minimal text block, exactly like the anthropic mapper.
//
// Caching is ON by default (Bedrock runs Claude), so this also proves the guard
// runs BEFORE the cache-point insertion stage: the empty assistant turn as the
// LAST message must not become a cachePoint-only content block — a text block
// has to exist first.
func TestMapAssistantMessage_EmptyContentIsRepaired(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)
	require.True(t, p.enableCaching, "test relies on default caching to exercise cache-point ordering")

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("run the tools")}},
			// The poisoned turn as the FINAL message — this is where the cache
			// point gets appended, so it's the sharpest test of ordering.
			{Role: llm.RoleAssistant, Content: []llm.Part{}},
		},
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)

	require.Len(t, input.Messages, 2)

	last := input.Messages[len(input.Messages)-1]
	require.Equal(t, types.ConversationRoleAssistant, last.Role)
	require.NotEmpty(t, last.Content, "repaired assistant turn mapped to empty content")

	// The FIRST block must be the substituted text (non-whitespace), proving the
	// guard ran before the cache point was appended.
	textBlock, ok := last.Content[0].(*types.ContentBlockMemberText)
	require.True(t, ok, "first block must be a text block, not a cachePoint — got %T", last.Content[0])
	assert.NotEmpty(t, textBlock.Value, "substituted text block must be non-empty")

	// And there must be at least one real (non-cachePoint) content block.
	hasNonCachePoint := false
	for _, b := range last.Content {
		if _, isCache := b.(*types.ContentBlockMemberCachePoint); !isCache {
			hasNonCachePoint = true
		}
	}
	assert.True(t, hasNonCachePoint, "assistant turn must not be cachePoint-only")
}

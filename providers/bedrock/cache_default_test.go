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

// TestDefaultProviderEnablesCaching pins the default: Bedrock runs Claude, and
// prompt caching being opt-in means consumers forget to opt in. A provider
// built the normal way (no caching option) must have caching ON.
func TestDefaultProviderEnablesCaching(t *testing.T) {
	t.Parallel()

	// WithAWSConfig bypasses the AWS credential/config-file lookup so this runs
	// fully offline.
	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)

	assert.True(t, p.enableCaching,
		"Bedrock caching must default to ON; consumers never call WithCaching()")
}

// TestDefaultCachingInsertsCachePoint proves the default flows through to the
// wire: a model built from a default provider must insert a CachePoint after
// the system block without anyone calling WithCaching().
func TestDefaultCachingInsertsCachePoint(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{Region: "us-east-1"}))
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet46)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	req := &llm.Request{
		Messages: []llm.Message{
			{
				Role:    llm.RoleSystem,
				Content: []llm.Part{llm.NewTextPart("You are a helpful assistant.")},
			},
			{
				Role:    llm.RoleUser,
				Content: []llm.Part{llm.NewTextPart("Hello")},
			},
		},
	}

	input, err := m.requestMapper.ToConverseInput(req)
	require.NoError(t, err)

	assert.True(t, hasSystemCachePoint(input.System),
		"system blocks must carry a CachePoint by default")
}

// hasSystemCachePoint reports whether the system blocks contain a cache point.
func hasSystemCachePoint(blocks []types.SystemContentBlock) bool {
	for _, b := range blocks {
		if _, ok := b.(*types.SystemContentBlockMemberCachePoint); ok {
			return true
		}
	}

	return false
}

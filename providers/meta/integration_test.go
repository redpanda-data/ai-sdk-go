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

package meta

import (
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestMuseSpark13_Integration(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("live Meta API test")
	}

	key := os.Getenv("MODEL_API_KEY")
	if key == "" {
		t.Skip("MODEL_API_KEY is not set")
	}

	p, err := NewProvider(key, openai.WithTimeout(60*time.Second))
	require.NoError(t, err)
	m, err := p.NewModel(ModelMuseSpark13, openai.WithMaxTokens(1024), openai.WithReasoningEffort(openai.ReasoningEffortMinimal))
	require.NoError(t, err)

	req := &llm.Request{Messages: []llm.Message{{Role: llm.RoleUser, Content: []llm.Part{llm.NewTextPart("Reply with exactly: META_OK")}}}}
	result, err := m.Generate(t.Context(), req)
	require.NoError(t, err)
	assert.Contains(t, result.TextContent(), "META_OK")
}

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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func mtTestModel(t *testing.T, opts ...Option) *Model {
	t.Helper()

	p, err := NewProvider("test-key")
	require.NoError(t, err)

	model, err := p.NewModel(ModelClaudeSonnet5, opts...)
	require.NoError(t, err)

	m, ok := model.(*Model)
	require.True(t, ok, "NewModel must return *Model")

	return m
}

func mtRequest(maxTokens *int) *llm.Request {
	req := &llm.Request{
		Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))},
	}
	if maxTokens != nil {
		req.Options = &RequestOptions{MaxTokens: maxTokens}
	}

	return req
}

// TestMaxTokens_PerRequestOverride pins the contract the maintainers asked for:
// the budget policy lives in the harness, so a caller must be able to set
// max_tokens per request (without rebuilding the model) via llm.Request.Options.
func TestMaxTokens_PerRequestOverride(t *testing.T) {
	t.Parallel()

	perReq := 8000

	cases := []struct {
		name string
		opts []Option
		req  *llm.Request
		want int64
	}{
		{
			name: "no override falls back to default",
			req:  mtRequest(nil),
			want: defaultMaxTokens,
		},
		{
			name: "per-request Options overrides the fallback",
			req:  mtRequest(&perReq),
			want: 8000,
		},
		{
			name: "per-request over MaxOutputTokens is clamped",
			req:  mtRequest(new(999_999)),
			want: 128_000, // ModelClaudeSonnet5 MaxOutputTokens
		},
		{
			name: "WithMaxTokens sets the per-model value",
			opts: []Option{WithMaxTokens(12_000)},
			req:  mtRequest(nil),
			want: 12_000,
		},
		{
			name: "per-request overrides WithMaxTokens",
			opts: []Option{WithMaxTokens(12_000)},
			req:  mtRequest(&perReq),
			want: 8000,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			m := mtTestModel(t, tc.opts...)
			apiReq, err := m.requestMapper.ToProvider(tc.req)
			require.NoError(t, err)
			assert.Equal(t, tc.want, apiReq.MaxTokens)
		})
	}
}

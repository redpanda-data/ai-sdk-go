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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestStoreWireParameter(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		opts []Option
		want any
	}{
		{name: "unset stays omitted"},
		{name: "explicit false", opts: []Option{WithStore(false)}, want: false},
		{name: "explicit true", opts: []Option{WithStore(true)}, want: true},
		{name: "last option wins", opts: []Option{WithStore(true), WithStore(false)}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider, err := NewProvider("placeholder-key")
			require.NoError(t, err)

			model, err := provider.NewCompatModel("custom-model", CompatModelDefinition{}, tt.opts...)
			require.NoError(t, err)

			concrete, ok := model.(*Model)
			require.True(t, ok)

			apiRequest, err := concrete.requestMapper.ToProvider(&llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello")),
			}})
			require.NoError(t, err)

			raw, err := apiRequest.MarshalJSON()
			require.NoError(t, err)

			var body map[string]any
			require.NoError(t, json.Unmarshal(raw, &body))

			if tt.want == nil {
				assert.NotContains(t, body, "store")
			} else {
				assert.Equal(t, tt.want, body["store"])
			}
		})
	}
}

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
	"slices"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestWireAPIsForModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		modelID string
		want    []llm.WireAPI
		wantOK  bool
	}{
		{
			name:    "claude registered profile variant answers Converse and Anthropic Messages",
			modelID: ModelClaudeSonnet45US,
			want:    []llm.WireAPI{llm.WireAPIBedrockConverse, llm.WireAPIAnthropicMessages},
			wantOK:  true,
		},
		{
			name:    "bare claude ID resolves through a registered sibling variant",
			modelID: ModelClaudeSonnet45,
			want:    []llm.WireAPI{llm.WireAPIBedrockConverse, llm.WireAPIAnthropicMessages},
			wantOK:  true,
		},
		{
			name:    "unregistered profile variant resolves through a registered sibling",
			modelID: "jp." + ModelClaudeHaiku45,
			want:    []llm.WireAPI{llm.WireAPIBedrockConverse, llm.WireAPIAnthropicMessages},
			wantOK:  true,
		},
		{
			name:    "nova is Converse-only",
			modelID: ModelNova2LiteUS,
			want:    []llm.WireAPI{llm.WireAPIBedrockConverse},
			wantOK:  true,
		},
		{
			name:    "gemma is mantle (Responses) only",
			modelID: ModelGemma431B,
			want:    []llm.WireAPI{llm.WireAPIOpenAIResponses},
			wantOK:  true,
		},
		{
			name:    "unknown model is not resolved",
			modelID: "meta.llama3-70b-instruct-v1:0",
			wantOK:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got, ok := WireAPIsForModel(tt.modelID)
			if ok != tt.wantOK {
				t.Fatalf("WireAPIsForModel(%q) ok = %v, want %v", tt.modelID, ok, tt.wantOK)
			}

			if !slices.Equal(got, tt.want) {
				t.Fatalf("WireAPIsForModel(%q) = %v, want %v", tt.modelID, got, tt.want)
			}
		})
	}
}

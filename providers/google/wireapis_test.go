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

package google

import (
	"slices"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestWireAPIsForModel(t *testing.T) {
	t.Parallel()

	want := []llm.WireAPI{llm.WireAPIGeminiGenerateContent, llm.WireAPIOpenAIChatCompletions}
	for _, id := range []string{"gemini-2.5-flash", "gemini-3.5-flash"} {
		got, ok := WireAPIsForModel(id)
		if !ok || !slices.Equal(got, want) {
			t.Fatalf("WireAPIsForModel(%q) = %v, %v; want %v, true", id, got, ok, want)
		}
	}

	if _, ok := WireAPIsForModel("claude-fable-5"); ok {
		t.Fatal("foreign model must not resolve")
	}
}

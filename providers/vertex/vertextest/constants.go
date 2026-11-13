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

package vertextest

import "github.com/redpanda-data/ai-sdk-go/providers/vertex"

const (
	// TestGeminiModelName is the Gemini model for the live Vertex tests.
	TestGeminiModelName = vertex.ModelGemini38Flash
	// TestClaudeModelName is the Claude model for the live Vertex tests. It
	// must be enabled in the test project's Model Garden, so moving it means
	// asking for a new grant.
	TestClaudeModelName = vertex.ModelClaudeHaiku45
	// TestClaudeReasoningModelName is the Claude model for reasoning tests,
	// under the same grant as TestClaudeModelName.
	TestClaudeReasoningModelName = vertex.ModelClaudeSonnet5
)

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

package anthropictest

import "github.com/redpanda-data/ai-sdk-go/providers/anthropic"

const (
	// TestModelName is the model to use for integration tests.
	// Sonnet 5.5 thinks adaptively even when thinking is not requested
	// (max_tokens caps thinking plus text), and it rejects non-default
	// sampling parameters and forced tool_choice.
	TestModelName = anthropic.ModelClaudeSonnet55
	// TestReasoningModelName is the model for reasoning tests.
	// Uses a model with forced (non-adaptive) extended thinking so the
	// conformance reasoning test can reliably assert thinking traces.
	TestReasoningModelName = anthropic.ModelClaudeOpus45
	// TestAdaptiveModelName is the model for adaptive thinking tests.
	TestAdaptiveModelName = anthropic.ModelClaudeSonnet46
	// TestNoThinkingModelName is for tests that need visible text inside a
	// tiny output budget. It does not think unless thinking is requested;
	// TestModelName would spend such a budget on a thinking block whose text
	// the API omits by default.
	TestNoThinkingModelName = anthropic.ModelClaudeSonnet46
)

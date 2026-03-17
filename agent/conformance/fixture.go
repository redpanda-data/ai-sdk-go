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

package conformance

import (
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Fixture defines the interface that provider packages must implement
// to participate in agent conformance testing.
//
// Each provider package (openai, anthropic, google, etc.) should create
// a fixture that demonstrates their provider works correctly with the
// agent layer, particularly for tool calling scenarios.
type Fixture interface {
	// Name returns the provider name (e.g., "OpenAI", "Anthropic", "Gemini")
	Name() string

	// StandardAgent creates an agent with a standard model and the given tool registry.
	// Returns nil if provider doesn't support agents or tools.
	StandardAgent(tools tool.Registry) (*llmagent.LLMAgent, error)

	// ReasoningAgent creates an agent with a reasoning model and the given tool registry.
	// Returns nil if provider doesn't support reasoning models.
	// This is optional - providers without reasoning models should return nil.
	ReasoningAgent(tools tool.Registry) (*llmagent.LLMAgent, error)
}

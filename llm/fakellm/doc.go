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

// Package fakellm provides a high-quality, flexible mock LLM implementation
// for integration testing without hitting real AI provider APIs.
//
// # Overview
//
// This package offers a realistic fake LLM provider that implements the llm.Model
// interface with configurable behavior, error simulation, and stateful scenarios.
// It's designed for integration testing of agents and workflows that depend on
// LLM interactions.
//
// # Design Philosophy
//
// Unlike auto-generated gomock mocks that require verbose expectations, this
// package provides a rule-based DSL that makes tests readable and maintainable.
// The mock behaves like a real LLM provider with:
//   - Configurable responses (text, tool calls, structured output)
//   - Realistic streaming with chunking and delays
//   - Token usage tracking
//   - Latency simulation
//   - Comprehensive error scenarios
//
// Rules are evaluated in registration order (first-match wins), allowing you
// to define specific rules first and general fallbacks later.
//
// # Basic Usage
//
// Create a fake model with simple text responses:
//
//	model := fakellm.NewFakeModel().
//	    When(fakellm.UserMessageContains("hello")).
//	    ThenRespondText("Hello! How can I help you?")
//
//	response, err := model.Generate(ctx, &llm.Request{
//	    Messages: []llm.Message{llm.UserText("hello")},
//	})
//
// # Streaming
//
// Simulate realistic streaming with configurable chunking:
//
//	model := fakellm.NewFakeModel().
//	    When(fakellm.Any()).
//	    ThenStreamText("This is a long response...", fakellm.StreamConfig{
//	        ChunkSize:       8,  // Characters per chunk
//	        InterChunkDelay: 50 * time.Millisecond,
//	    })
//
// # Error Simulation
//
// Test error handling and retry logic:
//
//	model := fakellm.NewFakeModel().
//	    RateLimitOnce(). // First call fails with rate limit
//	    When(fakellm.Any()).
//	    ThenRespondText("Success!") // Subsequent calls succeed
//
// # Tool Calling
//
// Simulate multi-turn tool calling scenarios:
//
//	model := fakellm.NewFakeModel().
//	    When(fakellm.HasTool("get_weather")).
//	    ThenRespondWithToolCall("get_weather", map[string]any{
//	        "location": "San Francisco",
//	    })
//
// # Stateful Scenarios
//
// Test complex multi-turn conversations:
//
//	model := fakellm.NewFakeModel().Scenario("weather-lookup", func(s *fakellm.ScenarioBuilder) {
//	    // Turn 0: Request tool call
//	    s.OnTurn(0).
//	        When(fakellm.HasTool("get_weather")).
//	        ThenRespondWithToolCall("get_weather", args)
//
//	    // Turn 1: After tool response, provide final answer
//	    s.OnTurn(1).
//	        When(fakellm.LastMessageHasToolResponse("get_weather")).
//	        ThenRespondText("The weather in SF is 68°F and sunny.")
//	})
//
// # Deterministic Behavior
//
// The mock provides deterministic fallback behavior when no rules match,
// ensuring tests don't break unexpectedly. By default, it echoes the last
// user message with proper token counting and finish reasons.
//
// # Token Usage
//
// Token counts are automatically calculated using a configurable tokenizer.
// The default uses a simple heuristic (4 chars ≈ 1 token), or you can provide
// a real tokenizer for accurate testing:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithTokenizer(myTikTokenizer),
//	)
//
// # Latency Simulation
//
// Add realistic timing to test timeouts and cancellation:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithLatency(fakellm.LatencyProfile{
//	        Base:     100 * time.Millisecond,  // Fixed overhead
//	        PerToken: 5 * time.Millisecond,     // Per output token
//	        PerChunk: 20 * time.Millisecond,    // Streaming delay
//	    }),
//	)
//
// # Fast Deterministic Time Testing
//
// Use Go 1.25's testing/synctest package for instant, deterministic time testing:
//
//	func TestLatency(t *testing.T) {
//	    synctest.Test(t, func(t *testing.T) {
//	        model := fakellm.NewFakeModel(
//	            fakellm.WithLatency(fakellm.LatencyProfile{Base: 100*time.Millisecond}),
//	        )
//	        start := time.Now()
//	        resp, _ := model.Generate(ctx, req)
//	        // Test runs instantly, time.Since() reports 100ms
//	        assert.Equal(t, 100*time.Millisecond, time.Since(start))
//	    })
//	}
package fakellm

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

package fakellm

import (
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Option configures a FakeModel.
type Option func(*FakeModel)

// WithModelName sets the model name returned by Name().
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithModelName("gpt-4-turbo-fake"),
//	)
func WithModelName(name string) Option {
	return func(m *FakeModel) {
		m.name = name
	}
}

// WithCapabilities sets the model's capabilities.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithCapabilities(llm.ModelCapabilities{
//	        Streaming: true,
//	        Tools:     false,  // This model doesn't support tools
//	    }),
//	)
func WithCapabilities(caps llm.ModelCapabilities) Option {
	return func(m *FakeModel) {
		m.caps = caps
	}
}

// WithTokenizer sets a custom tokenizer for accurate token counting.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithTokenizer(myTikTokenizer),
//	)
func WithTokenizer(tokenizer Tokenizer) Option {
	return func(m *FakeModel) {
		m.tokenizer = tokenizer
	}
}

// WithLatency sets the latency profile for simulating realistic timing.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithLatency(testing.LatencyProfile{
//	        Base:     100 * time.Millisecond,
//	        PerToken: 5 * time.Millisecond,
//	        PerChunk: 20 * time.Millisecond,
//	    }),
//	)
func WithLatency(profile LatencyProfile) Option {
	return func(m *FakeModel) {
		m.latency = profile
	}
}

// WithFallbackFinishReason sets the finish reason used when no rules match.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithFallbackFinishReason(llm.FinishReasonLength),
//	)
func WithFallbackFinishReason(reason llm.FinishReason) Option {
	return func(m *FakeModel) {
		m.defaults.FallbackFinishReason = reason
	}
}

// WithChunkSize sets the default streaming chunk size in runes.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithChunkSize(8),  // Small chunks for testing
//	)
func WithChunkSize(size int) Option {
	return func(m *FakeModel) {
		m.defaults.ChunkSize = size
	}
}

// WithSessionKeyFrom sets a custom function for deriving conversation keys.
// This allows centralizing session tracking logic instead of relying on
// metadata or the default hash-based approach.
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithSessionKeyFrom(func(req *llm.Request) string {
//	        // Extract from custom header or context
//	        return req.Metadata["x-correlation-id"]
//	    }),
//	)
func WithSessionKeyFrom(fn func(*llm.Request) string) Option {
	return func(m *FakeModel) {
		m.sessionKeyFrom = fn
	}
}

// LatencyProfile configures timing delays to simulate real LLM behavior.
type LatencyProfile struct {
	// Base is the fixed overhead per request (e.g., network RTT)
	Base time.Duration

	// PerToken is additional delay per output token generated
	PerToken time.Duration

	// PerChunk is the delay between streaming chunks
	PerChunk time.Duration
}

// Tokenizer counts tokens in text.
// Implement this interface to provide accurate token counting for your model.
type Tokenizer interface {
	Count(text string) int
}

// defaultTokenizer provides a simple approximation: 4 characters ≈ 1 token.
type defaultTokenizer struct{}

func (defaultTokenizer) Count(text string) int {
	runes := []rune(text)
	if len(runes) == 0 {
		return 0
	}
	// Rough approximation: 4 chars per token
	return (len(runes) + 3) / 4
}

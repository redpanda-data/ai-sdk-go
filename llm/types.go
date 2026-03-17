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

package llm

// TokenUsage tracks token consumption for AI model requests.
//
// Fields are populated on a best-effort basis — not all providers report every field.
// Zero values indicate the provider did not report that metric.
//
// # Provider Coverage
//
// All providers report InputTokens, OutputTokens, and TotalTokens.
// CachedTokens is widely supported. ReasoningTokens is provider-specific.
// See individual field docs for details.
type TokenUsage struct {
	// InputTokens is the number of tokens in the input/prompt.
	// Reported by all providers.
	InputTokens int `json:"input_tokens"`

	// OutputTokens is the number of tokens in the generated response.
	// Reported by all providers. For reasoning models on OpenAI, this includes
	// reasoning tokens — ReasoningTokens is a subset of OutputTokens, not additive.
	OutputTokens int `json:"output_tokens"`

	// TotalTokens is the total token count for the request.
	// Most providers return this directly from their API. Anthropic computes it
	// as InputTokens + OutputTokens since their API does not provide it.
	//
	// For non-reasoning models, TotalTokens == InputTokens + OutputTokens.
	// For reasoning models (OpenAI), TotalTokens still equals InputTokens + OutputTokens
	// because reasoning tokens are included in OutputTokens.
	TotalTokens int `json:"total_tokens"`

	// CachedTokens is the number of input tokens served from the provider's prompt cache.
	// These tokens are typically billed at a reduced rate. CachedTokens is a subset of
	// InputTokens, not additive.
	//
	// Supported by all providers: OpenAI (InputTokensDetails.CachedTokens),
	// Anthropic (CacheReadInputTokens), Google (CachedContentTokenCount),
	// Bedrock (CacheReadInputTokens), and OpenAI-compatible APIs.
	CachedTokens int `json:"cached_tokens,omitempty"`

	// ReasoningTokens is the number of tokens the model used for internal reasoning (thinking).
	// This is a subset of OutputTokens — it is NOT additive to the output count.
	// These tokens are billed as output tokens.
	//
	// Only reported by OpenAI (o-series, GPT-5) and OpenAI-compatible providers
	// (e.g., DeepSeek-R1). Anthropic, Google, and Bedrock do not report this field;
	// it will be zero for those providers.
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// MaxInputTokens is the model's context window size (maximum input tokens accepted).
	// This is a model capability, not a usage metric — it is populated from the model
	// definition at configuration time, not from API responses.
	MaxInputTokens int `json:"max_input_tokens,omitempty"`
}

// SumUsage aggregates multiple TokenUsage values into a single cumulative result.
// This is the preferred way to accumulate usage across multiple operations.
// Nil values are safely skipped. Returns nil if all inputs are nil.
//
// Example:
//
//	total := llm.SumUsage(turn1Usage, turn2Usage, turn3Usage)
func SumUsage(usages ...*TokenUsage) *TokenUsage {
	var result *TokenUsage

	for _, u := range usages {
		if u == nil {
			continue
		}

		if result == nil {
			// First non-nil usage, make a copy
			val := *u
			result = &val
		} else {
			// Accumulate into result
			maxInputTokens := max(u.MaxInputTokens, result.MaxInputTokens)

			result = &TokenUsage{
				InputTokens:     u.InputTokens + result.InputTokens,
				OutputTokens:    u.OutputTokens + result.OutputTokens,
				TotalTokens:     u.TotalTokens + result.TotalTokens,
				CachedTokens:    u.CachedTokens + result.CachedTokens,
				ReasoningTokens: u.ReasoningTokens + result.ReasoningTokens,
				MaxInputTokens:  maxInputTokens,
			}
		}
	}

	return result
}

// FinishReason indicates why model generation stopped.
type FinishReason string

// FinishReason constants provide standardized values across providers.
const (
	// FinishReasonStop indicates the model completed naturally.
	FinishReasonStop FinishReason = "stop"

	// FinishReasonLength indicates the response was truncated due to length limits.
	FinishReasonLength FinishReason = "length"

	// FinishReasonToolCalls indicates the model wants to execute tools.
	FinishReasonToolCalls FinishReason = "tool_calls"

	// FinishReasonContentFilter indicates content was blocked by safety filters.
	FinishReasonContentFilter FinishReason = "content_filter"

	// FinishReasonInterrupted indicates the request was cancelled or interrupted.
	FinishReasonInterrupted FinishReason = "interrupted"

	// FinishReasonUnknown is used when the provider returns an unrecognized reason.
	FinishReasonUnknown FinishReason = "unknown"
)

// ModelCapabilities describes what features a model supports.
// This enables compile-time and runtime validation of requests.
type ModelCapabilities struct {
	Streaming        bool // Supports streaming responses
	Tools            bool // Supports function/tool calling
	JSONMode         bool // Supports JSON mode (response_format: json_object) - ensures valid JSON output
	StructuredOutput bool // Supports Structured Outputs (response_format: json_schema) - ensures schema adherence
	Vision           bool // Supports image inputs
	Audio            bool // Supports audio inputs
	MultiTurn        bool // Supports conversation history
	SystemPrompts    bool // Supports system role messages
	Reasoning        bool // Supports reasoning controls and exposes reasoning traces
}

// ModelDiscoveryInfo provides metadata about a model that can be discovered at runtime.
// This is returned by provider.Models() for model discovery and capability checking.
type ModelDiscoveryInfo struct {
	// Name is the model identifier used in API calls
	Name string

	// Label is a human-readable display name
	Label string

	// Capabilities describes what features this model supports
	Capabilities ModelCapabilities

	// Provider is the name of the provider that offers this model
	Provider string
}

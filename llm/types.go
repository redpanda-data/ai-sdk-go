package llm

// TokenUsage tracks token consumption for AI model requests.
// Token counting varies by provider, so not all fields may be available.
type TokenUsage struct {
	// InputTokens is the number of tokens in the input/prompt
	InputTokens int `json:"input_tokens"`

	// OutputTokens is the number of tokens in the generated response
	OutputTokens int `json:"output_tokens"`

	// TotalTokens is the sum of prompt and completion tokens.
	// This should equal InputTokens + OutputTokens when both are available.
	TotalTokens int `json:"total_tokens"`

	// CachedTokens represents tokens that were served from cache, if supported.
	// This can help track cost savings from caching.
	CachedTokens int `json:"cached_tokens,omitempty"`

	// ReasoningTokens is the number of tokens used for reasoning (thinking) by reasoning models.
	// Only available for models that support reasoning (OpenAI o-series, GPT-5 series).
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`

	// MaxInputTokens is the maximum number of tokens the model can accept as input.
	// This represents the model's context window size.
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
			maxInputTokens := result.MaxInputTokens
			if u.MaxInputTokens > maxInputTokens {
				maxInputTokens = u.MaxInputTokens
			}
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

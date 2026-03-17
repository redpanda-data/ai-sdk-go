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

import (
	"context"
	"iter"
)

// ModelInfo provides metadata about a model.
//
// Use ModelInfo when you only need to query model properties without
// performing any generation. This is useful for model discovery, capability
// checking, and routing decisions.
//
// Example:
//
//	model := provider.NewModel("gpt-4o")
//	if model.Capabilities().Vision {
//		// This model supports image inputs
//	}
type ModelInfo interface {
	// Name returns the model identifier (e.g., "gpt-4o", "claude-3.5-sonnet").
	// The returned name should be consistent and can be used for logging,
	// metrics, and model selection logic.
	Name() string

	// Provider returns the name of the AI provider (e.g., "openai", "anthropic", "google").
	// This is useful for observability, routing decisions, and provider-specific handling.
	Provider() string

	// Capabilities returns what features this model supports.
	// Use this to check if specific features are available before making requests
	// that depend on them (e.g., streaming, tool calls, vision).
	Capabilities() ModelCapabilities

	// Constraints returns the model's validation rules and limitations.
	// This includes the maximum context window size, temperature range,
	// supported parameters, and other model-specific constraints.
	Constraints() ModelConstraints
}

// Generator provides non-streaming text generation capabilities.
//
// Use Generator when you need the complete response at once, such as for
// simple question-answering, classification tasks, or when building batch
// processing systems.
//
// Generator provides LLM generation capabilities for creating responses.
type Generator interface {
	// Generate performs a single, non-streaming request and returns the complete response.
	//
	// The context can be used for cancellation and timeouts. If the context is cancelled,
	// the request will be aborted and context.Canceled will be returned.
	//
	// Error Handling:
	//
	// Provider errors return *ProviderError with categorized sentinel errors:
	//   - ErrRateLimitExceeded: Retryable with exponential backoff
	//   - ErrInvalidInput: Not retryable, caller must fix input
	//   - ErrContentPolicyViolation: Not retryable, policy violation
	//   - ErrServerError: Retryable, transient provider issue
	//
	// SDK errors return standard errors:
	//   - ErrRequestMapping: Request conversion failed
	//   - ErrResponseMapping: Response conversion failed
	//   - ErrInvalidConfig: Provider configuration is invalid
	//
	// Example with error handling:
	//
	//	response, err := model.Generate(ctx, &llm.Request{
	//		Messages: []llm.Message{{
	//			Role:    llm.RoleUser,
	//			Content: []*llm.Part{llm.NewTextPart("Hello!")},
	//		}},
	//	})
	//	if err != nil {
	//		// Check error category with errors.Is()
	//		if errors.Is(err, llm.ErrRateLimitExceeded) {
	//			// Retry with exponential backoff
	//			return retryWithBackoff(ctx, model, request)
	//		} else if errors.Is(err, llm.ErrInvalidInput) {
	//			// Don't retry - fix the input
	//			return fmt.Errorf("invalid request: %w", err)
	//		}
	//
	//		// Get provider-specific details with type assertion
	//		if perr, ok := err.(*llm.ProviderError); ok {
	//			log.Printf("Provider error [%s]: %s", perr.Code, perr.Message)
	//		}
	//		return err
	//	}
	//
	//	// Check if response is complete
	//	if response.FinishReason != llm.FinishReasonStop {
	//		log.Printf("Response incomplete: %s", response.FinishReason)
	//	}
	Generate(ctx context.Context, req *Request) (*Response, error)
}

// EventsGenerator provides streaming text generation capabilities.
//
// Use EventsGenerator when you need real-time response streaming, such as for
// interactive chat applications, live content generation, or when you want to
// display partial results to users as they become available.
type EventsGenerator interface {
	// GenerateEvents returns an iterator for streaming LLM responses.
	// Iteration ends naturally after a final StreamEndEvent, or early if the consumer
	// breaks out of the loop. Resource cleanup is automatic in both cases.
	//
	// The iterator yields (event, nil) for successful events or (nil, err) for fatal
	// transport/mapping failures. Always check err on each iteration.
	//
	// Error Handling:
	//
	// Three types of errors are communicated through the iterator:
	//
	// 1. Recoverable warnings (ErrorEvent):
	//   - Yielded as (ErrorEvent{...}, nil)
	//   - Non-terminal, streaming continues
	//   - Log or handle as appropriate for your application
	//
	// 2. Terminal provider errors (StreamEndEvent.Error):
	//   - Yielded as (StreamEndEvent{Error: err}, nil)
	//   - Final event, indicates completion with error
	//   - Check with errors.Is():
	//     * ErrRateLimitExceeded: Retryable with backoff
	//     * ErrInvalidInput: Not retryable, fix input
	//     * ErrContentPolicyViolation: Not retryable, policy violation
	//     * ErrServerError: Retryable, transient issue
	//
	// 3. Fatal transport/mapping errors:
	//   - Yielded as (nil, err)
	//   - Stops iteration immediately
	//   - Check with errors.Is():
	//     * ErrRequestMapping: Request conversion failed
	//     * ErrResponseMapping: Response parsing failed
	//     * ErrAPICall: Network or transport failure
	//     * context.Canceled: Context was cancelled
	//
	// Example usage:
	//
	//	for event, err := range model.GenerateEvents(ctx, request) {
	//		if err != nil {
	//			return fmt.Errorf("stream failed: %w", err)
	//		}
	//
	//		switch e := event.(type) {
	//		case llm.ContentPartEvent:
	//			// Handle streaming content (text, tool calls, reasoning)
	//			fmt.Print(e.Part.Text())
	//
	//		case llm.ErrorEvent:
	//			// Handle recoverable warnings
	//			log.Printf("warning [%s]: %s", e.Code, e.Message)
	//
	//		case llm.StreamEndEvent:
	//			// Handle completion
	//			if e.Error != nil {
	//				// Provider returned an error
	//				if errors.Is(e.Error, llm.ErrRateLimitExceeded) {
	//					return retryWithBackoff(ctx, model, request)
	//				}
	//				return e.Error
	//			}
	//			// Success - e.Response contains usage stats and finish reason
	//			log.Printf("usage: %+v", e.Response.Usage)
	//		}
	//	}
	//	// Cleanup happens automatically, even on early break/return!
	//
	// For advanced use cases requiring fine-grained control, use iter.Pull2:
	//
	//	next, stop := iter.Pull2(model.GenerateEvents(ctx, request))
	//	defer stop()
	//
	//	for {
	//		event, err, ok := next()
	//		if !ok {
	//			break
	//		}
	//		if err != nil {
	//			return err
	//		}
	//		// Handle event with fine-grained control
	//	}
	GenerateEvents(ctx context.Context, req *Request) iter.Seq2[Event, error]
}

// Model represents the complete interface for interacting with AI models.
// It combines metadata, generation, and streaming capabilities into a single interface.
//
// All provider implementations must support both streaming and non-streaming generation.
// If a provider doesn't support streaming, GenerateStream should return ErrUnsupportedFeature.
//
// Design Note: Consumer code should prefer the narrowest interface needed:
//   - Use ModelInfo when you only need metadata
//   - Use Generator when you only need non-streaming generation
//   - Use StreamGenerator when you only need streaming
//   - Use Model when you need the complete feature set
//
// Example usage patterns:
//
//	// Prefer narrow interfaces in function signatures:
//	func processText(gen llm.Generator) { ... }
//	func streamChat(sg llm.StreamGenerator) { ... }
//
//	// Use type assertions for optional streaming:
//	if sg, ok := model.(llm.StreamGenerator); ok {
//		return streamResponse(sg)
//	}
//	return batchResponse(model)
type Model interface {
	ModelInfo
	Generator
	EventsGenerator
}

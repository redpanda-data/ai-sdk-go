package llm

import "context"

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

	// Capabilities returns what features this model supports.
	// Use this to check if specific features are available before making requests
	// that depend on them (e.g., streaming, tool calls, vision).
	Capabilities() ModelCapabilities
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

// StreamGenerator provides streaming text generation capabilities.
//
// Use StreamGenerator when you need real-time response streaming, such as for
// interactive chat applications, live content generation, or when you want to
// display partial results to users as they become available.
type StreamGenerator interface {
	// GenerateStream performs a streaming request and returns an EventStream for real-time results.
	//
	// The returned EventStream follows standard Go streaming patterns (similar to sql.Rows):
	//   - Call Recv() repeatedly until io.EOF or an error
	//   - Always call Close() to release resources (use defer)
	//   - Context cancellation will close the stream and return context.Canceled
	//
	//
	// Error Handling:
	//
	// GenerateStream initialization errors:
	//   - ErrRequestMapping: Request conversion failed
	//   - ErrUnsupportedFeature: Provider/model doesn't support streaming
	//   - ErrInvalidConfig: Provider configuration is invalid
	//
	// StreamEndEvent.Error contains provider errors (same as Generate):
	//   - ErrRateLimitExceeded: Retryable with backoff
	//   - ErrInvalidInput: Not retryable, fix input
	//   - ErrContentPolicyViolation: Not retryable, policy violation
	//   - ErrServerError: Retryable, transient issue
	//   - ErrResponseMapping: Response parsing failed
	//
	// ErrorEvent during streaming represents recoverable warnings (non-terminal).
	// See StreamEndEvent documentation for error handling examples.
	//
	//	stream, err := model.GenerateStream(ctx, request)
	//	if err != nil {
	//		return err
	//	}
	//	defer stream.Close()
	//
	//	for {
	//		event, err := stream.Recv()
	//		if err == io.EOF {
	//			break // Stream completed successfully
	//		}
	//		if err != nil {
	//			return err // Stream error
	//		}
	//		// Handle event (see StreamEndEvent docs for error handling)
	//	}
	GenerateStream(ctx context.Context, req *Request) (EventStream, error)
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
	StreamGenerator
}

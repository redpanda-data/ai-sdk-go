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

// Event represents all possible events that can occur during streaming.
// This is a discriminated union implemented using Go interfaces.
// The isStreamEvent() method is a type constraint that prevents external types
// from accidentally implementing this interface, ensuring type safety.
type Event interface {
	// isStreamEvent is an unexported method that acts as a type constraint.
	// Only types defined in this package can implement StreamEvent,
	// which prevents external code from accidentally satisfying the interface
	// and ensures all stream events are known and handled by the SDK.
	isStreamEvent()
}

// ContentPartEvent represents content during streaming generation.
// Based on provider streaming patterns:
// - Text content: streamed as incremental tokens for immediate display
// - Tool calls: buffered until complete and valid, then emitted as complete Parts
// - Reasoning traces: streamed as incremental reasoning steps for transparency.
type ContentPartEvent struct {
	// Index refers to which content position this part occupies in the response
	Index int `json:"index"`

	// Part contains the content using the Part system
	// - Text Parts: can be incremental deltas (tokens/words) for responsive display
	// - Tool Call Parts: always complete and valid JSON, ready for execution
	// - Other content types: depends on provider capabilities
	Part *Part `json:"part"`
}

// isStreamEvent implements the StreamEvent interface constraint.
func (ContentPartEvent) isStreamEvent() {}

// ErrorEvent represents an error that occurred during streaming.
// This can be either a recoverable error (stream continues) or
// a terminal error (stream ends).
type ErrorEvent struct {
	// Message provides a human-readable error description
	Message string `json:"message"`

	// Code provides a machine-readable error code for programmatic handling.
	// This is optional and may be provider-specific.
	Code string `json:"code,omitempty"`
}

// isStreamEvent implements the StreamEvent interface constraint.
func (ErrorEvent) isStreamEvent() {}

// StreamEndEvent signals completion of a stream (success or failure).
// This event is always the final event in the iterator sequence.
//
// Exactly one of Response or Error will be set:
//   - Response != nil: Generation succeeded (check Response.FinishReason for completeness)
//   - Error != nil: Generation failed
//
// Error Handling Examples:
//
//	case llm.StreamEndEvent:
//	    if evt.Error != nil {
//	        // Check error category with errors.Is()
//	        if errors.Is(evt.Error, llm.ErrRateLimitExceeded) {
//	            // Retry with exponential backoff
//	        } else if errors.Is(evt.Error, llm.ErrInvalidInput) {
//	            // Don't retry - fix the input
//	        }
//
//	        // Get provider-specific details with type assertion
//	        if perr, ok := evt.Error.(*llm.ProviderError); ok {
//	            log.Printf("Provider error [%s]: %s", perr.Code, perr.Message)
//	        }
//	        return evt.Error
//	    }
//
//	    // Success - check if response is complete
//	    switch evt.Response.FinishReason {
//	    case llm.FinishReasonStop:
//	        // Complete response
//	    case llm.FinishReasonLength:
//	        // Partial response due to token limits
//	    case llm.FinishReasonContentFilter:
//	        // Partial response due to content filtering
//	    }
type StreamEndEvent struct {
	// Response is the complete response from the LLM provider that contains all
	// chunked information from the streamed events.
	// This is nil when Error is set.
	Response *Response `json:"response,omitempty"`

	// Error contains provider or mapping errors that prevented successful completion.
	// This is nil when Response is set.
	//
	// Common error categories (check with errors.Is):
	//   - ErrRateLimitExceeded: Retryable with backoff
	//   - ErrInvalidInput: Not retryable, fix input
	//   - ErrContentPolicyViolation: Not retryable, policy violation
	//   - ErrServerError: Retryable, transient provider issue
	//
	// Type assert to *ProviderError to access provider-specific Code and Message.
	Error error `json:"-"`
}

// isStreamEvent implements the StreamEvent interface constraint.
func (StreamEndEvent) isStreamEvent() {}

// StreamResetEvent signals that a stream is being retried. Consumers should
// discard any accumulated content from the previous attempt and prepare
// to receive events from a fresh generation.
//
// This event is emitted by the retry interceptor when a retryable error
// occurs during streaming. The sequence is:
//
//	[deltas...] → StreamResetEvent → [deltas...] → StreamEndEvent
type StreamResetEvent struct {
	// Attempt is the retry attempt number (1-based).
	// Attempt 1 means the first retry after the initial attempt failed.
	Attempt int `json:"attempt"`

	// Reason describes why the stream is being reset (typically the error message).
	Reason string `json:"reason"`
}

// isStreamEvent implements the StreamEvent interface constraint.
func (StreamResetEvent) isStreamEvent() {}

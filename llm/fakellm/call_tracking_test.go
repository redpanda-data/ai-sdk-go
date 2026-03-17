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

package fakellm_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	llmtesting "github.com/redpanda-data/ai-sdk-go/llm/fakellm"
)

// TestCallLogging verifies that all calls are properly logged.
func TestCallLogging(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.UserMessageContains("hello")).
		Named("greeting-rule").
		ThenRespondText("Hi there!")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello world")}},
		},
	}

	// Make first call
	resp1, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Make second call
	resp2, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Verify calls were logged
	calls := model.Calls()
	assert.Len(t, calls, 2)

	// Check first call
	assert.Equal(t, llmtesting.CallGenerate, calls[0].Kind)
	assert.Equal(t, "greeting-rule", calls[0].RuleName)
	require.NoError(t, calls[0].Err)
	assert.Equal(t, resp1, calls[0].Response)

	// Check second call
	assert.Equal(t, llmtesting.CallGenerate, calls[1].Kind)
	assert.Equal(t, "greeting-rule", calls[1].RuleName)
	require.NoError(t, calls[1].Err)
	assert.Equal(t, resp2, calls[1].Response)

	// Verify CallCount
	assert.Equal(t, 2, model.CallCount())
}

// TestCallLogging_Stream verifies streaming calls are logged.
func TestCallLogging_Stream(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		Named("stream-rule").
		ThenStreamText("response", llmtesting.StreamConfig{})

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	// Consume the iterator to trigger the call
	for _, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)
	}

	calls := model.Calls()
	assert.Len(t, calls, 1)
	assert.Equal(t, llmtesting.CallGenerateEvents, calls[0].Kind)
	assert.Equal(t, "stream-rule", calls[0].RuleName)
	assert.NoError(t, calls[0].Err)
}

// TestCallLogging_Errors verifies errors are logged.
func TestCallLogging_Errors(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		RateLimitOnce()

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	_, err := model.Generate(context.Background(), req)
	require.ErrorIs(t, err, llm.ErrAPICall)

	calls := model.Calls()
	assert.Len(t, calls, 1)
	require.ErrorIs(t, calls[0].Err, llm.ErrAPICall)
	assert.Nil(t, calls[0].Response)
}

// TestResetCalls verifies call log can be cleared.
func TestResetCalls(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	// Make some calls
	_, _ = model.Generate(context.Background(), req)
	_, _ = model.Generate(context.Background(), req)
	assert.Equal(t, 2, model.CallCount())

	// Reset
	model.ResetCalls()
	assert.Equal(t, 0, model.CallCount())
	assert.Empty(t, model.Calls())
}

// TestAssertCalled verifies assertion helpers work.
func TestAssertCalled(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("weather check")}},
		},
	}

	_, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Should not panic - matcher matches
	model.AssertCalled(t, llmtesting.UserMessageContains("weather"))
}

// TestAssertNotCalled verifies negative assertions.
func TestAssertNotCalled(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}},
		},
	}

	_, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Should not panic - matcher doesn't match
	model.AssertNotCalled(t, llmtesting.UserMessageContains("weather"))
}

// TestAssertCallCount verifies call count assertions.
func TestAssertCallCount(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	_, _ = model.Generate(context.Background(), req)
	_, _ = model.Generate(context.Background(), req)

	// Should not panic
	model.AssertCallCount(t, 2)
}

// TestAssertGenerateCalled verifies Generate-specific assertions.
func TestAssertGenerateCalled(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	_, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Should not panic
	model.AssertGenerateCalled(t)
}

// TestAssertStreamCalled verifies stream-specific assertions.
func TestAssertStreamCalled(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenStreamText("response", llmtesting.StreamConfig{})

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	// Consume the iterator to trigger the call
	for _, err := range model.GenerateEvents(context.Background(), req) {
		require.NoError(t, err)
	}

	// Should not panic
	model.AssertStreamCalled(t)
}

// TestCallsMatching filters calls by matcher.
func TestCallsMatching(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	// Make calls with different messages
	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("what's the weather?")}},
		},
	})

	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello there")}},
		},
	})

	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("weather forecast")}},
		},
	})

	// Filter for weather-related calls
	weatherCalls := model.CallsMatching(llmtesting.UserMessageContains("weather"))
	assert.Len(t, weatherCalls, 2)
}

// TestLastCall and FirstCall helpers.
func TestLastAndFirstCall(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.Any()).
		ThenRespondText("response")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
		},
	}

	resp1, _ := model.Generate(context.Background(), req)
	resp2, _ := model.Generate(context.Background(), req)

	first, err := model.FirstCall()
	require.NoError(t, err)
	assert.Equal(t, resp1, first.Response)

	last, err := model.LastCall()
	require.NoError(t, err)
	assert.Equal(t, resp2, last.Response)
}

// TestWithTestingT_AutoVerify verifies automatic call count verification.
func TestWithTestingT_AutoVerify(t *testing.T) {
	t.Parallel()

	// This test verifies the cleanup function runs
	t.Run("expect_exact_calls", func(t *testing.T) {
		t.Parallel()

		model := llmtesting.NewFakeModel(
			llmtesting.WithTestingT(t, llmtesting.ExpectCallCount(2)),
		).When(llmtesting.Any()).ThenRespondText("response")

		req := &llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
			},
		}

		// Make exactly 2 calls as expected
		_, _ = model.Generate(context.Background(), req)
		_, _ = model.Generate(context.Background(), req)

		// Cleanup will verify call count automatically
	})

	t.Run("expect_at_least", func(t *testing.T) {
		t.Parallel()

		model := llmtesting.NewFakeModel(
			llmtesting.WithTestingT(t, llmtesting.ExpectAtLeastCalls(1)),
		).When(llmtesting.Any()).ThenRespondText("response")

		req := &llm.Request{
			Messages: []llm.Message{
				{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("test")}},
			},
		}

		// Make at least 1 call
		_, _ = model.Generate(context.Background(), req)
		_, _ = model.Generate(context.Background(), req)

		// Cleanup will verify we made at least 1 call
	})
}

// TestDebugCalls verifies debug output.
func TestDebugCalls(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.UserMessageContains("weather")).
		Named("weather-rule").
		ThenRespondText("It's sunny!")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("what's the weather?")}},
		},
	}

	_, err := model.Generate(context.Background(), req)
	require.NoError(t, err)

	// Get debug output
	debug := model.DebugCalls()
	assert.Contains(t, debug, "Total calls: 1")
	assert.Contains(t, debug, "Call #1:")
	assert.Contains(t, debug, "Kind: Generate")
	assert.Contains(t, debug, "Rule: weather-rule")
	assert.Contains(t, debug, "Last user msg:")
	assert.Contains(t, debug, "Response:")

	t.Log("\n" + debug) // Show in test output
}

// TestMatcherErrorMessages verifies descriptive error messages.
func TestMatcherErrorMessages(t *testing.T) {
	t.Parallel()

	// Create matcher that won't match
	matcher := llmtesting.UserMessageContains("weather")

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}},
		},
	}

	cc := &llmtesting.CallContext{}
	err := matcher(req, cc)

	// Verify descriptive error
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no user message contains")
	assert.Contains(t, err.Error(), "weather")
}

// TestComplexMatcherErrors verifies error composition.
func TestComplexMatcherErrors(t *testing.T) {
	t.Parallel()

	// And matcher - should show which matcher failed
	andMatcher := llmtesting.And(
		llmtesting.UserMessageContains("weather"),
		llmtesting.HasTool("get_weather"),
	)

	req := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("weather check")}},
		},
		// No tools
	}

	cc := &llmtesting.CallContext{}
	err := andMatcher(req, cc)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "matcher 1 failed") // Second matcher failed
	assert.Contains(t, err.Error(), "no tool named")

	// Or matcher - should show all failures
	orMatcher := llmtesting.Or(
		llmtesting.UserMessageContains("weather"),
		llmtesting.UserMessageContains("time"),
	)

	req2 := &llm.Request{
		Messages: []llm.Message{
			{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}},
		},
	}

	err2 := orMatcher(req2, cc)
	require.Error(t, err2)
	assert.Contains(t, err2.Error(), "no matcher succeeded")
}

// TestNamedRulesInCalls verifies rule names appear in call logs.
func TestNamedRulesInCalls(t *testing.T) {
	t.Parallel()

	model := llmtesting.NewFakeModel().
		When(llmtesting.UserMessageContains("weather")).
		Named("weather-query").
		ThenRespondText("Sunny").
		When(llmtesting.UserMessageContains("time")).
		Named("time-query").
		ThenRespondText("3 PM").
		When(llmtesting.Any()).
		Named("fallback").
		ThenRespondText("Unknown")

	// Make different calls
	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("weather?")}}},
	})

	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("time?")}}},
	})

	_, _ = model.Generate(context.Background(), &llm.Request{
		Messages: []llm.Message{{Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("hello")}}},
	})

	calls := model.Calls()
	assert.Equal(t, "weather-query", calls[0].RuleName)
	assert.Equal(t, "time-query", calls[1].RuleName)
	assert.Equal(t, "fallback", calls[2].RuleName)
}

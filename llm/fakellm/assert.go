package fakellm

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// WithTestingT integrates the fake model with Go's testing framework.
// It enables automatic verification at test cleanup and better error messages.
//
// Example:
//
//	func TestAgent(t *testing.T) {
//	    model := fakellm.NewFakeModel(
//	        fakellm.WithTestingT(t,
//	            fakellm.ExpectCallCount(2),
//	        ),
//	    )
//	    // Test code...
//	    // At cleanup, verifications run automatically
//	}
func WithTestingT(tb testing.TB, verifyOpts ...VerifyOption) Option {
	tb.Helper()

	return func(m *FakeModel) {
		m.t = tb

		// Build verification config with defaults (-1 = don't check)
		cfg := &verifyConfig{
			expectCallCount: -1,
			expectMinCalls:  -1,
		}
		for _, opt := range verifyOpts {
			opt(cfg)
		}

		// Register cleanup if any verifications are configured
		if cfg.expectCallCount >= 0 || cfg.expectMinCalls >= 0 {
			tb.Cleanup(func() {
				tb.Helper()

				callCount := m.CallCount()

				if cfg.expectCallCount >= 0 && callCount != cfg.expectCallCount {
					tb.Fatalf("expected exactly %d calls, got %d", cfg.expectCallCount, callCount)
				}

				if cfg.expectMinCalls >= 0 && callCount < cfg.expectMinCalls {
					tb.Fatalf("expected at least %d calls, got %d", cfg.expectMinCalls, callCount)
				}
			})
		}
	}
}

// verifyConfig holds verification options for testing integration.
type verifyConfig struct {
	expectCallCount int // Expected exact call count (-1 = don't check)
	expectMinCalls  int // Expected minimum calls (-1 = don't check)
}

// VerifyOption configures automatic verification at test cleanup.
type VerifyOption func(*verifyConfig)

// ExpectCallCount verifies that exactly n calls were made.
//
// Example:
//
//	fakellm.WithTestingT(t, fakellm.ExpectCallCount(2))
func ExpectCallCount(n int) VerifyOption {
	return func(cfg *verifyConfig) {
		cfg.expectCallCount = n
	}
}

// ExpectAtLeastCalls verifies that at least n calls were made.
//
// Example:
//
//	fakellm.WithTestingT(t, fakellm.ExpectAtLeastCalls(1))
func ExpectAtLeastCalls(n int) VerifyOption {
	return func(cfg *verifyConfig) {
		cfg.expectMinCalls = n
	}
}

// ExpectNoCalls verifies that no calls were made to the model.
// This is useful for testing code paths that should not invoke the LLM.
//
// Example:
//
//	fakellm.WithTestingT(t, fakellm.ExpectNoCalls())
func ExpectNoCalls() VerifyOption {
	return ExpectCallCount(0)
}

// CheckCalled returns an error if the matcher never matched any call.
// This is useful with require.NoError(t, err) style assertions.
//
// Example:
//
//	require.NoError(t, model.CheckCalled(fakellm.UserMessageContains("weather")))
func (m *FakeModel) CheckCalled(matcher Matcher) error {
	calls := m.Calls()
	for i, call := range calls {
		cc := &CallContext{
			TotalCalls:      i + 1,
			ConversationKey: conversationKey(call.Request),
		}

		err := matcher(call.Request, cc)
		if err == nil {
			// Match found
			return nil
		}
	}

	return fmt.Errorf("matcher never matched any of %d calls", len(calls))
}

// AssertCalled fails the test if the matcher never matched any call.
//
// Example:
//
//	model.AssertCalled(t, fakellm.UserMessageContains("weather"))
func (m *FakeModel) AssertCalled(tb testing.TB, matcher Matcher) {
	tb.Helper()

	err := m.CheckCalled(matcher)
	if err != nil {
		tb.Fatal(err)
	}
}

// CheckNotCalled returns an error if the matcher matched any call.
//
// Example:
//
//	require.NoError(t, model.CheckNotCalled(fakellm.HasTool("dangerous_tool")))
func (m *FakeModel) CheckNotCalled(matcher Matcher) error {
	calls := m.Calls()
	for i, call := range calls {
		cc := &CallContext{
			TotalCalls:      i + 1,
			ConversationKey: conversationKey(call.Request),
		}

		err := matcher(call.Request, cc)
		if err == nil {
			return fmt.Errorf("matcher matched call #%d (expected no matches)", i+1)
		}
	}

	return nil
}

// AssertNotCalled fails the test if the matcher matched any call.
//
// Example:
//
//	model.AssertNotCalled(t, fakellm.HasTool("dangerous_tool"))
func (m *FakeModel) AssertNotCalled(tb testing.TB, matcher Matcher) {
	tb.Helper()

	err := m.CheckNotCalled(matcher)
	if err != nil {
		tb.Fatal(err)
	}
}

// CheckCallCount returns an error if the call count doesn't match expected.
//
// Example:
//
//	require.NoError(t, model.CheckCallCount(2))
func (m *FakeModel) CheckCallCount(expected int) error {
	actual := m.CallCount()
	if actual != expected {
		return fmt.Errorf("expected %d calls, got %d", expected, actual)
	}

	return nil
}

// AssertCallCount fails the test if the call count doesn't match expected.
func (m *FakeModel) AssertCallCount(tb testing.TB, expected int) {
	tb.Helper()

	err := m.CheckCallCount(expected)
	if err != nil {
		tb.Fatal(err)
	}
}

// AssertGenerateCalled fails the test if Generate was never called.
func (m *FakeModel) AssertGenerateCalled(tb testing.TB) {
	tb.Helper()

	calls := m.Calls()
	for _, call := range calls {
		if call.Kind == CallGenerate {
			return
		}
	}

	tb.Fatalf("Generate was never called (out of %d total calls)", len(calls))
}

// AssertStreamCalled fails the test if GenerateStream was never called.
func (m *FakeModel) AssertStreamCalled(tb testing.TB) {
	tb.Helper()

	calls := m.Calls()
	for _, call := range calls {
		if call.Kind == CallGenerateStream {
			return
		}
	}

	tb.Fatalf("GenerateStream was never called (out of %d total calls)", len(calls))
}

// CallsMatching returns all calls that match the given matcher.
//
// Example:
//
//	weatherCalls := model.CallsMatching(fakellm.UserMessageContains("weather"))
//	assert.Len(t, weatherCalls, 2)
func (m *FakeModel) CallsMatching(matcher Matcher) []Call {
	calls := m.Calls()

	var matching []Call

	for i, call := range calls {
		cc := &CallContext{
			TotalCalls:      i + 1,
			ConversationKey: conversationKey(call.Request),
		}

		err := matcher(call.Request, cc)
		if err == nil {
			matching = append(matching, call)
		}
	}

	return matching
}

// LastCall returns the most recent call, or an error if no calls were made.
func (m *FakeModel) LastCall() (Call, error) {
	calls := m.Calls()
	if len(calls) == 0 {
		return Call{}, errors.New("no calls made")
	}

	return calls[len(calls)-1], nil
}

// FirstCall returns the first call, or an error if no calls were made.
func (m *FakeModel) FirstCall() (Call, error) {
	calls := m.Calls()
	if len(calls) == 0 {
		return Call{}, errors.New("no calls made")
	}

	return calls[0], nil
}

// DebugCalls returns a human-readable summary of all calls for debugging.
// This is useful when tests fail and you want to see what actually happened.
//
// Example:
//
//	t.Log(model.DebugCalls())
func (m *FakeModel) DebugCalls() string {
	calls := m.Calls()
	if len(calls) == 0 {
		return "No calls made"
	}

	var result strings.Builder
	result.WriteString(fmt.Sprintf("Total calls: %d\n", len(calls)))

	var (
		resultSb272 strings.Builder
		resultSb285 strings.Builder
	)

	for i, call := range calls {
		resultSb272.WriteString(fmt.Sprintf("\nCall #%d:\n", i+1))
		resultSb272.WriteString(fmt.Sprintf("  Kind: %s\n", call.Kind))
		resultSb272.WriteString(fmt.Sprintf("  When: %s\n", call.When.Format("15:04:05.000")))

		if call.RuleName != "" {
			resultSb272.WriteString(fmt.Sprintf("  Rule: %s\n", call.RuleName))
		}

		if call.Request != nil {
			resultSb272.WriteString(fmt.Sprintf("  Messages: %d\n", len(call.Request.Messages)))
			resultSb272.WriteString(fmt.Sprintf("  Tools: %d\n", len(call.Request.Tools)))

			// Show last user message if present
			var resultSb286 strings.Builder

			for j := len(call.Request.Messages) - 1; j >= 0; j-- {
				if call.Request.Messages[j].Role == llm.RoleUser {
					text := call.Request.Messages[j].TextContent()
					if len(text) > 50 {
						text = text[:50] + "..."
					}

					resultSb286.WriteString(fmt.Sprintf("  Last user msg: %q\n", text))

					break
				}
			}

			resultSb285.WriteString(resultSb286.String())
		}

		if call.Err != nil {
			resultSb272.WriteString(fmt.Sprintf("  Error: %v\n", call.Err))
		} else if call.Response != nil {
			text := call.Response.TextContent()
			if len(text) > 50 {
				text = text[:50] + "..."
			}

			result.WriteString(fmt.Sprintf("  Response: %q\n", text))
			result.WriteString(fmt.Sprintf("  Finish: %s\n", call.Response.FinishReason))
		}
	}

	result.WriteString(resultSb285.String())

	result.WriteString(resultSb272.String())

	return result.String()
}

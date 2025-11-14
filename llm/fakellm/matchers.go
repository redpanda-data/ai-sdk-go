package fakellm

import (
	"errors"
	"fmt"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Any matches all requests.
// Use this as a catch-all rule that should typically be added last.
//
// Example:
//
//	model.When(fakellm.Any()).ThenRespondText("Default response")
func Any() Matcher {
	return func(_ *llm.Request, _ *CallContext) error {
		return nil // Always matches
	}
}

// And combines multiple matchers with logical AND.
// All matchers must return nil (match) for the combined matcher to match.
//
// Example:
//
//	model.When(fakellm.And(
//	    fakellm.UserMessageContains("weather"),
//	    fakellm.HasTool("get_weather"),
//	)).ThenRespondWithToolCall("get_weather", args)
func And(matchers ...Matcher) Matcher {
	return func(req *llm.Request, cc *CallContext) error {
		for i, m := range matchers {
			err := m(req, cc)
			if err != nil {
				return fmt.Errorf("matcher %d failed: %w", i, err)
			}
		}

		return nil
	}
}

// Or combines multiple matchers with logical OR.
// At least one matcher must return nil (match) for the combined matcher to match.
//
// Example:
//
//	model.When(fakellm.Or(
//	    fakellm.UserMessageContains("hi"),
//	    fakellm.UserMessageContains("hello"),
//	)).ThenRespondText("Hello!")
func Or(matchers ...Matcher) Matcher {
	return func(req *llm.Request, cc *CallContext) error {
		var errs []error

		for _, m := range matchers {
			err := m(req, cc)
			if err == nil {
				return nil // At least one matched
			}

			errs = append(errs, err)
		}

		return fmt.Errorf("no matcher succeeded: %v", errs)
	}
}

// Not negates a matcher.
//
// Example:
//
//	model.When(fakellm.Not(fakellm.HasTool("dangerous_tool"))).
//	    ThenRespondText("Safe to proceed")
func Not(matcher Matcher) Matcher {
	return func(req *llm.Request, cc *CallContext) error {
		err := matcher(req, cc)
		// If original matcher failed (err != nil), then NOT succeeds (return nil)
		// If original matcher succeeded (err == nil), then NOT fails (return error)
		if err == nil {
			return errors.New("NOT condition failed: matcher succeeded when it shouldn't")
		}

		return nil // Original didn't match, so NOT matches
	}
}

// UserMessageContains matches if any user message contains the substring.
// The match is case-sensitive.
//
// Example:
//
//	model.When(fakellm.UserMessageContains("weather")).
//	    ThenRespondText("The weather is sunny!")
func UserMessageContains(substring string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleUser {
				content := msg.TextContent()
				if strings.Contains(content, substring) {
					return nil
				}
			}
		}

		return fmt.Errorf("no user message contains %q", substring)
	}
}

// UserMessageContainsAny matches if any user message contains any of the substrings.
//
// Example:
//
//	model.When(fakellm.UserMessageContainsAny("hi", "hello", "hey")).
//	    ThenRespondText("Hello!")
func UserMessageContainsAny(substrings ...string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleUser {
				content := msg.TextContent()
				for _, sub := range substrings {
					if strings.Contains(content, sub) {
						return nil
					}
				}
			}
		}

		return fmt.Errorf("no user message contains any of %v", substrings)
	}
}

// LastUserMessageContains matches if the last user message contains the substring.
//
// Example:
//
//	model.When(fakellm.LastUserMessageContains("summarize")).
//	    ThenRespondText("Here's a summary...")
func LastUserMessageContains(substring string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		// Find last user message
		for i := len(req.Messages) - 1; i >= 0; i-- {
			if req.Messages[i].Role == llm.RoleUser {
				content := req.Messages[i].TextContent()
				if strings.Contains(content, substring) {
					return nil
				}

				return fmt.Errorf("last user message does not contain %q", substring)
			}
		}

		return errors.New("no user messages found")
	}
}

// HasTool matches if the request includes a tool with the given name.
//
// Example:
//
//	model.When(fakellm.HasTool("get_weather")).
//	    ThenRespondWithToolCall("get_weather", map[string]any{
//	        "location": "San Francisco",
//	    })
func HasTool(toolName string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		for _, tool := range req.Tools {
			if tool.Name == toolName {
				return nil
			}
		}

		return fmt.Errorf("no tool named %q found", toolName)
	}
}

// HasAnyTool matches if the request includes any tools.
//
// Example:
//
//	model.When(fakellm.HasAnyTool()).
//	    ThenRespondText("I can use tools to help!")
func HasAnyTool() Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if len(req.Tools) > 0 {
			return nil
		}

		return errors.New("no tools found in request")
	}
}

// MessageCount matches if the request has exactly n messages.
//
// Example:
//
//	model.When(testing.MessageCount(1)).
//	    ThenRespondText("First message!")
func MessageCount(n int) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if len(req.Messages) == n {
			return nil
		}

		return fmt.Errorf("expected %d messages, got %d", n, len(req.Messages))
	}
}

// TurnIs matches if the conversation is on the specified turn (0-indexed).
// This is useful for scenario-based testing.
//
// Example:
//
//	model.When(fakellm.TurnIs(0)).
//	    ThenRespondWithToolCall("search", args)
func TurnIs(turn int) Matcher {
	return func(_ *llm.Request, cc *CallContext) error {
		if cc.Turn == turn {
			return nil
		}

		return fmt.Errorf("expected turn %d, got turn %d", turn, cc.Turn)
	}
}

// TurnGreaterThan matches if the conversation turn is greater than n.
//
// Example:
//
//	model.When(testing.TurnGreaterThan(5)).
//	    ThenRespondText("This conversation is getting long!")
func TurnGreaterThan(n int) Matcher {
	return func(_ *llm.Request, cc *CallContext) error {
		if cc.Turn > n {
			return nil
		}

		return fmt.Errorf("turn %d is not greater than %d", cc.Turn, n)
	}
}

// FirstTurn matches the first turn (turn 0) of a conversation.
// This is a convenience alias for TurnIs(0) that makes tests more readable.
//
// Example:
//
//	model.When(FirstTurn()).ThenRespondWithToolCall(...)
func FirstTurn() Matcher {
	return TurnIs(0)
}

// FirstCall matches only the first call to the model (across all conversations).
//
// Example:
//
//	model.When(testing.FirstCall()).
//	    ThenRespondText("Welcome! This is my first response.")
func FirstCall() Matcher {
	return func(_ *llm.Request, cc *CallContext) error {
		if cc.TotalCalls == 1 {
			return nil
		}

		return fmt.Errorf("not first call (call #%d)", cc.TotalCalls)
	}
}

// CallNumber matches the nth call to the model (1-indexed, across all conversations).
//
// Example:
//
//	model.When(testing.CallNumber(3)).
//	    ThenError(errors.New("third time's the charm"))
func CallNumber(n int) Matcher {
	return func(_ *llm.Request, cc *CallContext) error {
		if cc.TotalCalls == n {
			return nil
		}

		return fmt.Errorf("expected call #%d, got call #%d", n, cc.TotalCalls)
	}
}

// LastMessageHasToolResponse matches if the last message contains a tool response
// with the given tool name.
//
// Example:
//
//	model.When(fakellm.LastMessageHasToolResponse("get_weather")).
//	    ThenRespondText("Based on the weather data...")
func LastMessageHasToolResponse(toolName string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if len(req.Messages) == 0 {
			return errors.New("no messages in request")
		}

		lastMsg := req.Messages[len(req.Messages)-1]
		if lastMsg.Role != llm.RoleUser {
			return fmt.Errorf("last message is not a tool response (role: %s)", lastMsg.Role)
		}

		for _, part := range lastMsg.Content {
			if part.IsToolResponse() && part.ToolResponse != nil {
				if part.ToolResponse.Name == toolName {
					return nil
				}
			}
		}

		return fmt.Errorf("last message has no tool response for %q", toolName)
	}
}

// HasSystemPrompt matches if the request contains a system message.
//
// Example:
//
//	model.When(fakellm.HasSystemPrompt()).
//	    ThenRespondText("I understand my instructions")
func HasSystemPrompt() Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleSystem {
				return nil
			}
		}

		return errors.New("no system prompt found")
	}
}

// SystemPromptContains matches if any system message contains the substring.
//
// Example:
//
//	model.When(testing.SystemPromptContains("helpful assistant")).
//	    ThenRespondText("I'm here to help!")
func SystemPromptContains(substring string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		for _, msg := range req.Messages {
			if msg.Role == llm.RoleSystem {
				content := msg.TextContent()
				if strings.Contains(content, substring) {
					return nil
				}
			}
		}

		return fmt.Errorf("no system message contains %q", substring)
	}
}

// HasResponseFormat matches if the request specifies a response format.
//
// Example:
//
//	model.When(fakellm.HasResponseFormat()).
//	    ThenRespondWith(structuredResponseBuilder)
func HasResponseFormat() Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if req.ResponseFormat != nil {
			return nil
		}

		return errors.New("no response format specified")
	}
}

// ResponseFormatIs matches if the request's response format type matches.
//
// Example:
//
//	model.When(testing.ResponseFormatIs(llm.ResponseFormatJSONObject)).
//	    ThenRespondText(`{"result": "structured"}`)
func ResponseFormatIs(formatType string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if req.ResponseFormat == nil {
			return errors.New("no response format specified")
		}

		if req.ResponseFormat.Type == formatType {
			return nil
		}

		return fmt.Errorf("response format type is %q, expected %q", req.ResponseFormat.Type, formatType)
	}
}

// MetadataContains matches if the request metadata contains the key-value pair.
//
// Example:
//
//	model.When(testing.MetadataContains("priority", "high")).
//	    ThenRespondText("Processing high priority request")
func MetadataContains(key, value string) Matcher {
	return func(req *llm.Request, _ *CallContext) error {
		if req.Metadata == nil {
			return errors.New("no metadata in request")
		}

		v, ok := req.Metadata[key]
		if !ok {
			return fmt.Errorf("metadata key %q not found", key)
		}

		if v == value {
			return nil
		}

		return fmt.Errorf("metadata[%q] = %q, expected %q", key, v, value)
	}
}

// ConversationKey matches if the conversation key matches.
// The conversation key is derived from metadata or message count.
//
// Example:
//
//	model.When(testing.ConversationKey("session-123")).
//	    ThenRespondText("Welcome back to session 123")
func ConversationKey(key string) Matcher {
	return func(_ *llm.Request, cc *CallContext) error {
		if cc.ConversationKey == key {
			return nil
		}

		return fmt.Errorf("conversation key is %q, expected %q", cc.ConversationKey, key)
	}
}

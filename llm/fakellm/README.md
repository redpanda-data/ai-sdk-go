# FakeLLM Testing Package

A high-quality, flexible mock LLM implementation for integration testing without hitting real AI provider APIs.

## Overview

This package provides a realistic fake LLM provider (`FakeModel`) that implements the `llm.Model` interface with configurable behavior, error simulation, and stateful multi-turn scenarios. It's designed for integration testing of agents and workflows that depend on LLM interactions.

## Why This Package?

Unlike auto-generated gomock mocks that require verbose expectations, this package provides:

- **Ergonomic DSL**: Clean `When(...).Then...()` syntax for readable tests
- **Realistic Behavior**: Proper streaming, token counting, and latency simulation
- **Error Scenarios**: First-class support for rate limits, timeouts, connection drops
- **Stateful Testing**: Multi-turn conversations with tool calling loops
- **Zero External Dependencies**: No API keys or network calls needed
- **Deterministic**: Predictable fallback behavior when no rules match

## Quick Start

### Basic Text Response

```go
model := fakellm.NewFakeModel().
    When(fakellm.UserMessageContains("hello")).
    ThenRespondText("Hello! How can I help you?")

req := &llm.Request{
    Messages: []llm.Message{{
        Role: llm.RoleUser,
        Content: []*llm.Part{llm.NewTextPart("hello world")},
    }},
}

resp, err := model.Generate(ctx, req)
// resp.TextContent() == "Hello! How can I help you?"
```

### Multiple Rules

Rules are evaluated in order. First match wins:

```go
model := fakellm.NewFakeModel().
    When(fakellm.UserMessageContains("weather")).
    ThenRespondText("It's sunny!").
    When(fakellm.UserMessageContains("time")).
    ThenRespondText("It's 3 PM").
    When(fakellm.Any()).
    ThenRespondText("I don't understand")
```

## Core Features

### 1. Streaming Support

Simulate realistic streaming with configurable chunking and delays:

```go
model := fakellm.NewFakeModel().
    When(fakellm.Any()).
    ThenStreamText("This is a streaming response", fakellm.StreamConfig{
        ChunkSize:       8,  // Runes per chunk
        InterChunkDelay: 50 * time.Millisecond,
    })

for event, err := range model.GenerateEvents(ctx, req) {
    if err != nil {
        // Handle error
        break
    }
    // Handle event...
}
```

### 2. Tool Calling

Simulate tool requests and multi-turn tool calling loops:

```go
model := fakellm.NewFakeModel().
    When(fakellm.HasTool("get_weather")).
    ThenRespondWithToolCall("get_weather", map[string]any{
        "location": "San Francisco",
        "unit":     "fahrenheit",
    })

resp, _ := model.Generate(ctx, reqWithTools)
// resp.HasToolRequests() == true
// resp.FinishReason == llm.FinishReasonToolCalls
```

### 3. Error Simulation

Test error handling and retry logic:

```go
// Rate limit on first call, succeed on subsequent calls
model := fakellm.NewFakeModel().
    RateLimitOnce().
    When(fakellm.Any()).
    ThenRespondText("Success!")

// First call fails
_, err := model.Generate(ctx, req)
// errors.Is(err, llm.ErrAPICall) == true - matches production code

// Second call succeeds
resp, _ := model.Generate(ctx, req)
```

**Common Error Patterns:**

```go
// Rate limit N times
model.RateLimitNTimes(3)

// Timeout once
model.TimeoutOnce()

// API error once
model.APIErrorOnce()

// Error after N successful calls
model.ErrorAfterNCalls(5, errors.New("quota exceeded"))

// Pattern: E=Error, S=Success
model.ErrorPattern("EEESSS", llm.ErrAPICall)

// Mid-stream connection drop
model.When(fakellm.Any()).
    ThenStreamText("Will be interrupted", fakellm.StreamConfig{
        ErrorAfterChunks: 3,
        MidStreamError:   llm.ErrStreamClosed,
    })
```

### 4. Stateful Scenarios

Test complex multi-turn conversations with the Scenario builder:

```go
model := fakellm.NewFakeModel().
    Scenario("weather-lookup", func(s *fakellm.ScenarioBuilder) {
        // Turn 0: User asks, model requests tool
        s.OnTurn(0).
            When(fakellm.HasTool("get_weather")).
            ThenRespondWithToolCall("get_weather", map[string]any{
                "location": "San Francisco",
            })

        // Turn 1: Tool result received, model provides answer
        s.OnTurn(1).
            When(fakellm.LastMessageHasToolResponse("get_weather")).
            ThenRespondText("The weather in SF is 68°F and sunny.")
    })

// Use with metadata to track conversations
req1 := &llm.Request{
    Messages: []llm.Message{...},
    Tools:    []llm.ToolDefinition{{Name: "get_weather"}},
    Metadata: map[string]string{"session_id": "user-123"},
}

resp1, _ := model.Generate(ctx, req1) // Turn 0
// ... execute tool ...
resp2, _ := model.Generate(ctx, req2) // Turn 1
```

## Matchers

Matchers determine when a rule should apply:

### Basic Matchers

```go
fakellm.Any()                                    // Matches all requests
fakellm.UserMessageContains("hello")             // User message contains substring
fakellm.UserMessageContainsAny("hi", "hello")    // User message contains any substring
fakellm.LastUserMessageContains("summarize")     // Last user message contains substring
fakellm.MessageCount(3)                          // Exactly N messages
fakellm.FirstCall()                              // First call to model
fakellm.CallNumber(5)                            // Nth call to model
```

### Tool Matchers

```go
fakellm.HasTool("get_weather")                   // Request includes specific tool
fakellm.HasAnyTool()                             // Request includes any tools
fakellm.LastMessageHasToolResponse("tool_name")  // Last message has tool response
```

### Turn Matchers

```go
fakellm.TurnIs(0)                                // Specific turn number
fakellm.TurnGreaterThan(5)                       // Turn greater than N
```

### Advanced Matchers

```go
fakellm.HasSystemPrompt()                        // Has system message
fakellm.SystemPromptContains("helpful")          // System message contains text
fakellm.HasResponseFormat()                      // Has response format specified
fakellm.ResponseFormatIs(llm.ResponseFormatJSONObject)  // Specific format type
fakellm.MetadataContains("key", "value")         // Metadata key-value match
fakellm.ConversationKey("session-123")           // Specific conversation
```

### Combining Matchers

```go
// AND: All must match
fakellm.And(
    fakellm.UserMessageContains("weather"),
    fakellm.HasTool("get_weather"),
    fakellm.TurnIs(0),
)

// OR: At least one must match
fakellm.Or(
    fakellm.UserMessageContains("hi"),
    fakellm.UserMessageContains("hello"),
)

// NOT: Negates a matcher
fakellm.Not(fakellm.HasTool("dangerous_tool"))
```

## Configuration Options

### Model Configuration

```go
model := fakellm.NewFakeModel(
    fakellm.WithModelName("gpt-4-fake"),
    fakellm.WithCapabilities(llm.ModelCapabilities{
        Streaming: true,
        Tools:     true,
        StructuredOutput: false,
    }),
    fakellm.WithLatency(fakellm.LatencyProfile{
        Base:     100 * time.Millisecond,  // Fixed overhead per request
        PerToken: 5 * time.Millisecond,     // Per output token
        PerChunk: 20 * time.Millisecond,    // Between streaming chunks
    }),
    fakellm.WithTokenizer(customTokenizer),
    fakellm.WithChunkSize(8),
    fakellm.WithFallbackFinishReason(llm.FinishReasonLength),
)
```

### Response Options

```go
model.When(fakellm.Any()).
    ThenRespondText("Response text",
        fakellm.WithFinishReason(llm.FinishReasonLength),
    )
```

## Testing Patterns

### Agent Integration Tests

```go
func TestAgent_ToolCallingLoop(t *testing.T) {
    mockLLM := fakellm.NewFakeModel().
        Scenario("calculator", func(s *fakellm.ScenarioBuilder) {
            s.OnTurn(0).ThenRespondWithToolCall("calculate", args)
            s.OnTurn(1).ThenRespondText("The answer is 42")
        })

    agent := agent.New("test-agent", "You are helpful", mockLLM,
        agent.WithTools(toolRegistry))

    result, err := agent.Run(ctx, "session-1", userMessage)
    assert.NoError(t, err)
    assert.Contains(t, result.FinalMessage.TextContent(), "42")
}
```

### Retry Logic Tests

```go
func TestRetryLogic(t *testing.T) {
    mockLLM := fakellm.NewFakeModel().
        RateLimitNTimes(2).  // Fail twice
        When(fakellm.Any()).
        ThenRespondText("Success after retries")

    // Your retry logic here
    var resp *llm.Response
    for i := 0; i < 3; i++ {
        resp, err = mockLLM.Generate(ctx, req)
        if err == nil {
            break
        }
        time.Sleep(backoff)
    }

    require.NoError(t, err)
    assert.Equal(t, "Success after retries", resp.TextContent())
}
```

### Context Cancellation Tests

```go
func TestContextCancellation(t *testing.T) {
    mockLLM := fakellm.NewFakeModel(
        fakellm.WithLatency(fakellm.LatencyProfile{
            Base: 200 * time.Millisecond,
        }),
    )

    ctx, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
    defer cancel()

    _, err := mockLLM.Generate(ctx, req)
    assert.ErrorIs(t, err, context.DeadlineExceeded)
}
```

### Fast Deterministic Time Testing (Go 1.25+)

Use `testing/synctest` for instant, deterministic time tests:

```go
import "testing/synctest"

func TestLatencySimulation(t *testing.T) {
    synctest.Test(t, func(t *testing.T) {
        mockLLM := fakellm.NewFakeModel(
            fakellm.WithLatency(fakellm.LatencyProfile{
                Base: 100 * time.Millisecond,
            }),
        )

        start := time.Now()
        resp, _ := mockLLM.Generate(context.Background(), req)

        // Test runs instantly, but time.Since() reports 100ms
        assert.Equal(t, 100*time.Millisecond, time.Since(start))
        assert.NotNil(t, resp)
    })
}
```

The `synctest.Test()` function runs your test in a "bubble" where `time.Sleep` uses a fake clock. This makes latency-dependent tests run instantly while maintaining realistic timing behavior.

### Streaming Error Recovery

```go
func TestStreamingErrorRecovery(t *testing.T) {
    mockLLM := fakellm.NewFakeModel().
        When(fakellm.Any()).
        ThenStreamText("Content", fakellm.StreamConfig{
            ErrorAfterChunks: 2,
            MidStreamError:   fakellm.ErrConnectionDrop,
        })

    // Verify partial content before error
    chunks := 0
    for _, err := range mockLLM.GenerateEvents(ctx, req) {
        if err != nil {
            // Check against llm package error (matches production code)
            assert.ErrorIs(t, err, llm.ErrStreamClosed)
            break
        }
        chunks++
    }
}
```

## Custom Behaviors

### Custom Response Builder

```go
model.When(fakellm.UserMessageContains("random")).
    ThenRespondWith(func(req *llm.Request, cc *fakellm.CallContext) (*llm.Response, error) {
        // Access request details
        lastMsg := req.Messages[len(req.Messages)-1].TextContent()

        // Access conversation state
        count := cc.Turn + 1

        return &llm.Response{
            Message: llm.Message{
                Role: llm.RoleAssistant,
                Content: []*llm.Part{
                    llm.NewTextPart(fmt.Sprintf("Turn %d: You said: %s", count, lastMsg)),
                },
            },
            FinishReason: llm.FinishReasonStop,
        }, nil
    })
```

### Stateful Responses

```go
s.OnTurn(0).
    WithState("intent", "search").  // Store state
    ThenRespondWithToolCall("search", args)

s.OnTurn(1).
    When(func(req *llm.Request, cc *fakellm.CallContext) bool {
        // Access stored state
        intent, _ := cc.Vars["intent"].(string)
        return intent == "search"
    }).
    ThenRespondText("Here are the results...")
```

## Error Types

The fakellm package uses errors directly from the llm package to match what real providers return:

```go
llm.ErrAPICall           // Provider/API failures (rate limits, server errors, etc.)
llm.ErrInvalidConfig     // Configuration errors (auth, invalid model, bad requests)
llm.ErrStreamClosed      // Stream closed or connection dropped
llm.ErrUnsupportedFeature // Feature not supported
context.DeadlineExceeded // Timeouts and context cancellation
```

### Error Injection and Checking

The fakellm package provides convenient helper methods to inject these errors:

```go
// Configure the fake to return errors
model := fakellm.NewFakeModel().
    RateLimitOnce().  // Returns llm.ErrAPICall on first call
    When(fakellm.Any()).
    ThenRespondText("Success")

// Check errors in assertions using llm package errors
_, err := model.Generate(ctx, req)
errors.Is(err, llm.ErrAPICall)  // true

// You can also inject custom errors or use llm errors directly
model.When(fakellm.Any()).ThenError(llm.ErrInvalidConfig)
```

This ensures that tests using fakellm behave identically to production code using real LLM providers.

## Token Counting

Token usage is automatically calculated:

```go
resp, _ := model.Generate(ctx, req)

assert.Greater(t, resp.Usage.InputTokens, 0)
assert.Greater(t, resp.Usage.OutputTokens, 0)
assert.Equal(t,
    resp.Usage.InputTokens + resp.Usage.OutputTokens,
    resp.Usage.TotalTokens)
```

### Custom Tokenizer

```go
type MyTokenizer struct{}

func (t MyTokenizer) Count(text string) int {
    // Use tiktoken or your own logic
    return len(text) / 4  // Simple approximation
}

model := fakellm.NewFakeModel(
    fakellm.WithTokenizer(MyTokenizer{}),
)
```

## Fallback Behavior

When no rules match, the model provides deterministic fallback behavior:

```go
// Model with no rules
model := fakellm.NewFakeModel()

req := &llm.Request{
    Messages: []llm.Message{
        {Role: llm.RoleUser, Content: []*llm.Part{llm.NewTextPart("Hello")}},
    },
}

resp, _ := model.Generate(ctx, req)
// Echoes the last user message
assert.Equal(t, "Hello", resp.TextContent())
assert.Equal(t, llm.FinishReasonStop, resp.FinishReason)
```

## Best Practices

1. **Be Specific with Matchers**: Use precise matchers to avoid unexpected matches

   ```go
   // Good: Specific
   fakellm.And(
       fakellm.LastUserMessageContains("weather"),
       fakellm.HasTool("get_weather"),
   )

   // Avoid: Too broad
   fakellm.Any()
   ```

2. **Use Scenarios for Multi-Turn Tests**: Scenarios make complex flows readable

   ```go
   // Good: Clear turn-by-turn behavior
   model.Scenario("flow", func(s *fakellm.ScenarioBuilder) {
       s.OnTurn(0).ThenRespondWithToolCall(...)
       s.OnTurn(1).ThenRespondText(...)
   })

   // Avoid: Multiple separate rules with turn matchers
   ```

3. **Test Error Recovery**: Don't just test the happy path

   ```go
   model.RateLimitNTimes(2).When(fakellm.Any()).ThenRespondText("Success")
   // Verify your code handles retries correctly
   ```

4. **Use Metadata for Session Tracking**: Helps with conversation state

   ```go
   req.Metadata = map[string]string{"session_id": "user-123"}
   ```

5. **Verify Token Usage**: Ensure your code respects token limits

   ```go
   assert.LessOrEqual(t, resp.Usage.TotalTokens, maxTokens)
   ```

## Examples

See `fake_test.go` for examples including:

- Basic text responses
- Multiple rule matching
- Tool calling
- Streaming with chunking
- Error patterns
- Context cancellation
- Stateful scenarios
- Token counting
- Complex matchers

## Architecture

The package uses a rule-based matching system:

1. **FakeModel**: Implements `llm.Model` interface
2. **Rules**: Each rule has matchers and actions
3. **Matchers**: Functions that determine if a rule applies
4. **Actions**: Generate responses (non-streaming or streaming)
5. **CallContext**: Tracks state across calls
6. **Scenarios**: Higher-level builder for multi-turn flows

## License

Internal use within redpanda-data/cloudv2 repository.

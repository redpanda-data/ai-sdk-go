package fakellm

import (
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Scenario enables building complex multi-turn conversation flows with stateful behavior.
// Each step in a scenario corresponds to a turn in the conversation.
//
// Scenarios are useful for testing:
//   - Tool calling loops (request tool → execute → respond)
//   - Multi-turn conversations with context
//   - State-dependent responses
//   - Complex agent workflows
//
// Example:
//
//	model := fakellm.NewFakeModel().Scenario("weather-lookup", func(s *fakellm.ScenarioBuilder) {
//	    // Turn 0: User asks about weather, model requests tool call
//	    s.OnTurn(0).
//	        When(fakellm.HasTool("get_weather")).
//	        ThenRespondWithToolCall("get_weather", map[string]any{
//	            "location": "San Francisco",
//	        })
//
//	    // Turn 1: After tool response, model provides final answer
//	    s.OnTurn(1).
//	        When(fakellm.LastMessageHasToolResponse("get_weather")).
//	        ThenRespondText("The weather in San Francisco is 68°F and sunny.")
//	})
func (m *FakeModel) Scenario(name string, build func(*ScenarioBuilder)) *FakeModel {
	sb := &ScenarioBuilder{
		model: m,
		name:  name,
	}

	build(sb)

	return m
}

// ScenarioBuilder provides a fluent API for building multi-turn scenarios.
type ScenarioBuilder struct {
	model *FakeModel
	name  string
}

// OnTurn starts configuring behavior for a specific turn in the conversation.
// Turns are 0-indexed, where turn 0 is the first interaction.
//
// Example:
//
//	s.OnTurn(0).
//	    When(fakellm.UserMessageContains("search")).
//	    ThenRespondWithToolCall("search", args)
func (sb *ScenarioBuilder) OnTurn(turn int) *ScenarioTurnBuilder {
	return &ScenarioTurnBuilder{
		model:    sb.model,
		scenario: sb.name,
		turn:     turn,
	}
}

// ScenarioTurnBuilder configures behavior for a specific turn.
type ScenarioTurnBuilder struct {
	model    *FakeModel
	scenario string
	turn     int
	matchers []Matcher
}

// When adds matchers for this turn.
// The turn matcher is automatically added, so you only need to specify
// additional conditions.
//
// Example:
//
//	s.OnTurn(1).
//	    When(fakellm.HasTool("calculate")).
//	    ThenRespondWithToolCall("calculate", args)
func (stb *ScenarioTurnBuilder) When(matchers ...Matcher) *ScenarioTurnBuilder {
	stb.matchers = append(stb.matchers, matchers...)
	return stb
}

// ThenRespondText sets a text response for this turn.
func (stb *ScenarioTurnBuilder) ThenRespondText(text string, opts ...ResponseOption) *ScenarioBuilder {
	// Combine turn matcher with user matchers
	allMatchers := append([]Matcher{TurnIs(stb.turn)}, stb.matchers...)

	stb.model.When(allMatchers...).
		Named(fmt.Sprintf("%s-turn-%d", stb.scenario, stb.turn)).
		ThenRespondText(text, opts...)

	return &ScenarioBuilder{
		model: stb.model,
		name:  stb.scenario,
	}
}

// ThenRespondWith sets a custom response builder for this turn.
func (stb *ScenarioTurnBuilder) ThenRespondWith(builder func(req *llm.Request, cc *CallContext) (*llm.Response, error)) *ScenarioBuilder {
	allMatchers := append([]Matcher{TurnIs(stb.turn)}, stb.matchers...)

	stb.model.When(allMatchers...).
		Named(fmt.Sprintf("%s-turn-%d", stb.scenario, stb.turn)).
		ThenRespondWith(builder)

	return &ScenarioBuilder{
		model: stb.model,
		name:  stb.scenario,
	}
}

// ThenRespondWithToolCall sets a tool call response for this turn.
func (stb *ScenarioTurnBuilder) ThenRespondWithToolCall(toolName string, arguments map[string]any) *ScenarioBuilder {
	allMatchers := append([]Matcher{TurnIs(stb.turn)}, stb.matchers...)

	stb.model.When(allMatchers...).
		Named(fmt.Sprintf("%s-turn-%d", stb.scenario, stb.turn)).
		ThenRespondWithToolCall(toolName, arguments)

	return &ScenarioBuilder{
		model: stb.model,
		name:  stb.scenario,
	}
}

// ThenStreamText sets a streaming text response for this turn.
func (stb *ScenarioTurnBuilder) ThenStreamText(text string, config StreamConfig) *ScenarioBuilder {
	allMatchers := append([]Matcher{TurnIs(stb.turn)}, stb.matchers...)

	stb.model.When(allMatchers...).
		Named(fmt.Sprintf("%s-turn-%d", stb.scenario, stb.turn)).
		ThenStreamText(text, config)

	return &ScenarioBuilder{
		model: stb.model,
		name:  stb.scenario,
	}
}

// ThenError sets an error response for this turn.
func (stb *ScenarioTurnBuilder) ThenError(err error) *ScenarioBuilder {
	allMatchers := append([]Matcher{TurnIs(stb.turn)}, stb.matchers...)

	stb.model.When(allMatchers...).
		Named(fmt.Sprintf("%s-turn-%d", stb.scenario, stb.turn)).
		ThenError(err)

	return &ScenarioBuilder{
		model: stb.model,
		name:  stb.scenario,
	}
}

// WithState stores a value in the conversation state for later access.
// This is useful for passing data between turns.
//
// Example:
//
//	s.OnTurn(0).
//	    ThenRespondWith(func(req *llm.Request, cc *CallContext) (*llm.Response, error) {
//	        cc.Vars["user_intent"] = "search"
//	        return &llm.Response{...}, nil
//	    })
//
//	s.OnTurn(1).
//	    When(func(req *llm.Request, cc *CallContext) error {
//	        intent, ok := cc.Vars["user_intent"].(string)
//	        if ok && intent == "search" {
//	            return nil  // Match
//	        }
//	        return fmt.Errorf("user_intent is not 'search'")
//	    }).
//	    ThenRespondText("Performing search...")
func (stb *ScenarioTurnBuilder) WithState(key string, value any) *ScenarioTurnBuilder {
	// Add a matcher that stores state when matched
	stb.matchers = append(stb.matchers, func(_ *llm.Request, cc *CallContext) error {
		cc.Vars[key] = value
		return nil // Always matches and stores state
	})

	return stb
}

package fakellm

import (
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"iter"
	"sync"
	"testing"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// FakeModel implements llm.Model with configurable rule-based behavior.
// It provides a realistic mock for integration testing without calling real LLM APIs.
//
// FakeModel uses a rule matching system where each rule has matchers and actions.
// Rules are evaluated in order, and the first matching rule determines the response.
// If no rules match, a deterministic fallback response is generated.
type FakeModel struct {
	mu    sync.RWMutex
	name  string
	caps  llm.ModelCapabilities
	rules []rule
	state *modelState

	tokenizer Tokenizer
	latency   LatencyProfile
	defaults  defaults

	// Conversation tracking
	sessionKeyFrom func(*llm.Request) string

	// Call tracking
	calls []Call

	// Testing integration
	t testing.TB
}

// Ensure FakeModel implements llm.Model at compile time.
var _ llm.Model = (*FakeModel)(nil)

// modelState tracks call history and conversation state for scenario testing.
type modelState struct {
	totalCalls    int
	conversations map[string]*conversationState
}

// conversationState tracks state for a single conversation/session.
type conversationState struct {
	turn int
	vars map[string]any
}

// rule represents a single behavior rule with matchers and actions.
type rule struct {
	name    string
	matcher Matcher
	action  action
	times   *timesConstraint
}

// timesConstraint limits how many times a rule can match.
type timesConstraint struct {
	max int
	hit int
}

// CallKind indicates which method was invoked.
type CallKind int

const (
	// CallGenerate indicates Generate was called.
	CallGenerate CallKind = iota

	// CallGenerateEvents indicates GenerateEvents was called.
	CallGenerateEvents
)

// String returns a human-readable representation of the CallKind.
func (k CallKind) String() string {
	switch k {
	case CallGenerate:
		return "Generate"
	case CallGenerateEvents:
		return "GenerateEvents"
	default:
		return fmt.Sprintf("CallKind(%d)", k)
	}
}

// Call captures a single invocation for later assertions.
type Call struct {
	// When is the time this call was made
	When time.Time

	// Kind indicates whether Generate or GenerateEvents was called
	Kind CallKind

	// Request is the request that was passed
	Request *llm.Request

	// Response is the response that was returned (nil for streaming or errors)
	Response *llm.Response

	// Err is any error that was returned
	Err error

	// RuleName is the name of the rule that matched (empty if no rule matched)
	RuleName string
}

// CallContext provides context about the current Generate/GenerateEvents call.
// This is passed to matchers and actions for stateful decision-making.
type CallContext struct {
	// TotalCalls is the total number of calls to this model across all conversations
	TotalCalls int

	// ConversationKey identifies the conversation/session
	ConversationKey string

	// Turn is the current turn number within this conversation (0-indexed)
	Turn int

	// Vars provides conversation-scoped storage for custom state
	Vars map[string]any
}

// Matcher determines if a rule should be applied to a request.
// Return nil if the rule should apply, or an error explaining why it doesn't match.
// The error is used for debugging and better error messages.
type Matcher func(req *llm.Request, ctx *CallContext) error

// action defines what to do when a rule matches.
// Exactly one of Generate or GenerateEvents should be non-nil.
type action struct {
	Generate       func(ctx context.Context, req *llm.Request, cc *CallContext) (*llm.Response, error)
	GenerateEvents func(ctx context.Context, req *llm.Request, cc *CallContext) iter.Seq2[llm.Event, error]
}

// defaults configures fallback behavior when no rules match.
type defaults struct {
	// FallbackFinishReason is used when no explicit finish reason is provided
	FallbackFinishReason llm.FinishReason

	// ChunkSize is the default number of runes per streaming chunk
	ChunkSize int
}

// NewFakeModel creates a new fake LLM model with optional configuration.
//
// The model starts with sensible defaults:
//   - Model name: "fake-model"
//   - All capabilities enabled
//   - Simple tokenizer (4 chars ≈ 1 token)
//   - Minimal latency
//   - Default chunk size: 16 runes
//
// Example:
//
//	model := fakellm.NewFakeModel(
//	    fakellm.WithModelName("gpt-4-fake"),
//	    fakellm.WithLatency(fakellm.LatencyProfile{
//	        Base: 100 * time.Millisecond,
//	    }),
//	)
func NewFakeModel(opts ...Option) *FakeModel {
	m := &FakeModel{
		name: "fake-model",
		caps: llm.ModelCapabilities{
			Streaming:        true,
			Tools:            true,
			StructuredOutput: true,
			Vision:           false,
			Audio:            false,
			MultiTurn:        true,
			SystemPrompts:    true,
			Reasoning:        true,
		},
		state: &modelState{
			conversations: make(map[string]*conversationState),
		},
		tokenizer: defaultTokenizer{},
		latency: LatencyProfile{
			Base:     5 * time.Millisecond,
			PerToken: time.Millisecond,
			PerChunk: 10 * time.Millisecond,
		},
		defaults: defaults{
			FallbackFinishReason: llm.FinishReasonStop,
			ChunkSize:            16,
		},
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// Name returns the model identifier.
func (m *FakeModel) Name() string {
	return m.name
}

// Capabilities returns the model's supported features.
func (m *FakeModel) Capabilities() llm.ModelCapabilities {
	return m.caps
}

// Generate performs non-streaming generation following the configured rules.
func (m *FakeModel) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	cc := m.beginCall(req)
	defer m.endCall(cc)

	var (
		resp     *llm.Response
		err      error
		ruleName string
	)

	// Find and execute matching rule

	if action, name := m.findMatchingAction(req, cc); action != nil && action.Generate != nil {
		ruleName = name
		resp, err = action.Generate(ctx, req, cc)
	} else {
		// Fallback: deterministic response
		resp, err = m.generateFallback(ctx, req, cc)
	}

	// Log the call
	m.logCall(Call{
		When:     time.Now(),
		Kind:     CallGenerate,
		Request:  req,
		Response: resp,
		Err:      err,
		RuleName: ruleName,
	})

	return resp, err
}

// GenerateEvents performs streaming generation following the configured rules.
func (m *FakeModel) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	// Enforce capability constraints
	if !m.caps.Streaming {
		return func(yield func(llm.Event, error) bool) {
			yield(nil, llm.ErrUnsupportedFeature)
		}
	}

	cc := m.beginCall(req)

	// Find and execute matching rule
	if action, name := m.findMatchingAction(req, cc); action != nil && action.GenerateEvents != nil {
		ruleName := name

		// Log the call (successful stream start)
		m.logCall(Call{
			When:     time.Now(),
			Kind:     CallGenerateEvents,
			Request:  req,
			RuleName: ruleName,
		})

		// Get the iterator from the action
		seq := action.GenerateEvents(ctx, req, cc)

		// Wrap it to call endCall when done
		return func(yield func(llm.Event, error) bool) {
			defer m.endCall(cc)

			for event, err := range seq {
				if !yield(event, err) {
					return
				}
			}
		}
	}

	// Fallback: stream the deterministic response
	seq := m.generateEventsFallback(ctx, req, cc)

	// Log the call
	m.logCall(Call{
		When:    time.Now(),
		Kind:    CallGenerateEvents,
		Request: req,
	})

	// Wrap to call endCall
	return func(yield func(llm.Event, error) bool) {
		defer m.endCall(cc)

		for event, err := range seq {
			if !yield(event, err) {
				return
			}
		}
	}
}

// When starts building a new rule with the given matchers.
// All matchers must return true for the rule to match.
//
// Rules are evaluated in the order they were added (first-match wins).
// If multiple rules could match a request, only the first one will be applied.
// This allows you to define specific rules first, then more general fallbacks.
//
// Example:
//
//	model.When(fakellm.UserMessageContains("weather")).
//	    ThenRespondText("It's sunny!")
func (m *FakeModel) When(matchers ...Matcher) *RuleBuilder {
	return &RuleBuilder{
		model: m,
		rule: rule{
			name:    "rule",
			matcher: And(matchers...),
		},
	}
}

// Calls returns a copy of all recorded calls.
func (m *FakeModel) Calls() []Call {
	m.mu.RLock()
	defer m.mu.RUnlock()

	calls := make([]Call, len(m.calls))
	copy(calls, m.calls)

	return calls
}

// CallCount returns the number of calls made to this model.
func (m *FakeModel) CallCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return len(m.calls)
}

// ResetCalls clears the call log.
func (m *FakeModel) ResetCalls() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.calls = nil
}

// ResetState clears conversation state and turn tracking.
// This is useful for table-driven tests where you want to reuse
// the same model across multiple test cases.
func (m *FakeModel) ResetState() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.state = &modelState{
		conversations: make(map[string]*conversationState),
	}
}

// findMatchingAction finds the first matching rule and returns its action and name.
// It uses a read lock to safely access rules, then upgrades to a write lock
// only if a match is found that needs to update hit count.
func (m *FakeModel) findMatchingAction(req *llm.Request, cc *CallContext) (*action, string) {
	m.mu.RLock()

	// Snapshot rules to avoid holding lock during matcher evaluation
	rulesCopy := make([]rule, len(m.rules))
	copy(rulesCopy, m.rules)
	m.mu.RUnlock()

	// Find matching rule without holding lock
	for i := range rulesCopy {
		r := &rulesCopy[i]

		// Check matcher first (expensive, do outside lock)
		if r.matcher != nil {
			err := r.matcher(req, cc)
			if err != nil {
				// No match, continue to next rule
				continue
			}
		}

		// Matcher succeeded - now acquire write lock and re-check times constraint
		m.mu.Lock()

		if i < len(m.rules) {
			rule := &m.rules[i]

			// Re-check times constraint atomically under write lock
			if rule.times != nil && rule.times.max > 0 && rule.times.hit >= rule.times.max {
				m.mu.Unlock()
				continue // Already exhausted, try next rule
			}

			// Atomically increment hit counter
			if rule.times != nil {
				rule.times.hit++
			}

			action := rule.action
			name := rule.name

			m.mu.Unlock()

			return &action, name
		}

		m.mu.Unlock()
	}

	return nil, ""
}

// logCall records a call for later assertions.
func (m *FakeModel) logCall(call Call) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.calls = append(m.calls, call)
}

// beginCall initializes context for a new call.
func (m *FakeModel) beginCall(req *llm.Request) *CallContext {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.state.totalCalls++

	// Determine conversation key using custom hook or default
	var key string
	if m.sessionKeyFrom != nil {
		key = m.sessionKeyFrom(req)
	} else {
		key = conversationKey(req)
	}

	// Get or create conversation state
	conv := m.state.conversations[key]
	if conv == nil {
		conv = &conversationState{
			turn: 0,
			vars: make(map[string]any),
		}
		m.state.conversations[key] = conv
	} else {
		conv.turn++
	}

	return &CallContext{
		TotalCalls:      m.state.totalCalls,
		ConversationKey: key,
		Turn:            conv.turn,
		Vars:            conv.vars,
	}
}

// endCall is called after a call completes.
func (*FakeModel) endCall(_ *CallContext) {
	// Hook for cleanup or metrics collection if needed
}

// generateFallback creates a deterministic response when no rules match.
func (m *FakeModel) generateFallback(ctx context.Context, req *llm.Request, cc *CallContext) (*llm.Response, error) {
	text, finishReason := m.fallbackContent(req)

	response := &llm.Response{
		Message: llm.Message{
			Role:    llm.RoleAssistant,
			Content: []*llm.Part{llm.NewTextPart(text)},
		},
		FinishReason: finishReason,
		ID:           fmt.Sprintf("fake-%d", cc.TotalCalls),
	}

	return m.addUsageAndLatency(ctx, req, response, text)
}

// generateEventsFallback creates a streaming fallback response.
func (m *FakeModel) generateEventsFallback(ctx context.Context, req *llm.Request, _ *CallContext) iter.Seq2[llm.Event, error] {
	text, finishReason := m.fallbackContent(req)

	events := m.textToStreamEvents(text, finishReason)

	// Return an iterator that yields events with optional delays
	return func(yield func(llm.Event, error) bool) {
		for i, event := range events {
			// Simulate inter-chunk delay
			if m.latency.PerChunk > 0 && i > 0 {
				select {
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				case <-time.After(m.latency.PerChunk):
				}
			}

			// Check context before yielding
			select {
			case <-ctx.Done():
				yield(nil, ctx.Err())
				return
			default:
			}

			if !yield(event, nil) {
				return
			}
		}
	}
}

// fallbackContent generates deterministic content based on the request.
func (m *FakeModel) fallbackContent(req *llm.Request) (string, llm.FinishReason) {
	// Echo the last user message
	var text string

	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == llm.RoleUser {
			text = req.Messages[i].TextContent()
			break
		}
	}

	if text == "" {
		text = "OK"
	}

	finishReason := m.defaults.FallbackFinishReason

	return text, finishReason
}

// addUsageAndLatency adds token usage and simulates latency.
func (m *FakeModel) addUsageAndLatency(ctx context.Context, req *llm.Request, resp *llm.Response, outputText string) (*llm.Response, error) {
	// Calculate tokens
	inputTokens := m.countInputTokens(req)
	outputTokens := m.tokenizer.Count(outputText)

	resp.Usage = &llm.TokenUsage{
		InputTokens:  inputTokens,
		OutputTokens: outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}

	// Simulate latency
	delay := m.latency.Base + time.Duration(outputTokens)*m.latency.PerToken
	if delay > 0 {
		done := make(chan struct{})

		go func() {
			time.Sleep(delay)
			close(done)
		}()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-done:
			// Continue
		}
	}

	return resp, nil
}

// countInputTokens counts tokens in the request messages.
func (m *FakeModel) countInputTokens(req *llm.Request) int {
	total := 0
	for _, msg := range req.Messages {
		total += m.tokenizer.Count(msg.TextContent())
	}

	return total
}

// textToStreamEvents converts text into streaming events with chunking.
func (m *FakeModel) textToStreamEvents(text string, finishReason llm.FinishReason) []llm.Event {
	chunks := chunkText(text, m.defaults.ChunkSize)

	events := make([]llm.Event, 0, len(chunks)+1)
	for i, chunk := range chunks {
		events = append(events, llm.ContentPartEvent{
			Index: i,
			Part:  llm.NewTextPart(chunk),
		})
	}

	// Add finish event
	events = append(events, llm.StreamEndEvent{
		Response: &llm.Response{
			FinishReason: finishReason,
			Usage: &llm.TokenUsage{
				InputTokens:  0, // Calculated in stream wrapper if needed
				OutputTokens: m.tokenizer.Count(text),
				TotalTokens:  m.tokenizer.Count(text),
			},
		},
	})

	return events
}

// responseToStreamEvents converts a Response into streaming events.
// This is used to make ThenRespondWith work for both Generate and GenerateEvents.
func (m *FakeModel) responseToStreamEvents(req *llm.Request, resp *llm.Response) []llm.Event {
	events := make([]llm.Event, 0, len(resp.Message.Content)+1)

	// Emit each content part as a separate event
	for i, part := range resp.Message.Content {
		events = append(events, llm.ContentPartEvent{
			Index: i,
			Part:  part,
		})
	}

	// Calculate usage if not already set
	if resp.Usage == nil {
		inputTokens := m.countInputTokens(req)
		outputTokens := m.tokenizer.Count(resp.Message.TextContent())
		resp.Usage = &llm.TokenUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TotalTokens:  inputTokens + outputTokens,
		}
	}

	// Add finish event
	events = append(events, llm.StreamEndEvent{
		Response: resp,
	})

	return events
}

// conversationKey derives a stable key for conversation tracking.
// It first checks for an explicit session_id in metadata, then falls back
// to a fast non-cryptographic hash (FNV-1a) of all messages.
func conversationKey(req *llm.Request) string {
	// Use metadata if available (preferred)
	if req.Metadata != nil {
		if sessionID, ok := req.Metadata["session_id"]; ok {
			return sessionID
		}
	}

	// Fallback: derive from fast hash of all messages
	// FNV-1a is faster than SHA256 and provides adequate collision resistance
	// for test scenarios
	h := fnv.New64a()
	for _, msg := range req.Messages {
		h.Write([]byte(msg.Role))
		h.Write([]byte{0}) // separator
		h.Write([]byte(msg.TextContent()))
		h.Write([]byte{0}) // separator
	}

	return fmt.Sprintf("conv-%x", h.Sum64())
}

// chunkText splits text into chunks of approximately n runes each.
func chunkText(text string, n int) []string {
	if n <= 0 {
		n = 16
	}

	runes := []rune(text)

	var chunks []string

	for i := 0; i < len(runes); i += n {
		end := min(i+n, len(runes))

		chunks = append(chunks, string(runes[i:end]))
	}

	return chunks
}

// RuleBuilder provides a fluent API for constructing behavior rules.
type RuleBuilder struct {
	model *FakeModel
	rule  rule
}

// Named sets a descriptive name for this rule (useful for debugging).
func (rb *RuleBuilder) Named(name string) *RuleBuilder {
	rb.rule.name = name
	return rb
}

// Times limits how many times this rule can match.
// After reaching the limit, the rule is skipped.
func (rb *RuleBuilder) Times(maxTimes int) *RuleBuilder {
	rb.rule.times = &timesConstraint{max: maxTimes}
	return rb
}

// ThenRespondText configures the rule to return a simple text response.
func (rb *RuleBuilder) ThenRespondText(text string, opts ...ResponseOption) *FakeModel {
	rb.rule.action.Generate = func(ctx context.Context, req *llm.Request, cc *CallContext) (*llm.Response, error) {
		resp := &llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart(text)},
			},
			FinishReason: llm.FinishReasonStop,
			ID:           fmt.Sprintf("fake-%d", cc.TotalCalls),
		}

		// Apply options
		for _, opt := range opts {
			opt(resp)
		}

		return rb.model.addUsageAndLatency(ctx, req, resp, text)
	}

	rb.commit()

	return rb.model
}

// ThenRespondWith configures the rule to return a custom-built response.
// This configures both Generate and GenerateEvents so the response works regardless
// of which method the caller uses.
func (rb *RuleBuilder) ThenRespondWith(builder func(req *llm.Request, cc *CallContext) (*llm.Response, error)) *FakeModel {
	// Configure non-streaming Generate
	rb.rule.action.Generate = func(ctx context.Context, req *llm.Request, cc *CallContext) (*llm.Response, error) {
		resp, err := builder(req, cc)
		if err != nil {
			return nil, err
		}

		// Ensure finish reason is set
		if resp.FinishReason == "" {
			resp.FinishReason = llm.FinishReasonStop
		}

		// Calculate output text for token counting
		outputText := resp.Message.TextContent()

		return rb.model.addUsageAndLatency(ctx, req, resp, outputText)
	}

	// Configure streaming GenerateEvents to return the same response
	rb.rule.action.GenerateEvents = func(ctx context.Context, req *llm.Request, cc *CallContext) iter.Seq2[llm.Event, error] {
		return func(yield func(llm.Event, error) bool) {
			resp, err := builder(req, cc)
			if err != nil {
				yield(nil, err)
				return
			}

			// Ensure finish reason is set
			if resp.FinishReason == "" {
				resp.FinishReason = llm.FinishReasonStop
			}

			// Convert response to streaming events
			events := rb.model.responseToStreamEvents(req, resp)

			// Yield events with optional delays
			for i, event := range events {
				if rb.model.latency.PerChunk > 0 && i > 0 {
					select {
					case <-ctx.Done():
						yield(nil, ctx.Err())
						return
					case <-time.After(rb.model.latency.PerChunk):
					}
				}

				if !yield(event, nil) {
					return
				}
			}
		}
	}

	rb.commit()

	return rb.model
}

// ThenRespondWithToolCall configures the rule to return a tool call request.
func (rb *RuleBuilder) ThenRespondWithToolCall(toolName string, arguments map[string]any) *FakeModel {
	return rb.ThenRespondWith(func(_ *llm.Request, cc *CallContext) (*llm.Response, error) {
		argsJSON, err := json.Marshal(arguments)
		if err != nil {
			return nil, fmt.Errorf("marshal tool arguments: %w", err)
		}

		toolReq := &llm.ToolRequest{
			ID:        fmt.Sprintf("call_%d", cc.TotalCalls),
			Name:      toolName,
			Arguments: argsJSON,
		}

		return &llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewToolRequestPart(toolReq)},
			},
			FinishReason: llm.FinishReasonToolCalls,
			ID:           fmt.Sprintf("fake-%d", cc.TotalCalls),
		}, nil
	})
}

// ThenRespondJSON configures the rule to return a structured JSON response.
// The value is marshaled to JSON and returned as text content.
// Works for both Generate and GenerateEvents.
//
// Example:
//
//	type WeatherData struct {
//	    Temp int    `json:"temp"`
//	    City string `json:"city"`
//	}
//	model.When(fakellm.Any()).ThenRespondJSON(WeatherData{Temp: 72, City: "SF"})
func (rb *RuleBuilder) ThenRespondJSON(value any) *FakeModel {
	return rb.ThenRespondWith(func(_ *llm.Request, cc *CallContext) (*llm.Response, error) {
		jsonBytes, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("marshal JSON response: %w", err)
		}

		return &llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart(string(jsonBytes))},
			},
			FinishReason: llm.FinishReasonStop,
			ID:           fmt.Sprintf("fake-%d", cc.TotalCalls),
		}, nil
	})
}

// ThenStreamText configures the rule to return a streaming text response.
func (rb *RuleBuilder) ThenStreamText(text string, config StreamConfig) *FakeModel {
	finishReason := config.FinishReason
	if finishReason == "" {
		finishReason = llm.FinishReasonStop
	}

	// Configure non-streaming Generate for compatibility
	rb.rule.action.Generate = func(ctx context.Context, req *llm.Request, _ *CallContext) (*llm.Response, error) {
		inputTokens := rb.model.countInputTokens(req)
		outputTokens := rb.model.tokenizer.Count(text)

		resp := &llm.Response{
			Message: llm.Message{
				Role:    llm.RoleAssistant,
				Content: []*llm.Part{llm.NewTextPart(text)},
			},
			FinishReason: finishReason,
			Usage: &llm.TokenUsage{
				InputTokens:  inputTokens,
				OutputTokens: outputTokens,
				TotalTokens:  inputTokens + outputTokens,
			},
		}

		return rb.model.addUsageAndLatency(ctx, req, resp, text)
	}

	// Configure streaming GenerateEvents
	rb.rule.action.GenerateEvents = func(ctx context.Context, req *llm.Request, _ *CallContext) iter.Seq2[llm.Event, error] {
		return func(yield func(llm.Event, error) bool) {
			chunkSize := config.ChunkSize
			if chunkSize <= 0 {
				chunkSize = rb.model.defaults.ChunkSize
			}

			chunks := chunkText(text, chunkSize)
			events := make([]llm.Event, 0, len(chunks)+1)

			// Add text chunks
			for i, chunk := range chunks {
				events = append(events, llm.ContentPartEvent{
					Index: i,
					Part:  llm.NewTextPart(chunk),
				})
			}

			// Add finish event
			inputTokens := rb.model.countInputTokens(req)
			outputTokens := rb.model.tokenizer.Count(text)
			events = append(events, llm.StreamEndEvent{
				Response: &llm.Response{
					Message: llm.Message{
						Role:    llm.RoleAssistant,
						Content: []*llm.Part{llm.NewTextPart(text)},
					},
					FinishReason: finishReason,
					Usage: &llm.TokenUsage{
						InputTokens:  inputTokens,
						OutputTokens: outputTokens,
						TotalTokens:  inputTokens + outputTokens,
					},
				},
			})

			delay := config.InterChunkDelay
			if delay == 0 {
				delay = rb.model.latency.PerChunk
			}

			// Yield events with delays and optional error injection
			for i, event := range events {
				// Check for mid-stream error
				if config.ErrorAfterChunks > 0 && i >= config.ErrorAfterChunks && config.MidStreamError != nil {
					yield(nil, config.MidStreamError)
					return
				}

				// Simulate inter-chunk delay
				if delay > 0 && i > 0 {
					select {
					case <-ctx.Done():
						yield(nil, ctx.Err())
						return
					case <-time.After(delay):
					}
				}

				// Check context before yielding
				select {
				case <-ctx.Done():
					yield(nil, ctx.Err())
					return
				default:
				}

				if !yield(event, nil) {
					return
				}
			}
		}
	}

	rb.commit()

	return rb.model
}

// ThenError configures the rule to return an error.
func (rb *RuleBuilder) ThenError(err error) *FakeModel {
	rb.rule.action.Generate = func(_ context.Context, _ *llm.Request, _ *CallContext) (*llm.Response, error) {
		return nil, err
	}

	rb.rule.action.GenerateEvents = func(_ context.Context, _ *llm.Request, _ *CallContext) iter.Seq2[llm.Event, error] {
		return func(yield func(llm.Event, error) bool) {
			yield(nil, err)
		}
	}

	rb.commit()

	return rb.model
}

// commit adds the rule to the model.
func (rb *RuleBuilder) commit() {
	rb.model.mu.Lock()
	defer rb.model.mu.Unlock()

	rb.model.rules = append(rb.model.rules, rb.rule)
}

// ResponseOption modifies a response.
type ResponseOption func(*llm.Response)

// WithFinishReason sets the finish reason for a response.
func WithFinishReason(reason llm.FinishReason) ResponseOption {
	return func(r *llm.Response) {
		r.FinishReason = reason
	}
}

// StreamConfig configures streaming behavior.
type StreamConfig struct {
	// ChunkSize is the number of runes per chunk (default: model's default)
	ChunkSize int

	// InterChunkDelay is the delay between chunks (default: model's latency profile)
	InterChunkDelay time.Duration

	// FinishReason to use when streaming completes (default: stop)
	FinishReason llm.FinishReason

	// MidStreamError, if set, causes the stream to fail after ErrorAfterChunks chunks
	MidStreamError error

	// ErrorAfterChunks determines when to inject MidStreamError (0 = no error)
	ErrorAfterChunks int
}

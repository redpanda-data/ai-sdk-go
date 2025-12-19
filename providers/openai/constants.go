package openai

import (
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"
)

// Model name constants for commonly used OpenAI models.
// These constants help avoid typos and provide IntelliSense support.
const (
	// ModelGPT5 is the GPT-5 model.
	ModelGPT5 = shared.ChatModelGPT5
	// ModelGPT5Mini is the GPT-5 Mini model.
	ModelGPT5Mini = shared.ChatModelGPT5Mini
	// ModelGPT5Nano is the GPT-5 Nano model.
	ModelGPT5Nano = shared.ChatModelGPT5Nano

	// ModelGPT5_1 is the GPT-5.1 model with configurable adaptive reasoning.
	// Unlike GPT-5, reasoning defaults to 'none' - use WithReasoningEffort() to enable.
	ModelGPT5_1 = shared.ChatModelGPT5_1

	// ModelGPT5_2 is the GPT-5.2 Thinking model (default variant).
	// Designed for complex structured work: coding, math, document analysis, planning.
	// Note: Uses manual string since OpenAI SDK doesn't yet include GPT-5.2.
	// TODO: migrate to shared.ChatModelGPT5_2 when SDK adds support.
	ModelGPT5_2 = shared.ChatModel("gpt-5.2")

	// ModelGPT5_2Instant is the GPT-5.2 Instant model (speed-optimized variant).
	// Optimized for routine queries: writing, translation, information seeking.
	// Note: Uses manual string since OpenAI SDK doesn't yet include GPT-5.2.
	// TODO: migrate to shared.ChatModelGPT5_2Instant when SDK adds support.
	ModelGPT5_2Instant = shared.ChatModel("gpt-5.2-chat-latest")

	// ModelGPT5_2Pro is the GPT-5.2 Pro model (maximum accuracy variant).
	// Achieves highest accuracy on difficult problems requiring deep reasoning.
	// Note: Uses manual string since OpenAI SDK doesn't yet include GPT-5.2.
	// TODO: migrate to shared.ResponsesModelGPT5_2Pro when SDK adds support.
	ModelGPT5_2Pro = shared.ResponsesModel("gpt-5.2-pro")

	// ModelGPT4O is the GPT-4o model.
	ModelGPT4O = shared.ChatModelGPT4o
	// ModelGPT4OMini is the GPT-4o Mini model.
	ModelGPT4OMini = shared.ChatModelGPT4oMini

	// ModelGPT4Turbo is the legacy GPT-4 Turbo model (still supported).
	ModelGPT4Turbo = shared.ChatModelGPT4Turbo
	// ModelGPT35Turbo is the legacy GPT-3.5 Turbo model (still supported).
	ModelGPT35Turbo = shared.ChatModelGPT3_5Turbo

	// ModelGPT41 is the GPT-4.1 model.
	ModelGPT41 = shared.ChatModelGPT4_1
	// ModelGPT41Mini is the GPT-4.1 Mini model.
	ModelGPT41Mini = shared.ChatModelGPT4_1Mini

	// ModelO1Pro is the O1 Pro reasoning model.
	ModelO1Pro = shared.ResponsesModelO1Pro
	// ModelO3 is the O3 reasoning model.
	ModelO3 = shared.ChatModelO3
	// ModelO3Pro is the O3 Pro reasoning model.
	ModelO3Pro = shared.ResponsesModelO3Pro
	// ModelO4Mini is the O4 Mini reasoning model.
	ModelO4Mini = shared.ChatModelO4Mini
)

// ReasoningEffort controls the computational effort for reasoning models.
type ReasoningEffort string

const (
	// ReasoningEffortNone disables reasoning (supported by GPT-5.1+ only).
	ReasoningEffortNone = ReasoningEffort(shared.ReasoningEffortNone)
	// ReasoningEffortMinimal uses the least computational effort for reasoning.
	ReasoningEffortMinimal = ReasoningEffort(shared.ReasoningEffortMinimal)
	// ReasoningEffortLow uses low computational effort for reasoning.
	ReasoningEffortLow = ReasoningEffort(shared.ReasoningEffortLow)
	// ReasoningEffortMedium uses medium computational effort for reasoning (default).
	ReasoningEffortMedium = ReasoningEffort(shared.ReasoningEffortMedium)
	// ReasoningEffortHigh uses high computational effort for reasoning.
	ReasoningEffortHigh = ReasoningEffort(shared.ReasoningEffortHigh)
	// ReasoningEffortXHigh uses extra high computational effort for reasoning (GPT-5.2+).
	ReasoningEffortXHigh = ReasoningEffort(shared.ReasoningEffortXhigh)
)

// ReasoningSummary controls whether and how reasoning traces are summarized.
type ReasoningSummary string

const (
	// ReasoningSummaryAuto automatically determines the best summary level.
	ReasoningSummaryAuto = ReasoningSummary(shared.ReasoningSummaryAuto)
	// ReasoningSummaryConcise provides a brief summary of reasoning traces.
	ReasoningSummaryConcise = ReasoningSummary(shared.ReasoningSummaryConcise)
	// ReasoningSummaryDetailed provides a comprehensive summary of reasoning traces.
	ReasoningSummaryDetailed = ReasoningSummary(shared.ReasoningSummaryDetailed)
)

// Internal API constants derived from the OpenAI SDK.
// These help avoid magic strings and typos in our implementation.
// They are not exported since they are implementation details that users don't need.
//
// Reference: https://platform.openai.com/docs/api-reference/responses

// Response output types - values returned by the API in response.output[].type field.
// These are used for string comparisons in switch statements.
// See: https://platform.openai.com/docs/api-reference/responses/object#responses/object-output
var (
	// outputTypeMessage represents regular text responses from the model.
	outputTypeMessage = string(constant.Message("").Default())

	// outputTypeFunctionCall represents tool calls requested by the model.
	outputTypeFunctionCall = string(constant.FunctionCall("").Default())

	// outputTypeReasoning represents reasoning traces from reasoning models.
	outputTypeReasoning = string(constant.Reasoning("").Default())
)

// Content types - values that appear within message content arrays.
// These are used for string comparisons.
// See: https://platform.openai.com/docs/api-reference/responses/object#responses/object-content
var (
	// contentTypeOutputText represents standard text output from the model.
	contentTypeOutputText = string(constant.OutputText("").Default())
)

// Streaming event types - values returned by the streaming API in event.Type field.
// These are used for event type comparisons in streaming responses.
// See: https://platform.openai.com/docs/api-reference/responses/streaming
var (
	// streamEventOutputTextDelta represents incremental text content during streaming.
	streamEventOutputTextDelta = string(constant.ResponseOutputTextDelta("").Default())

	// streamEventReasoningSummaryTextDelta represents incremental reasoning content during streaming.
	streamEventReasoningSummaryTextDelta = string(constant.ResponseReasoningSummaryTextDelta("").Default())

	// streamEventOutputItemDone represents when an output item (like tool call) is complete during streaming.
	streamEventOutputItemDone = string(constant.ResponseOutputItemDone("").Default())

	// streamEventError represents error events during streaming.
	streamEventError = string(constant.Error("").Default())

	// streamEventResponseCompleted represents the final completion event during streaming.
	streamEventResponseCompleted = string(constant.ResponseCompleted("").Default())
)

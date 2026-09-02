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

package openai

import (
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Model name constants for commonly used OpenAI models.
// These constants help avoid typos and provide IntelliSense support.
const (
	// ModelGPT5 is the GPT-5 model.
	//
	// Deprecated: OpenAI deprecated GPT-5 on 2026-06-11; it shuts down
	// 2026-12-11. Use [ModelGPT5_6Sol].
	ModelGPT5 = shared.ChatModelGPT5
	// ModelGPT5Mini is the GPT-5 Mini model.
	//
	// Deprecated: OpenAI deprecated GPT-5 Mini on 2026-06-11; it shuts
	// down 2026-12-11. Use [ModelGPT5_6Terra].
	ModelGPT5Mini = shared.ChatModelGPT5Mini
	// ModelGPT5Nano is the GPT-5 Nano model.
	//
	// Deprecated: OpenAI deprecated GPT-5 Nano on 2026-06-11; it shuts
	// down 2026-12-11. Use [ModelGPT5_6Luna].
	ModelGPT5Nano = shared.ChatModelGPT5Nano

	// ModelGPT5_1 is the GPT-5.1 model with configurable adaptive reasoning.
	// Unlike GPT-5, reasoning defaults to 'none' - use WithReasoningEffort() to enable.
	ModelGPT5_1 = shared.ChatModelGPT5_1

	// ModelGPT5_2 is the GPT-5.2 Thinking model (default variant).
	ModelGPT5_2 = shared.ChatModelGPT5_2
	// ModelGPT5_2Pro is the GPT-5.2 Pro model (maximum accuracy variant).
	ModelGPT5_2Pro = shared.ChatModelGPT5_2Pro

	// ModelGPT5_2Instant is the GPT-5.2 Instant model.
	//
	// Deprecated: retired by OpenAI on 2026-08-10; requests fail. Use
	// [ModelGPT5_6Sol]. The catalog entry remains so historical usage
	// stays priceable.
	ModelGPT5_2Instant = shared.ChatModelGPT5_2ChatLatest

	// ModelGPT5_3ChatLatest is the GPT-5.3 Chat Latest model.
	//
	// Deprecated: retired by OpenAI on 2026-08-10; requests fail. Use
	// [ModelGPT5_6Sol]. The catalog entry remains so historical usage
	// stays priceable.
	ModelGPT5_3ChatLatest = shared.ChatModelGPT5_3ChatLatest

	// ModelGPT5_6Luna is the cost-optimized GPT-5.6 model.
	ModelGPT5_6Luna = shared.ChatModelGPT5_6Luna
	// ModelGPT5_6Terra balances capability and cost in the GPT-5.6 family.
	ModelGPT5_6Terra = shared.ChatModelGPT5_6Terra
	// ModelGPT5_6Sol is the flagship GPT-5.6 model.
	ModelGPT5_6Sol = shared.ChatModelGPT5_6Sol
	// ModelGPT5_6 is the official alias for GPT-5.6 Sol.
	ModelGPT5_6 = "gpt-5.6"

	// ModelGPT5_5 is the GPT-5.5 flagship model (1M context, coding/professional work).
	// Raw string until the OpenAI SDK adds the constant.
	ModelGPT5_5 = "gpt-5.5"

	// ModelGPT5_4 is the GPT-5.4 flagship model with full reasoning.
	ModelGPT5_4 = shared.ChatModelGPT5_4
	// ModelGPT5_4Mini is the GPT-5.4 Mini model (efficient, full reasoning, 400K context).
	ModelGPT5_4Mini = shared.ChatModelGPT5_4Mini
	// ModelGPT5_4Nano is the GPT-5.4 Nano model (speed-optimized, no reasoning/vision/audio).
	ModelGPT5_4Nano = shared.ChatModelGPT5_4Nano

	// ModelGPT4O is the GPT-4o model.
	ModelGPT4O = shared.ChatModelGPT4o
	// ModelGPT4OMini is the GPT-4o Mini model.
	ModelGPT4OMini = shared.ChatModelGPT4oMini

	// ModelGPT4Turbo is the legacy GPT-4 Turbo model.
	//
	// Deprecated: OpenAI deprecated GPT-4 Turbo on 2026-04-22; it shuts
	// down 2026-10-23. Use [ModelGPT5_6Sol].
	ModelGPT4Turbo = shared.ChatModelGPT4Turbo
	// ModelGPT35Turbo is the legacy GPT-3.5 Turbo model.
	//
	// Deprecated: OpenAI deprecated GPT-3.5 Turbo on 2026-04-22; it shuts
	// down 2026-10-23. Use [ModelGPT5_6Terra].
	ModelGPT35Turbo = shared.ChatModelGPT3_5Turbo

	// ModelGPT41 is the GPT-4.1 model.
	ModelGPT41 = shared.ChatModelGPT4_1
	// ModelGPT41Mini is the GPT-4.1 Mini model.
	ModelGPT41Mini = shared.ChatModelGPT4_1Mini

	// ModelO1Pro is the O1 Pro reasoning model.
	//
	// Deprecated: OpenAI deprecated o1-pro on 2026-04-22; it shuts down
	// 2026-10-23. Use [ModelGPT5_6Sol].
	ModelO1Pro = shared.ResponsesModelO1Pro
	// ModelO3 is the O3 reasoning model.
	//
	// Deprecated: OpenAI deprecated o3 on 2026-06-11; it shuts down
	// 2026-12-11. Use [ModelGPT5_6Sol].
	ModelO3 = shared.ChatModelO3
	// ModelO3Pro is the O3 Pro reasoning model.
	//
	// Deprecated: OpenAI deprecated o3-pro on 2026-06-11; it shuts down
	// 2026-12-11. Use [ModelGPT5_6Sol].
	ModelO3Pro = shared.ResponsesModelO3Pro
	// ModelO4Mini is the O4 Mini reasoning model.
	//
	// Deprecated: OpenAI deprecated o4-mini on 2026-04-22; it shuts down
	// 2026-10-23. Use [ModelGPT5_6Terra].
	ModelO4Mini = shared.ChatModelO4Mini
)

// ReasoningEffort controls the computational effort for reasoning models.
//
// It is an alias of [llm.ReasoningEffort] so effort values are portable
// across provider packages; the constants below declare the values OpenAI
// models accept (derived from the vendor SDK so they always match the wire
// format). Which subset a specific model supports is validated against the
// model catalog in NewModel.
type ReasoningEffort = llm.ReasoningEffort

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
	// ReasoningEffortMax uses maximum computational effort for reasoning (GPT-5.6+).
	ReasoningEffortMax = ReasoningEffort(shared.ReasoningEffortMax)
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

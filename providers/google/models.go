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

package google

import (
	"sync"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Model ID constants for Google Gemini models.
const (
	ModelGemini37Flash      = "gemini-3.7-flash"
	ModelGemini36Flash      = "gemini-3.6-flash"
	ModelGemini35FlashLite  = "gemini-3.5-flash-lite"
	ModelGemini35Flash      = "gemini-3.5-flash"
	ModelGemini31ProPreview = "gemini-3.1-pro-preview"
	// ModelGemini3ProPreview is Gemini 3 Pro Preview.
	//
	// Deprecated: retired by Google on 2026-03-09 and no longer listed by
	// the models API; requests fail. Use [ModelGemini31ProPreview]. The
	// catalog entry remains so historical usage stays priceable.
	ModelGemini3ProPreview   = "gemini-3-pro-preview"
	ModelGemini3FlashPreview = "gemini-3-flash-preview"
	ModelGemini25Pro         = "gemini-2.5-pro"
	ModelGemini25Flash       = "gemini-2.5-flash"
	ModelGemini25FlashLite   = "gemini-2.5-flash-lite"
)

var catalogOnce = sync.OnceValue(func() *catalog.Catalog {
	return catalog.MustNew("google", entries())
})

// Catalog returns the validated Google model catalog: every offering with
// its capabilities, constraints, modalities, reasoning controls, pricing,
// and lifecycle. The catalog is immutable and shared; all reads return
// deep copies.
func Catalog() *catalog.Catalog {
	return catalogOnce()
}

// thinkingBudgetConstraints is the wire-level validation for Gemini 2.5's
// numeric thinking budgets. The public signal that a model accepts a
// budget is catalog Reasoning.Budget; the exact numeric ranges are a
// Google wire detail that stays provider-local, keyed by offering ID in
// thinkingBudgets below.
type thinkingBudgetConstraints struct {
	min          int32
	max          int32
	allowZero    bool
	allowDynamic bool
}

func (c *thinkingBudgetConstraints) supports(budget int32) bool {
	if c == nil {
		return false
	}

	if budget == -1 {
		return c.allowDynamic
	}

	if budget == 0 {
		return c.allowZero
	}

	return budget >= c.min && budget <= c.max
}

var (
	// Gemini 3 uses discrete thinking levels, while Gemini 2.5 uses token
	// budgets with model-specific ranges:
	// https://ai.google.dev/gemini-api/docs/generate-content/thinking
	gemini3ReasoningEfforts = []ReasoningEffort{
		ReasoningEffortMinimal,
		ReasoningEffortLow,
		ReasoningEffortMedium,
		ReasoningEffortHigh,
	}
	gemini3ProReasoningEfforts = []ReasoningEffort{
		ReasoningEffortLow,
		ReasoningEffortHigh,
	}
	gemini31ProReasoningEfforts = []ReasoningEffort{
		ReasoningEffortLow,
		ReasoningEffortMedium,
		ReasoningEffortHigh,
	}
	// Gemini 3.7 Flash rejects "minimal" ("minimal is not supported and
	// returns an error" — model page).
	gemini37FlashReasoningEfforts = []ReasoningEffort{
		ReasoningEffortLow,
		ReasoningEffortMedium,
		ReasoningEffortHigh,
	}
)

// thinkingBudgets holds the numeric budget ranges per offering. Every
// offering with catalog Reasoning.Budget == true must have an entry here
// and vice versa (pinned by TestThinkingBudgetTableMatchesCatalog).
var thinkingBudgets = map[string]*thinkingBudgetConstraints{
	ModelGemini25Pro: {
		min:          128,
		max:          32768,
		allowDynamic: true,
	},
	ModelGemini25Flash: {
		min:          1,
		max:          24576,
		allowZero:    true,
		allowDynamic: true,
	},
	ModelGemini25FlashLite: {
		min:          512,
		max:          24576,
		allowZero:    true,
		allowDynamic: true,
	},
}

// geminiCaps is the capability set shared by every catalogued Gemini
// model.
var geminiCaps = llm.ModelCapabilities{
	Streaming:        true,
	Tools:            true,
	JSONMode:         true, // via response_mime_type
	StructuredOutput: true,
	Vision:           true,
	Audio:            true,
	MultiTurn:        true,
	SystemPrompts:    true,
	Reasoning:        true,
}

// geminiModalities is shared by every catalogued Gemini model: text,
// image, audio, video, and PDF inputs; text output.
var geminiModalities = catalog.Modalities{
	Input: []catalog.Modality{
		catalog.ModalityText, catalog.ModalityImage, catalog.ModalityAudio,
		catalog.ModalityVideo, catalog.ModalityDocument,
	},
	Output: []catalog.Modality{catalog.ModalityText},
}

// geminiParams is the request-parameter surface shared by every
// catalogued Gemini model.
var geminiParams = []string{"temperature", "top_p", "top_k", "max_tokens", "stop", "presence_penalty", "frequency_penalty"}

// entries returns the authored Google catalog.
// Model data: https://ai.google.dev/gemini-api/docs/models
//
// Lifecycle sourcing: https://ai.google.dev/gemini-api/docs/deprecations.
// Google publishes "earliest possible" shutdown dates — those are floors,
// not exact dates, so Retires stays unset. None of the catalogued models
// has an announced shutdown.
func entries() []catalog.Entry {
	return []catalog.Entry{
		{
			ID:           ModelGemini37Flash,
			Model:        catalog.ModelGemini37Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Efforts: gemini37FlashReasoningEfforts,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576,
				MaxOutputTokens:  65536,
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-08-13"),
			},
			// Google lists $0.75/$3.75 (cache $0.075) through 2026-12-31,
			// rising to $1.50/$7.50/$0.15 on 2027-01-01. Only the rate in
			// effect is tracked. No long-context tier is published.
			Pricing: pricing.FlatInfo(0.75, 3.75, 0.075),
		},
		{
			ID:           ModelGemini36Flash,
			Model:        catalog.ModelGemini36Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576,
				MaxOutputTokens:  65536,
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-07-21"),
			},
			// Google lists $0.75/$3.75 through 2026-12-31, rising to
			// $1.50/$7.50 on 2027-01-01. Only the rate in effect is tracked.
			Pricing: pricing.FlatInfo(0.75, 3.75, 0.075),
		},
		{
			ID:           ModelGemini35FlashLite,
			Model:        catalog.ModelGemini35FlashLite,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576,
				MaxOutputTokens:  65536,
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-07-21"),
			},
			Pricing: pricing.FlatInfo(0.30, 2.50, 0.03),
		},
		{
			ID:           ModelGemini35Flash,
			Model:        catalog.ModelGemini35Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Efforts: gemini3ReasoningEfforts,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576,
				MaxOutputTokens:  65535,
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2026-05-19"),
			},
			Pricing: pricing.FlatInfo(1.50, 9.00, 0.15),
		},
		{
			ID:           ModelGemini31ProPreview,
			Model:        catalog.ModelGemini31ProPreview,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Efforts: gemini31ProReasoningEfforts,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 64K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Stage:     catalog.StagePreview,
				Available: catalog.MustDate("2026-02-19"),
			},
			Pricing: pricing.TieredInfo(
				pricing.NewRates(2.00, 12.00, 0.20),
				pricing.Bracket{
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(4.00, 18.00, 0.40),
				},
			),
		},
		{
			ID:           ModelGemini3ProPreview,
			Model:        catalog.ModelGemini3ProPreview,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Efforts: gemini3ProReasoningEfforts,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 65K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Stage:      catalog.StagePreview,
				Available:  catalog.MustDate("2025-11-18"),
				Retires:    catalog.MustDate("2026-03-09"),
				ReplacedBy: ModelGemini31ProPreview,
			},
			Pricing: pricing.TieredInfo(
				pricing.NewRates(2.00, 12.00, 0.20),
				pricing.Bracket{
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(4.00, 18.00, 0.40),
				},
			),
		},
		{
			ID:           ModelGemini3FlashPreview,
			Model:        catalog.ModelGemini3FlashPreview,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Efforts: gemini3ReasoningEfforts,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 65K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Stage:      catalog.StagePreview,
				Available:  catalog.MustDate("2025-12-17"),
				ReplacedBy: ModelGemini36Flash,
			},
			Pricing: pricing.FlatInfo(0.50, 3.00, 0.05),
		},
		{
			ID:           ModelGemini25Pro,
			Model:        catalog.ModelGemini25Pro,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				// Gemini 2.5 thinking is controlled by a numeric token budget
				// (ranges in thinkingBudgets); -1 selects dynamic thinking.
				Adaptive: true,
				Budget:   true,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 65K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-17"),
				Retires:    catalog.MustDate("2026-10-20"),
				ReplacedBy: ModelGemini35Flash,
			},
			Pricing: pricing.TieredInfo(
				pricing.NewRates(1.25, 10.00, 0.125),
				pricing.Bracket{
					MinContextTokens: 200_001,
					Rates:            pricing.NewRates(2.50, 15.00, 0.25),
				},
			),
		},
		{
			ID:           ModelGemini25Flash,
			Model:        catalog.ModelGemini25Flash,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Adaptive: true,
				Budget:   true,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 65K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available:  catalog.MustDate("2025-06-17"),
				Retires:    catalog.MustDate("2026-10-20"),
				ReplacedBy: ModelGemini35FlashLite,
			},
			Pricing: pricing.FlatInfo(0.30, 2.50, 0.03),
		},
		{
			ID:           ModelGemini25FlashLite,
			Model:        catalog.ModelGemini25FlashLite,
			Capabilities: geminiCaps,
			Modalities:   geminiModalities,
			Reasoning: catalog.ReasoningSupport{
				Adaptive: true,
				Budget:   true,
			},
			Constraints: llm.ModelConstraints{
				TemperatureRange: [2]float64{0.0, 2.0},
				MaxInputTokens:   1048576, // 1M input tokens
				MaxOutputTokens:  65535,   // 65K output tokens
				SupportedParams:  geminiParams,
			},
			Life: catalog.Lifecycle{
				Available: catalog.MustDate("2025-07-22"),
				Retires:   catalog.MustDate("2026-10-20"),
				// Google recommends Gemini 3.1 Flash-Lite or Gemma 4; this
				// catalog carries neither, so ReplacedBy stays unset.
			},
			Pricing: pricing.FlatInfo(0.10, 0.40, 0.01),
		},
	}
}

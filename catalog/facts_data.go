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

package catalog

// Canonical ModelIDs for every logical model in the SDK's provider
// catalogs. The vendor segment names the model's creator, not the host:
// Bedrock's "us.anthropic.claude-opus-5" offering and Anthropic's
// "claude-opus-5" offering both reference ModelClaudeOpus5.
const (
	// Anthropic models.

	ModelClaudeFable5   ModelID = "anthropic/claude-fable-5"
	ModelClaudeOpus5    ModelID = "anthropic/claude-opus-5"
	ModelClaudeOpus48   ModelID = "anthropic/claude-opus-4-8"
	ModelClaudeOpus47   ModelID = "anthropic/claude-opus-4-7"
	ModelClaudeOpus46   ModelID = "anthropic/claude-opus-4-6"
	ModelClaudeOpus45   ModelID = "anthropic/claude-opus-4-5"
	ModelClaudeOpus41   ModelID = "anthropic/claude-opus-4-1"
	ModelClaudeSonnet5  ModelID = "anthropic/claude-sonnet-5"
	ModelClaudeSonnet46 ModelID = "anthropic/claude-sonnet-4-6"
	ModelClaudeSonnet45 ModelID = "anthropic/claude-sonnet-4-5"
	ModelClaudeHaiku45  ModelID = "anthropic/claude-haiku-4-5"

	// OpenAI models.

	ModelGPT5          ModelID = "openai/gpt-5"
	ModelGPT5Mini      ModelID = "openai/gpt-5-mini"
	ModelGPT5Nano      ModelID = "openai/gpt-5-nano"
	ModelGPT5_1        ModelID = "openai/gpt-5.1"
	ModelGPT5_2        ModelID = "openai/gpt-5.2"
	ModelGPT5_2Instant ModelID = "openai/gpt-5.2-chat-latest"
	ModelGPT5_2Pro     ModelID = "openai/gpt-5.2-pro"
	ModelGPT5_3Instant ModelID = "openai/gpt-5.3-chat-latest"
	ModelGPT5_4        ModelID = "openai/gpt-5.4"
	ModelGPT5_4Mini    ModelID = "openai/gpt-5.4-mini"
	ModelGPT5_4Nano    ModelID = "openai/gpt-5.4-nano"
	ModelGPT5_5        ModelID = "openai/gpt-5.5"
	ModelGPT5_6Sol     ModelID = "openai/gpt-5.6-sol"
	ModelGPT5_6Terra   ModelID = "openai/gpt-5.6-terra"
	ModelGPT5_6Luna    ModelID = "openai/gpt-5.6-luna"
	ModelGPT4o         ModelID = "openai/gpt-4o"
	ModelGPT4oMini     ModelID = "openai/gpt-4o-mini"
	ModelGPT4Turbo     ModelID = "openai/gpt-4-turbo"
	ModelGPT35Turbo    ModelID = "openai/gpt-3.5-turbo"
	ModelGPT41         ModelID = "openai/gpt-4.1"
	ModelGPT41Mini     ModelID = "openai/gpt-4.1-mini"
	ModelO1Pro         ModelID = "openai/o1-pro"
	ModelO3            ModelID = "openai/o3"
	ModelO3Pro         ModelID = "openai/o3-pro"
	ModelO4Mini        ModelID = "openai/o4-mini"

	// Google models.

	ModelGemini36Flash       ModelID = "google/gemini-3.6-flash"
	ModelGemini35Flash       ModelID = "google/gemini-3.5-flash"
	ModelGemini35FlashLite   ModelID = "google/gemini-3.5-flash-lite"
	ModelGemini31ProPreview  ModelID = "google/gemini-3.1-pro-preview"
	ModelGemini3ProPreview   ModelID = "google/gemini-3-pro-preview"
	ModelGemini3FlashPreview ModelID = "google/gemini-3-flash-preview"
	ModelGemini25Pro         ModelID = "google/gemini-2.5-pro"
	ModelGemini25Flash       ModelID = "google/gemini-2.5-flash"
	ModelGemini25FlashLite   ModelID = "google/gemini-2.5-flash-lite"

	// Models from vendors offered only through Bedrock.

	ModelMistralLarge3 ModelID = "mistral/mistral-large-3"
	ModelNova2Lite     ModelID = "amazon/nova-2-lite"
	ModelGemma4E2B     ModelID = "google/gemma-4-e2b"
	ModelGemma431B     ModelID = "google/gemma-4-31b"
	ModelGemma426BA4B  ModelID = "google/gemma-4-26b-a4b"
)

// defaultRegistry is the built-in Facts registry.
//
// Sourcing: release and knowledge dates were cross-checked against
// models.dev (https://models.dev/api.json) and provider announcements
// at authoring time. Knowledge cutoffs published as a month only are
// normalized to the last day of that month. Corrections flow through
// the lifecycle-sync process, never through code that reads them.
//
// Series discipline: a Series is a NON-BRANCHING succession line. The
// flagship, mini, and nano ladders of one brand are separate series —
// a mini is not the successor of a base model. OpenAI's 5.6 renames
// are mapped onto the lines they succeed (sol ← gpt, terra ← gpt-mini,
// luna ← gpt-nano), matching OpenAI's own recommended replacements.
var defaultRegistry = Registry{
	// ---- Anthropic ----
	ModelClaudeFable5: {
		Name: "Claude Fable 5", Series: "claude-fable",
		Released: MustDate("2026-06-07"),
	},
	ModelClaudeOpus5: {
		Name: "Claude Opus 5", Series: "claude-opus",
		Released: MustDate("2026-07-24"), Knowledge: MustDate("2026-05-31"),
	},
	ModelClaudeOpus48: {
		Name: "Claude Opus 4.8", Series: "claude-opus",
		Released: MustDate("2026-05-28"), Knowledge: MustDate("2026-01-31"),
	},
	ModelClaudeOpus47: {
		Name: "Claude Opus 4.7", Series: "claude-opus",
		Released: MustDate("2026-04-14"), Knowledge: MustDate("2026-01-31"),
	},
	ModelClaudeOpus46: {
		Name: "Claude Opus 4.6", Series: "claude-opus",
		Released: MustDate("2026-02-04"), Knowledge: MustDate("2025-05-31"),
	},
	ModelClaudeOpus45: {
		Name: "Claude Opus 4.5", Series: "claude-opus",
		Released: MustDate("2025-11-24"), Knowledge: MustDate("2025-05-31"),
	},
	ModelClaudeOpus41: {
		Name: "Claude Opus 4.1", Series: "claude-opus",
		Released: MustDate("2025-08-05"), Knowledge: MustDate("2025-03-31"),
	},
	ModelClaudeSonnet5: {
		Name: "Claude Sonnet 5", Series: "claude-sonnet",
		Released: MustDate("2026-06-29"), Knowledge: MustDate("2026-01-31"),
	},
	ModelClaudeSonnet46: {
		Name: "Claude Sonnet 4.6", Series: "claude-sonnet",
		Released: MustDate("2026-02-17"), Knowledge: MustDate("2025-08-31"),
	},
	ModelClaudeSonnet45: {
		Name: "Claude Sonnet 4.5", Series: "claude-sonnet",
		Released: MustDate("2025-09-29"), Knowledge: MustDate("2025-07-31"),
	},
	ModelClaudeHaiku45: {
		Name: "Claude Haiku 4.5", Series: "claude-haiku",
		Released: MustDate("2025-10-15"), Knowledge: MustDate("2025-02-28"),
	},

	// ---- OpenAI ----
	//
	// The flagship line runs gpt-3.5-turbo → gpt-4-turbo → gpt-4o →
	// gpt-4.1 → gpt-5 → 5.1 → 5.2 → 5.4 → 5.5 → gpt-5.6-sol; the -mini
	// ladder ends in terra, the -nano ladder in luna. The chat-tuned
	// "instant" models and the pro models are their own lines.
	ModelGPT5: {
		Name: "GPT-5", Series: "gpt",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-09-30"),
	},
	ModelGPT5Mini: {
		Name: "GPT-5 Mini", Series: "gpt-mini",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-05-31"),
	},
	ModelGPT5Nano: {
		Name: "GPT-5 Nano", Series: "gpt-nano",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-05-31"),
	},
	ModelGPT5_1: {
		Name: "GPT-5.1", Series: "gpt",
		Released: MustDate("2025-11-13"), Knowledge: MustDate("2024-09-30"),
	},
	ModelGPT5_2: {
		Name: "GPT-5.2", Series: "gpt",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_2Instant: {
		Name: "GPT-5.2 Instant", Series: "gpt-chat",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_2Pro: {
		Name: "GPT-5.2 Pro", Series: "gpt-pro",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_3Instant: {
		Name: "GPT-5.3 Instant", Series: "gpt-chat",
		Released: MustDate("2026-03-03"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_4: {
		Name: "GPT-5.4", Series: "gpt",
		Released: MustDate("2026-03-05"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_4Mini: {
		Name: "GPT-5.4 Mini", Series: "gpt-mini",
		Released: MustDate("2026-03-17"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_4Nano: {
		Name: "GPT-5.4 Nano", Series: "gpt-nano",
		Released: MustDate("2026-03-17"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_5: {
		Name: "GPT-5.5", Series: "gpt",
		Released: MustDate("2026-04-23"), Knowledge: MustDate("2025-12-01"),
	},
	ModelGPT5_6Sol: {
		Name: "GPT-5.6 Sol", Series: "gpt",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
	},
	ModelGPT5_6Terra: {
		Name: "GPT-5.6 Terra", Series: "gpt-mini",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
	},
	ModelGPT5_6Luna: {
		Name: "GPT-5.6 Luna", Series: "gpt-nano",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
	},
	ModelGPT4o: {
		Name: "GPT-4o", Series: "gpt",
		Released: MustDate("2024-05-13"), Knowledge: MustDate("2023-09-30"),
	},
	ModelGPT4oMini: {
		Name: "GPT-4o Mini", Series: "gpt-mini",
		Released: MustDate("2024-07-18"), Knowledge: MustDate("2023-09-30"),
	},
	ModelGPT4Turbo: {
		Name: "GPT-4 Turbo", Series: "gpt",
		Released: MustDate("2023-11-06"), Knowledge: MustDate("2023-12-31"),
	},
	ModelGPT35Turbo: {
		Name: "GPT-3.5 Turbo", Series: "gpt",
		Released: MustDate("2023-03-01"), Knowledge: MustDate("2021-09-30"),
	},
	ModelGPT41: {
		Name: "GPT-4.1", Series: "gpt",
		Released: MustDate("2025-04-14"), Knowledge: MustDate("2024-04-30"),
	},
	ModelGPT41Mini: {
		Name: "GPT-4.1 Mini", Series: "gpt-mini",
		Released: MustDate("2025-04-14"), Knowledge: MustDate("2024-04-30"),
	},
	ModelO1Pro: {
		Name: "o1-pro", Series: "gpt-pro",
		Released: MustDate("2025-03-19"), Knowledge: MustDate("2023-09-30"),
	},
	ModelO3: {
		Name: "o3", Series: "o",
		Released: MustDate("2025-04-16"), Knowledge: MustDate("2024-05-31"),
	},
	ModelO3Pro: {
		Name: "o3-pro", Series: "gpt-pro",
		Released: MustDate("2025-06-10"), Knowledge: MustDate("2024-05-31"),
	},
	ModelO4Mini: {
		Name: "o4-mini", Series: "o-mini",
		Released: MustDate("2025-04-16"), Knowledge: MustDate("2024-05-31"),
	},

	// ---- Google ----
	ModelGemini36Flash: {
		Name: "Gemini 3.6 Flash", Series: "gemini-flash",
		Released: MustDate("2026-07-21"), Knowledge: MustDate("2026-03-31"),
	},
	ModelGemini35Flash: {
		Name: "Gemini 3.5 Flash", Series: "gemini-flash",
		Released: MustDate("2026-05-19"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini35FlashLite: {
		Name: "Gemini 3.5 Flash-Lite", Series: "gemini-flash-lite",
		Released: MustDate("2026-07-21"), Knowledge: MustDate("2026-03-31"),
	},
	ModelGemini31ProPreview: {
		Name: "Gemini 3.1 Pro Preview", Series: "gemini-pro",
		Released: MustDate("2026-02-19"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini3ProPreview: {
		Name: "Gemini 3 Pro Preview", Series: "gemini-pro",
		Released: MustDate("2025-11-18"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini3FlashPreview: {
		Name: "Gemini 3 Flash Preview", Series: "gemini-flash",
		Released: MustDate("2025-12-17"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini25Pro: {
		Name: "Gemini 2.5 Pro", Series: "gemini-pro",
		Released: MustDate("2025-06-17"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini25Flash: {
		Name: "Gemini 2.5 Flash", Series: "gemini-flash",
		Released: MustDate("2025-06-17"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini25FlashLite: {
		Name: "Gemini 2.5 Flash-Lite", Series: "gemini-flash-lite",
		Released: MustDate("2025-06-17"), Knowledge: MustDate("2025-01-31"),
	},

	// ---- Bedrock-only vendors ----
	ModelMistralLarge3: {
		Name: "Mistral Large 3", Series: "mistral-large",
		Released: MustDate("2025-12-02"), Knowledge: MustDate("2025-01-31"),
		OpenWeights: true,
	},
	ModelNova2Lite: {
		Name: "Amazon Nova 2 Lite", Series: "nova-lite",
		Released: MustDate("2025-12-02"),
	},
	ModelGemma4E2B: {
		Name: "Gemma 4 E2B", Series: "gemma-e2b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
	},
	ModelGemma431B: {
		Name: "Gemma 4 31B", Series: "gemma-31b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
	},
	ModelGemma426BA4B: {
		Name: "Gemma 4 26B A4B", Series: "gemma-26b-a4b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
	},
}

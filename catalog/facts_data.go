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

	ModelClaudeFable51  ModelID = "anthropic/claude-fable-5-1"
	ModelClaudeFable5   ModelID = "anthropic/claude-fable-5"
	ModelClaudeOpus55   ModelID = "anthropic/claude-opus-5-5"
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
	ModelGPT6Astra     ModelID = "openai/gpt-6-astra"
	ModelGPT6Sol       ModelID = "openai/gpt-6-sol"
	ModelGPT6Luna      ModelID = "openai/gpt-6-luna"
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

	ModelGemini38Flash       ModelID = "google/gemini-3.8-flash"
	ModelGemini37Flash       ModelID = "google/gemini-3.7-flash"
	ModelGemini36Flash       ModelID = "google/gemini-3.6-flash"
	ModelGemini35Flash       ModelID = "google/gemini-3.5-flash"
	ModelGemini35FlashLite   ModelID = "google/gemini-3.5-flash-lite"
	ModelGemini31ProPreview  ModelID = "google/gemini-3.1-pro-preview"
	ModelGemini31FlashLite   ModelID = "google/gemini-3.1-flash-lite"
	ModelGemini3ProPreview   ModelID = "google/gemini-3-pro-preview"
	ModelGemini3FlashPreview ModelID = "google/gemini-3-flash-preview"
	ModelGemini25Pro         ModelID = "google/gemini-2.5-pro"
	ModelGemini25Flash       ModelID = "google/gemini-2.5-flash"
	ModelGemini25FlashLite   ModelID = "google/gemini-2.5-flash-lite"

	// Meta models.

	// ModelMuseSpark13 identifies Meta's Muse Spark 1.3.
	ModelMuseSpark13 ModelID = "meta/muse-spark-1.3"

	// Models from vendors offered only through Bedrock.

	ModelMistralLarge3 ModelID = "mistral/mistral-large-3"
	ModelNova2Lite     ModelID = "amazon/nova-2-lite"
	ModelGemma4E2B     ModelID = "google/gemma-4-e2b"
	ModelGemma431B     ModelID = "google/gemma-4-31b"
	ModelGemma426BA4B  ModelID = "google/gemma-4-26b-a4b"
)

// defaultRegistry is the built-in Facts registry.
//
// Sourcing: release dates, knowledge cutoffs, and descriptions come
// from provider announcements and model cards, cross-checked at
// authoring time. Knowledge cutoffs published as a month only are
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
	ModelClaudeFable51: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Fable 5.1", Series: "claude-fable",
		// Released and reliable knowledge cutoff ("Jun 2026") per
		// platform.claude.com/docs/en/models/fable-5-1/overview.
		Released: MustDate("2026-09-01"), Knowledge: MustDate("2026-06-30"),
		Description: "Claude Fable 5.1 is Anthropic’s most capable Mythos-class model for demanding reasoning and long-horizon agentic work.",
	},
	ModelClaudeFable5: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Fable 5", Series: "claude-fable",
		// Released and reliable knowledge cutoff ("Jan 2026") per
		// platform.claude.com/docs/en/models/fable-5/overview.
		Released: MustDate("2026-06-09"), Knowledge: MustDate("2026-01-31"),
		Description: "Claude Fable 5 is a Mythos-class model from Anthropic, built for autonomous knowledge work and coding.",
	},
	ModelClaudeOpus55: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 5.5", Series: "claude-opus",
		// Released and reliable knowledge cutoff ("Jun 2026") per
		// platform.claude.com/docs/en/models/opus-5-5/overview.
		Released: MustDate("2026-09-22"), Knowledge: MustDate("2026-06-30"),
		Description: "Claude Opus 5.5 is Anthropic's Opus model for long-running agentic coding and knowledge work.",
	},
	ModelClaudeOpus5: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 5", Series: "claude-opus",
		Released: MustDate("2026-07-24"), Knowledge: MustDate("2026-05-31"),
		Description: "Claude Opus 5 is an Anthropic Opus model for demanding reasoning, coding, and long-horizon agentic work.",
	},
	ModelClaudeOpus48: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 4.8", Series: "claude-opus",
		Released: MustDate("2026-05-28"), Knowledge: MustDate("2026-01-31"),
		Description: "Claude Opus 4.8 is an Anthropic Opus model for complex reasoning, coding, and agentic work.",
	},
	ModelClaudeOpus47: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 4.7", Series: "claude-opus",
		// Released per platform.claude.com/docs/en/models/opus-4-7/overview.
		Released: MustDate("2026-04-16"), Knowledge: MustDate("2026-01-31"),
		Description: "Opus 4.7 is the next generation of Anthropic's Opus family, built for long-running, asynchronous agents.",
	},
	ModelClaudeOpus46: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 4.6", Series: "claude-opus",
		// Released per platform.claude.com/docs/en/models/opus-4-6/overview.
		Released: MustDate("2026-02-05"), Knowledge: MustDate("2025-05-31"),
		Description: "Opus 4.6 is an Anthropic Opus model for coding and long-running professional tasks.",
	},
	ModelClaudeOpus45: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 4.5", Series: "claude-opus",
		Released: MustDate("2025-11-24"), Knowledge: MustDate("2025-05-31"),
		Description: "Claude Opus 4.5 is Anthropic’s frontier reasoning model optimized for complex software engineering, agentic workflows, and long-horizon computer use.",
	},
	ModelClaudeOpus41: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Opus 4.1", Series: "claude-opus",
		Released: MustDate("2025-08-05"), Knowledge: MustDate("2025-03-31"),
		Description: "Claude Opus 4.1 is an updated version of Anthropic’s flagship model, offering improved performance in coding, reasoning, and agentic tasks.",
	},
	ModelClaudeSonnet5: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Sonnet 5", Series: "claude-sonnet",
		// Released per platform.claude.com/docs/en/models/sonnet-5/overview.
		Released: MustDate("2026-06-30"), Knowledge: MustDate("2026-01-31"),
		Description: "Sonnet 5 is Anthropic's most capable Sonnet-class model, with frontier performance across coding, agents, and professional work.",
	},
	ModelClaudeSonnet46: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Sonnet 4.6", Series: "claude-sonnet",
		Released: MustDate("2026-02-17"), Knowledge: MustDate("2025-08-31"),
		Description: "Sonnet 4.6 is an Anthropic Sonnet-class model with frontier performance across coding, agents, and professional work.",
	},
	ModelClaudeSonnet45: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Sonnet 4.5", Series: "claude-sonnet",
		// Reliable knowledge cutoff Jan 2025; training data cutoff is Jul 2025.
		Released: MustDate("2025-09-29"), Knowledge: MustDate("2025-01-31"),
		Description: "Claude Sonnet 4.5 is an Anthropic Sonnet model optimized for real-world agents and coding workflows.",
	},
	ModelClaudeHaiku45: {
		Publisher:   PublisherAnthropic,
		DisplayName: "Claude Haiku 4.5", Series: "claude-haiku",
		Released: MustDate("2025-10-15"), Knowledge: MustDate("2025-02-28"),
		Description: "Claude Haiku 4.5 is Anthropic’s fastest and most efficient model, delivering near-frontier intelligence at a fraction of the cost and latency of larger Claude models.",
	},

	// ---- OpenAI ----
	//
	// The flagship line runs gpt-3.5-turbo → gpt-4-turbo → gpt-4o →
	// gpt-4.1 → gpt-5 → 5.1 → 5.2 → 5.4 → 5.5 → gpt-5.6-sol → gpt-6-astra; the -mini
	// ladder runs through gpt-5.6-terra to gpt-6-sol, the -nano ladder through
	// gpt-5.6-luna to gpt-6-luna. GPT-6 moved "Sol" down a tier: Astra is the
	// flagship, so gpt-6-sol succeeds the mid tier, not gpt-5.6-sol. The chat-tuned
	// "instant" models and the pro models are their own lines.
	ModelGPT5: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5", Series: "gpt",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-09-30"),
		Description: "GPT-5 is an OpenAI reasoning model offering major improvements in reasoning, code quality, and user experience.",
	},
	ModelGPT5Mini: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5 Mini", Series: "gpt-mini",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-05-31"),
		Description: "GPT-5 Mini is a compact version of GPT-5, designed to handle lighter-weight reasoning tasks. It provides the same instruction-following and safety-tuning benefits as GPT-5, but with reduced latency and cost.",
	},
	ModelGPT5Nano: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5 Nano", Series: "gpt-nano",
		Released: MustDate("2025-08-07"), Knowledge: MustDate("2024-05-31"),
		Description: "GPT-5-Nano is the smallest and fastest variant in the GPT-5 system, optimized for developer tools, rapid interactions, and ultra-low latency environments.",
	},
	ModelGPT5_1: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.1", Series: "gpt",
		Released: MustDate("2025-11-13"), Knowledge: MustDate("2024-09-30"),
		Description: "GPT-5.1 is a frontier-grade model in the GPT-5 series, offering stronger general-purpose reasoning, improved instruction adherence, and a more natural conversational style compared to GPT-5.",
	},
	ModelGPT5_2: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.2", Series: "gpt",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.2 is a frontier-grade model in the GPT-5 series, offering stronger agentic and long context performance compared to GPT-5.1.",
	},
	ModelGPT5_2Instant: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.2 Instant", Series: "gpt-chat",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.2 Chat (AKA Instant) is the fast, lightweight member of the 5.2 family, optimized for low-latency chat while retaining strong general intelligence.",
	},
	ModelGPT5_2Pro: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.2 Pro", Series: "gpt-pro",
		Released: MustDate("2025-12-11"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.2 Pro is the pro variant of GPT-5.2, offering major improvements in agentic coding and long context performance over GPT-5 Pro.",
	},
	ModelGPT5_3Instant: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.3 Instant", Series: "gpt-chat",
		Released: MustDate("2026-03-03"), Knowledge: MustDate("2025-08-31"),
	},
	ModelGPT5_4: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.4", Series: "gpt",
		Released: MustDate("2026-03-05"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.4 is an OpenAI frontier model unifying the Codex and GPT lines into a single system.",
	},
	ModelGPT5_4Mini: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.4 Mini", Series: "gpt-mini",
		Released: MustDate("2026-03-17"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.4 mini brings the core capabilities of GPT-5.4 to a faster, more efficient model optimized for high-throughput workloads.",
	},
	ModelGPT5_4Nano: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.4 Nano", Series: "gpt-nano",
		Released: MustDate("2026-03-17"), Knowledge: MustDate("2025-08-31"),
		Description: "GPT-5.4 nano is the most lightweight and cost-efficient variant of the GPT-5.4 family, optimized for speed-critical and high-volume tasks.",
	},
	ModelGPT5_5: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.5", Series: "gpt",
		Released: MustDate("2026-04-23"), Knowledge: MustDate("2025-12-01"),
		Description: "GPT-5.5 is OpenAI’s frontier model designed for complex professional workloads, building on GPT-5.4 with stronger reasoning, higher reliability, and improved token efficiency on hard tasks.",
	},
	ModelGPT5_6Sol: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.6 Sol", Series: "gpt",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
		Description: "GPT-5.6 Sol is the flagship model in OpenAI's GPT-5.6 series.",
	},
	ModelGPT5_6Terra: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.6 Terra", Series: "gpt-mini",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
		Description: "GPT-5.6 Terra is a balanced model in OpenAI's GPT-5.6 series, positioned between the flagship Sol tier and the cost-efficient Luna tier.",
	},
	ModelGPT5_6Luna: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-5.6 Luna", Series: "gpt-nano",
		Released: MustDate("2026-07-09"), Knowledge: MustDate("2026-02-16"),
		Description: "GPT-5.6 Luna is a fast, cost-efficient model in OpenAI's GPT-5.6 series.",
	},
	// Announcement: https://openai.com/index/gpt-6-astra/
	// Cutoff: https://developers.openai.com/api/docs/models/gpt-6-astra
	ModelGPT6Astra: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-6 Astra", Series: "gpt",
		Released: MustDate("2026-09-03"), Knowledge: MustDate("2026-04-30"),
		Description: "GPT-6 Astra is OpenAI's flagship model for complex reasoning, coding, and long-running professional work.",
	},
	// Released per developers.openai.com/api/docs/changelog (Sep 22);
	// cutoffs per developers.openai.com/api/docs/models/<id>.
	ModelGPT6Sol: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-6 Sol", Series: "gpt-mini",
		Released: MustDate("2026-09-22"), Knowledge: MustDate("2026-04-20"),
		Description: "GPT-6 Sol is OpenAI's mid-tier GPT-6 model, built for complex coding and agentic workflows.",
	},
	ModelGPT6Luna: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-6 Luna", Series: "gpt-nano",
		Released: MustDate("2026-09-22"), Knowledge: MustDate("2026-05-18"),
		Description: "GPT-6 Luna is OpenAI's most efficient GPT-6 model, for focused, high-volume tasks.",
	},
	ModelGPT4o: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-4o", Series: "gpt",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2024-05-13"), Knowledge: MustDate("2023-10-01"),
		Description: "GPT-4o (\"o\" for \"omni\") is an OpenAI multimodal model supporting both text and image inputs with text outputs.",
	},
	ModelGPT4oMini: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-4o Mini", Series: "gpt-mini",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2024-07-18"), Knowledge: MustDate("2023-10-01"),
		Description: "GPT-4o mini is a small, low-cost successor to GPT-4o, supporting both text and image inputs with text outputs.",
	},
	ModelGPT4Turbo: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-4 Turbo", Series: "gpt",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2023-11-06"), Knowledge: MustDate("2023-12-01"),
		Description: "The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling.",
	},
	ModelGPT35Turbo: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-3.5 Turbo", Series: "gpt",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2023-03-01"), Knowledge: MustDate("2021-09-01"),
		Description: "GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, and is optimized for chat and traditional completion tasks.",
	},
	ModelGPT41: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-4.1", Series: "gpt",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-04-14"), Knowledge: MustDate("2024-06-01"),
		Description: "GPT-4.1 is a flagship large language model optimized for advanced instruction following, real-world software engineering, and long-context reasoning.",
	},
	ModelGPT41Mini: {
		Publisher:   PublisherOpenAI,
		DisplayName: "GPT-4.1 Mini", Series: "gpt-mini",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-04-14"), Knowledge: MustDate("2024-06-01"),
		Description: "GPT-4.1 Mini is a mid-sized model delivering performance competitive with GPT-4o at substantially lower latency and cost.",
	},
	ModelO1Pro: {
		Publisher:   PublisherOpenAI,
		DisplayName: "o1-pro", Series: "gpt-pro",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-03-19"), Knowledge: MustDate("2023-10-01"),
		Description: "The o1 series of models are trained with reinforcement learning to think before they answer and perform complex reasoning.",
	},
	ModelO3: {
		Publisher:   PublisherOpenAI,
		DisplayName: "o3", Series: "o",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-04-16"), Knowledge: MustDate("2024-06-01"),
		Description: "o3 is a well-rounded and powerful model across domains. It sets a new standard for math, science, coding, and visual reasoning tasks. It also excels at technical writing and instruction-following.",
	},
	ModelO3Pro: {
		Publisher:   PublisherOpenAI,
		DisplayName: "o3-pro", Series: "gpt-pro",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-06-10"), Knowledge: MustDate("2024-06-01"),
		Description: "The o-series of models are trained with reinforcement learning to think before they answer and perform complex reasoning.",
	},
	ModelO4Mini: {
		Publisher:   PublisherOpenAI,
		DisplayName: "o4-mini", Series: "o-mini",
		// Knowledge cutoff (exact day) per developers.openai.com/api/docs/models.
		Released: MustDate("2025-04-16"), Knowledge: MustDate("2024-06-01"),
		Description: "OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient performance while retaining strong multimodal and agentic capabilities.",
	},

	// ---- Meta ----
	// https://research.meta.ai/blog/introducing-muse-spark-1-3
	ModelMuseSpark13: {
		Publisher:   PublisherMeta,
		DisplayName: "Muse Spark 1.3", Series: "muse-spark",
		Released: MustDate("2026-09-02"),
		// No knowledge cutoff published in the Meta Model API docs.
		Description: "Muse Spark 1.3 is Meta's model for coding, long-context reasoning, and multi-step agentic workflows.",
	},

	// ---- Google ----
	ModelGemini38Flash: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.8 Flash", Series: "gemini-flash",
		// Released 2026-09-02 per the Gemini API changelog and deprecations
		// page. Knowledge cutoff "March 2026" per the DeepMind model card.
		Released: MustDate("2026-09-02"), Knowledge: MustDate("2026-03-31"),
		Description: "Gemini 3.8 Flash is Google's most intelligent Flash model, built for long-horizon software engineering, autonomous agents, and complex enterprise workflows.",
	},
	ModelGemini37Flash: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.7 Flash", Series: "gemini-flash",
		// Google publishes "Latest update: August 2026" without a day;
		// 2026-08-13 is the public listing date. Knowledge cutoff "March 2026"
		// per deepmind.google/models/model-cards/gemini-3-7-flash.
		Released: MustDate("2026-08-13"), Knowledge: MustDate("2026-03-31"),
		Description: "Gemini 3.7 Flash is a multimodal model from Google for fast agentic workflows, coding, and complex multi-step reasoning.",
	},
	ModelGemini36Flash: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.6 Flash", Series: "gemini-flash",
		Released: MustDate("2026-07-21"), Knowledge: MustDate("2026-03-31"),
		Description: "Gemini 3.6 Flash is a high-efficiency model from Google for coding, agentic workflows, and web and app development.",
	},
	ModelGemini35Flash: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.5 Flash", Series: "gemini-flash",
		Released: MustDate("2026-05-19"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 3.5 Flash is Google's high-efficiency multimodal model, bringing near-Pro level coding and reasoning at Flash-tier cost and speed.",
	},
	ModelGemini35FlashLite: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.5 Flash-Lite", Series: "gemini-flash-lite",
		Released: MustDate("2026-07-21"), Knowledge: MustDate("2026-03-31"),
		Description: "Gemini 3.5 Flash Lite is a high-efficiency model from Google with upgraded agentic capabilities. It is suited for subagents that execute focused tasks within complex, multi-agent workflows.",
	},
	ModelGemini31FlashLite: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.1 Flash-Lite", Series: "gemini-flash-lite",
		// GA on 2026-05-07 per ai.google.dev/gemini-api/docs/changelog, matching
		// the 2.5 Flash-Lite GA-date convention; the preview launched 2026-03-03.
		// Knowledge "Jan 2025" per ai.google.dev/gemini-api/docs/gemini-3.
		Released: MustDate("2026-05-07"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 3.1 Flash-Lite is a low-latency, cost-effective multimodal model optimized for high-frequency, lightweight tasks.",
	},
	ModelGemini31ProPreview: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3.1 Pro Preview", Series: "gemini-pro",
		Released: MustDate("2026-02-19"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 3.1 Pro Preview is Google’s frontier reasoning model, delivering enhanced software engineering performance, improved agentic reliability, and more efficient token usage across complex workflows.",
	},
	ModelGemini3ProPreview: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3 Pro Preview", Series: "gemini-pro",
		Released: MustDate("2025-11-18"), Knowledge: MustDate("2025-01-31"),
	},
	ModelGemini3FlashPreview: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 3 Flash Preview", Series: "gemini-flash",
		Released: MustDate("2025-12-17"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 3 Flash Preview is a high speed, high value thinking model designed for agentic workflows, multi turn chat, and coding assistance.",
	},
	ModelGemini25Pro: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 2.5 Pro", Series: "gemini-pro",
		Released: MustDate("2025-06-17"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathematics, and scientific tasks.",
	},
	ModelGemini25Flash: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 2.5 Flash", Series: "gemini-flash",
		Released: MustDate("2025-06-17"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced reasoning, coding, mathematics, and scientific tasks.",
	},
	ModelGemini25FlashLite: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemini 2.5 Flash-Lite", Series: "gemini-flash-lite",
		// GA on 2025-07-22 per Google's deprecations table; 2025-06-17 was the preview.
		Released: MustDate("2025-07-22"), Knowledge: MustDate("2025-01-31"),
		Description: "Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra-low latency and cost efficiency.",
	},

	// ---- Bedrock-only vendors ----
	ModelMistralLarge3: {
		Publisher:   PublisherMistral,
		DisplayName: "Mistral Large 3", Series: "mistral-large",
		Released: MustDate("2025-12-02"), Knowledge: MustDate("2025-01-31"),
		OpenWeights: true,
		Description: "Mistral Large 3 2512 is Mistral’s most capable model to date, featuring a sparse mixture-of-experts architecture with 41B active parameters (675B total), and released under the Apache 2.0 license.",
	},
	ModelNova2Lite: {
		Publisher:   PublisherAmazon,
		DisplayName: "Amazon Nova 2 Lite", Series: "nova-lite",
		Released:    MustDate("2025-12-02"),
		Description: "Nova 2 Lite is a fast, cost-effective reasoning model for everyday workloads that can process text, images, and videos to generate text.",
	},
	ModelGemma4E2B: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemma 4 E2B", Series: "gemma-e2b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
	},
	ModelGemma431B: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemma 4 31B", Series: "gemma-31b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
		Description: "Gemma 4 31B Instruct is Google DeepMind's 30.7B dense multimodal model supporting text and image input with text output.",
	},
	ModelGemma426BA4B: {
		Publisher:   PublisherGoogle,
		DisplayName: "Gemma 4 26B A4B", Series: "gemma-26b-a4b",
		Released: MustDate("2026-04-02"), OpenWeights: true,
		Description: "Gemma 4 26B A4B IT is an instruction-tuned Mixture-of-Experts (MoE) model from Google DeepMind.",
	},
}

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

// Package pricing implements single-call, token-metered inference pricing
// for LLM models.
//
// # Scope
//
// This package answers one question: "given the token usage reported by
// the provider for one completed model call, what did that call cost at
// the listed per-million-token rates?"
//
// Deliberately out of scope:
//
//   - Non-token charges (Gemini context-cache storage $/M/hour, Bedrock
//     Provisioned Throughput hourly capacity, OpenAI server-tool fees,
//     image generation per-image charges).
//   - Batch/async discount modifiers applied post hoc.
//   - Compositional / stackable modifiers (e.g. Anthropic's "cache ×
//     batch × data-residency" stacking). Selector resolution is
//     winner-take-all.
//   - Multimodal per-modality rates. Today this package models one input
//     rate and one output rate; audio/image buckets would be additive
//     UsageFields when we take that on.
//
// Callers that need any of the above should layer their own accounting
// on top of this package's output rather than extending its types.
//
// # Type overview
//
// The core types compose like this:
//
//	Rates     — the leaf: per-million microcent values for each
//	            UsageField (input, output, cached input, cache writes).
//	Bracket   — one rung of a context-size ladder: MinContextTokens
//	            plus the Rates that apply above it.
//	RateCard  — Base Rates plus optional Brackets. The rate card that
//	            applies under one set of request conditions.
//	Selector  — the request conditions (service tier, speed, region).
//	Override  — binds a Selector to a RateCard.
//	Info      — per-model entry: a Default RateCard and zero-or-more
//	            Selector-scoped Overrides.
//	Catalog   — immutable in-memory lookup table of Infos.
//
// Resolution: Calculate(modelID, usage, req) picks a RateCard (via the
// best matching Override or Default), picks Rates inside it (via the
// matching Bracket or Base), then multiplies usage counts by rates.
//
// # Why microcents
//
// Prices are stored as int64 microcents per million tokens (1 cent =
// 1,000,000 microcents). Microcents represent every current provider
// rate exactly — e.g. GPT-5 Nano's $0.005/M cached input is 500_000
// microcents/M.
//
// Per-call amounts are computed as tokens * rate / 1_000_000 with
// integer floor division, so a single bucket can under-count by up to
// one microcent (1e-8 dollars) per call. Aggregation systems that sum
// thousands of calls per user per month see bounded drift: at worst one
// microcent × number of priced UsageFields × number of calls. That is
// within reporting tolerance; billing systems should not round-trip
// these values as an authoritative source of truth.
//
// To convert a dollar price from a provider's pricing page:
//
//	$2.50/M  → 250_000_000 microcents/M   (dollars × 100_000_000)
//	$0.005/M →     500_000 microcents/M
//
// Always add a dollar-amount comment next to each pricing value for
// readability.
package pricing

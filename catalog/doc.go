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

// Package catalog is the model-metadata read model shared by every
// provider: identity, capabilities, constraints, modalities, pricing,
// and lifecycle for each offering, plus the derivations a discovery
// surface needs (current vs previous generation, successor, retirement,
// price tier).
//
// # Shape
//
// Two identity layers keep cross-provider drift structurally impossible:
//
//   - A ModelID ("anthropic/claude-opus-5") names the logical model and
//     keys a Registry of host-independent Facts (display label, series,
//     release date, knowledge cutoff). Facts are authored once; every
//     offering references them.
//   - An offering ID ("us.anthropic.claude-opus-5") names one invokable
//     row on one provider and carries everything that genuinely varies
//     per host: capabilities, constraints, modalities, pricing, and
//     lifecycle.
//
// # Identity vocabulary
//
// One word per concept, used consistently across the SDK:
//
//   - ID: the invokable, provider-native offering identifier — what
//     NewModel accepts and llm.ModelInfo.Name() returns.
//     "claude-opus-5" on Anthropic, "us.anthropic.claude-opus-5" on
//     Bedrock.
//   - ModelID: the canonical cross-provider identity, linking those two
//     offerings to one model: "anthropic/claude-opus-5". A registry key,
//     never sent to a provider.
//   - DisplayName: the human display name: "Claude Opus 5 (EU)"
//     (Entry.DisplayName, defaulting from the undecorated
//     Facts.DisplayName "Claude Opus 5"); never resolved or invoked.
//
// Timestamped snapshots are minted per revision, so the catalog does not
// enumerate them: the offering ID is the stable family, and Resolve maps
// snapshot forms onto it by longest prefix
// ("claude-sonnet-4-5-20250929" → the "claude-sonnet-4-5" offering).
// model.Name() keeps the verbatim string (normalizing would unpin the
// snapshot); response mappers collapse provider-reported snapshots back
// to the offering ID. Join a model to the catalog with
// Resolve(model.Name()) — it understands every spelling; Lookup does not.
//
// Providers author []Entry values and freeze them with New, which
// validates and returns an immutable Catalog. All reads deep-copy.
//
// # Derive, never store
//
// Anything that is a function of the rest of the catalog is computed at
// read time: adding a model must never require editing or re-checking
// existing entries. Generation ordering derives from Facts.Released;
// succession from series order (Successor) unless the provider announced
// a replacement (Lifecycle.ReplacedBy); retirement and deprecation from
// their dates. Clock-dependent classification lives exclusively on View
// (Catalog.At / Catalog.Now), so derived artifacts built from Catalog
// alone are reproducible.
//
// # Unknown models
//
// Resolve reports ok == false for models the catalog does not know.
// Unknown means "stop enforcing", never "assume a baseline": callers
// must not guess capabilities, constraints, or pricing for it. In
// particular, pricing lookups for unknown models must surface as
// unpriced rather than free.
package catalog

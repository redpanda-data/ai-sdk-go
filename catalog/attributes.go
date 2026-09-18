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

import (
	"errors"
	"fmt"
)

// AttributePublisher is the Entry.Attributes key naming the vendor that
// published the model, independent of the provider serving it: Bedrock's
// "us.anthropic.claude-opus-5" and Anthropic's "claude-opus-5" both
// publish "anthropic".
//
// This key lives in the shared package rather than per provider because
// every catalog declares it — it is the one attribute that is not a
// provider quirk. Consumers read a model's vendor from it instead of
// pattern-matching the offering ID, and the snapshot command's
// TestEveryOfferingDeclaresAPublisher fails the build when an offering in
// any catalog it covers omits it. That test walks allCatalogs in
// cmd/catalog-snapshot/lifecycle_test.go, which is hand-kept and separate
// from the generator's own list in cmd/catalog-snapshot/main.go: a new
// provider package is guarded only once it is added to both.
//
// Shared keys are named Attribute*, and a key only one provider carries
// stays ModelMetadata* in that provider's own package. This is the first
// shared one, so it is also the rule.
//
// The vocabulary is lowercase vendor names, and it is open: a new vendor
// appearing in any catalog adds a value. The Publisher consts below name
// the ones in use today.
const AttributePublisher = "publisher"

// The publisher values in use across the catalogs today. The set stays
// open: a new vendor adds a const here and its catalog declares it. These
// exist so a consumer branching on a publisher value compares against a
// symbol instead of hand-spelling the string, the way http.MethodGet sits
// beside an arbitrary method string.
const (
	PublisherAmazon    = "amazon"
	PublisherAnthropic = "anthropic"
	PublisherGoogle    = "google"
	PublisherMeta      = "meta"
	PublisherMistral   = "mistral"
	PublisherOpenAI    = "openai"
)

// MustDeclarePublisher declares publisher on every entry in entries and
// returns them, for the single-vendor catalogs where one name covers the
// whole slice: Anthropic serves only Anthropic models, Meta only Meta's. The
// value is still authored — one declaration at the catalog's root instead
// of the same literal repeated on two dozen entries — and never derived
// from an offering ID.
//
// Multi-vendor catalogs cannot use it: Bedrock authors a publisher per
// model family. An entry that already declares a different publisher is an
// authoring error, and panics, because it means the catalog is not
// single-vendor after all.
//
// entries is mutated in place and returned, so call it on a freshly built
// slice, which is what a provider's entries() returns.
//
// It carries the Must prefix because it panics on an authoring error,
// like MustNew in catalog.go and MustDate in date.go.
func MustDeclarePublisher(publisher string, entries []Entry) []Entry {
	if publisher == "" {
		panic(errors.New("catalog: MustDeclarePublisher needs a publisher")) //nolint:forbidigo // authoring error, not runtime
	}

	for i := range entries {
		if existing := entries[i].Attributes[AttributePublisher]; existing != "" && existing != publisher {
			panic(fmt.Errorf("catalog: entry %s declares publisher %s, not %s", entries[i].ID, existing, publisher)) //nolint:forbidigo // authoring error, not runtime
		}

		if entries[i].Attributes == nil {
			entries[i].Attributes = make(map[string]string, 1)
		}

		entries[i].Attributes[AttributePublisher] = publisher
	}

	return entries
}

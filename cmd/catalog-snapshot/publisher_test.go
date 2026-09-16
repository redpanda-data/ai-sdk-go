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

package main

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
)

// TestEveryOfferingDeclaresAPublisher is the tripwire that keeps a newly
// added model from shipping without a vendor. The console reads
// Model.publisher to pick a model's brand mark; an offering with no
// publisher falls through to a neutral glyph, and no fallback covers it.
//
// It runs over allCatalogs rather than a fixed list of provider packages,
// so a new provider package fails here until it declares a publisher — from
// the moment it is added to allCatalogs, which registering it for the
// snapshot already requires.
func TestEveryOfferingDeclaresAPublisher(t *testing.T) {
	t.Parallel()

	for _, cat := range allCatalogs() {
		want, singleVendor := singleVendorPublishers[cat.Provider()]

		for _, o := range cat.All() {
			assert.NotEmptyf(t, o.Attributes[catalog.AttributePublisher],
				"%s/%s declares no %s attribute", cat.Provider(), o.ID, catalog.AttributePublisher)

			if singleVendor {
				assert.Equalf(t, want, o.Attributes[catalog.AttributePublisher],
					"%s/%s publisher", cat.Provider(), o.ID)
			}
		}
	}
}

// singleVendorPublishers is the publisher every offering of a
// single-vendor catalog must carry. The provider name is the publisher
// for these four, and the two multi-word provider names are the reason
// this is a lookup rather than blanket equality: aws.bedrock is
// multi-vendor and covered by TestBedrockPublisherMatchesBareIDVendor,
// and gcp.vertex publishes google and anthropic models and is covered by
// a want-map in providers/vertex/models_test.go.
var singleVendorPublishers = map[string]string{
	"anthropic": "anthropic",
	"google":    "google",
	"meta":      "meta",
	"openai":    "openai",
}

// TestBedrockPublisherMatchesBareIDVendor checks the multi-vendor catalog
// against an independent derivation. Bedrock offering IDs carry an
// inference-profile geo prefix ("us.anthropic.claude-opus-5") while the
// vendor namespace belongs to the bare ID, so a publisher authored per
// family must agree across all of that family's variants — bare,
// geo-prefixed and global.
//
// The split lives here and only here: production authors the publisher on
// the family declaration, and this test re-derives it from the ID to prove
// the two agree.
func TestBedrockPublisherMatchesBareIDVendor(t *testing.T) {
	t.Parallel()

	cat := bedrock.Catalog()

	for _, o := range cat.All() {
		assert.Equalf(t, bareIDVendor(o.ID), o.Attributes[catalog.AttributePublisher],
			"%s publisher does not match its bare ID vendor", o.ID)
	}
}

// geoPrefixes are the inference-profile geographies a Bedrock offering ID
// may be prefixed with. Kept local to the test so it derives the vendor
// without borrowing production's tables.
var geoPrefixes = map[string]bool{
	"global": true,
	"us":     true,
	"eu":     true,
	"au":     true,
	"jp":     true,
}

// bareIDVendor returns the vendor namespace of a Bedrock offering ID,
// skipping a leading geo prefix: both "anthropic.claude-opus-5" and
// "us.anthropic.claude-opus-5" yield "anthropic".
func bareIDVendor(id string) string {
	head, rest, ok := strings.Cut(id, ".")
	if !ok {
		return head
	}

	if geoPrefixes[head] {
		head, _, _ = strings.Cut(rest, ".")
	}

	return head
}

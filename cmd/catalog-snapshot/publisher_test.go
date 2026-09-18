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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
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
		singleVendor := singleVendorProviders[cat.Provider()]
		require.Truef(t, singleVendor || multiVendorProviders[cat.Provider()],
			"%s is in neither singleVendorProviders nor multiVendorProviders", cat.Provider())

		for _, o := range cat.All() {
			assert.NotEmptyf(t, o.Attributes[catalog.AttributePublisher],
				"%s/%s declares no %s attribute", cat.Provider(), o.ID, catalog.AttributePublisher)

			if singleVendor {
				assert.Equalf(t, cat.Provider(), o.Attributes[catalog.AttributePublisher],
					"%s/%s publisher", cat.Provider(), o.ID)
			}
		}
	}
}

// singleVendorProviders are the catalogs whose provider name is the
// publisher of every offering they carry, so equality against the
// provider name is the whole check.
//
// multiVendorProviders are the rest, where a publisher never equals the
// provider name and the value is checked in the provider's own package:
// aws.bedrock by TestPublisherMatchesBareIDVendor in
// providers/bedrock/publisher_test.go, and gcp.vertex by
// TestOfferingAttributes in providers/vertex/models_test.go.
//
// Every catalog must appear in exactly one of the two. A catalog in
// neither fails the test above rather than silently keeping the presence
// check and losing the value check.
var (
	singleVendorProviders = map[string]bool{
		"anthropic": true,
		"google":    true,
		"meta":      true,
		"openai":    true,
	}

	multiVendorProviders = map[string]bool{
		"aws.bedrock": true,
		"gcp.vertex":  true,
	}
)

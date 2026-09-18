// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package bedrock

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPublisherMatchesBareIDVendor checks the multi-vendor catalog against
// an independent derivation. A family's BareID opens with the vendor
// namespace ("anthropic.claude-opus-5") while the publisher is authored
// separately on the same declaration, so the two must agree.
//
// It asserts on the family declarations rather than on the expanded
// offerings, which is what keeps the derivation free of production's
// tables: a BareID carries no inference-profile geo prefix, so splitting
// it needs no list of geographies. Nothing is lost by not walking the
// offerings - variant copies f.Publisher into every entry of the family
// unchanged (families.go), and per-offering presence is
// TestEveryOfferingDeclaresAPublisher in cmd/catalog-snapshot.
func TestPublisherMatchesBareIDVendor(t *testing.T) {
	t.Parallel()

	for _, f := range bedrockFamilies {
		vendor, _, ok := strings.Cut(f.BareID, ".")
		require.Truef(t, ok, "%s carries no vendor namespace", f.BareID)
		assert.Equalf(t, vendor, f.Publisher, "%s publisher does not match its bare ID vendor", f.BareID)
	}
}

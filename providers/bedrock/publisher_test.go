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

package bedrock

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPublisherMatchesBareIDVendor checks each publisher against the
// vendor segment of the ID. A family's BareID leads with that vendor,
// as in "anthropic.claude-opus-5", but an expanded offering ID may put
// a geo prefix in front of it, as in "us.anthropic.claude-opus-5", so
// the offering loop accepts the vendor in either position.
func TestPublisherMatchesBareIDVendor(t *testing.T) {
	t.Parallel()

	for _, f := range bedrockFamilies {
		vendor, _, ok := strings.Cut(f.BareID, ".")
		require.Truef(t, ok, "%s carries no vendor namespace", f.BareID)
		assert.Equalf(t, vendor, f.Publisher, "%s publisher does not match its bare ID vendor", f.BareID)
	}

	for _, o := range Catalog().All() {
		publisher := o.Publisher
		require.NotEmptyf(t, publisher, "%s declares no publisher", o.ID)
		assert.Truef(t,
			strings.HasPrefix(o.ID, publisher+".") || strings.Contains(o.ID, "."+publisher+"."),
			"%s publisher %q is not a vendor segment of its ID", o.ID, publisher)
	}
}

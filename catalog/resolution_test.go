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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolutionPolicyValidation(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.ResolutionPolicy = "unsupported_resolution"
	_, err := New("acme", []Entry{e}, WithRegistry(testRegistry()))
	require.ErrorContains(t, err, `ResolutionPolicy "unsupported_resolution" is not a valid resolution policy`)
}

func TestResolutionPolicy(t *testing.T) {
	t.Parallel()

	for _, policy := range []ResolutionPolicy{ResolutionPolicyVersionStamp, ResolutionPolicyExact} {
		t.Run(string(policy), func(t *testing.T) {
			t.Parallel()

			e := wireSupportEntry()
			e.ResolutionPolicy = policy

			c := mustCatalog(t, e)
			for _, id := range []string{e.ID, e.Aliases[0]} {
				o, ok := c.Resolve(id)
				require.True(t, ok)
				assert.Equal(t, e.ID, o.ID)
				assert.Equal(t, policy, o.ResolutionPolicy)

				for _, suffix := range []string{"-2026-11-04", "@001", "-001"} {
					stamped, known := c.Resolve(id + suffix)

					resolvedID, resolved := c.ResolveID(id + suffix)
					if policy == ResolutionPolicyExact {
						assert.False(t, known, "%s%s", id, suffix)
						assert.False(t, resolved)
						assert.Empty(t, resolvedID)
						assert.Empty(t, stamped.WireAPIs)
						assert.Empty(t, stamped.WireAPISupport)
					} else {
						require.True(t, known, "%s%s", id, suffix)
						assert.True(t, resolved)
						assert.Equal(t, e.ID, resolvedID)
						assert.Equal(t, o.WireAPIs, stamped.WireAPIs)
						assert.Equal(t, o.WireAPISupport, stamped.WireAPISupport)
					}
				}
			}

			_, ok := c.PricingByID()[e.ID+"-001"]
			assert.False(t, ok, "unlisted IDs must not be synthesized in the exact pricing map")
		})
	}
}

func TestResolutionPolicyIsPerOffering(t *testing.T) {
	t.Parallel()

	exact := wireSupportEntry()
	exact.ResolutionPolicy = ResolutionPolicyExact
	legacy := validEntry("legacy-robin", exact.Model)
	legacy.Aliases = []string{"legacy-latest"}
	registered := validEntry(exact.ID+"-20260101", exact.Model)
	registered.ResolutionPolicy = ResolutionPolicyExact
	c := mustCatalog(t, exact, legacy, registered)

	for _, id := range []string{exact.ID, exact.Aliases[0], registered.ID} {
		_, ok := c.ResolveID(id)
		assert.True(t, ok)
		_, ok = c.ResolveID(id + "@001")
		assert.False(t, ok)
	}

	for _, id := range []string{legacy.ID, legacy.Aliases[0]} {
		resolved, ok := c.ResolveID(id + "-20260101")
		require.True(t, ok)
		assert.Equal(t, legacy.ID, resolved, "the exact-policy addition must not change legacy offerings")
	}
}

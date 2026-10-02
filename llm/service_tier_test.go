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

package llm

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeServiceTier(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		raw  string
		want ServiceTier
	}{
		{"", ""},
		{"  ", ""},
		{"default", ServiceTierDefault},
		{"standard", ServiceTierDefault},
		{"auto", ServiceTierDefault},
		{"flex", ServiceTierFlex},
		{"priority", ServiceTierPriority},
		// OpenAI renamed Priority processing to Fast mode and accepts and
		// reports either name for the same tier.
		{"fast", ServiceTierPriority},
		{" FAST ", ServiceTierPriority},
		{"batch", ServiceTierBatch},
		{"scale", ServiceTierScale},
		{"reserved", ServiceTierReserved},
		{"provisioned-throughput", ServiceTierProvisionedThroughput},
		{"Some-New-Tier", ServiceTier("some_new_tier")},
	} {
		t.Run(tt.raw, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, NormalizeServiceTier(tt.raw))
		})
	}
}

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

package reconcile

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseSnapshot(t *testing.T) {
	t.Parallel()

	snap, err := ParseSnapshot([]byte(`{"facts": {"anthropic/claude-sonnet-5-5": {"released": "2026-09-30"}},
		"providers": [{"provider": "aws.bedrock", "offerings": [{"id": "global.anthropic.claude-sonnet-5-5",
			"pricing": {"default": {"base": {"input_microcents_per_million": 9007199254740993}}}}]}]}`))
	require.NoError(t, err)

	t.Run("finds an offering by package and ID", func(t *testing.T) {
		t.Parallel()

		offering, ok := snap.Offering("bedrock", "global.anthropic.claude-sonnet-5-5")
		require.True(t, ok)
		assert.Equal(t, "global.anthropic.claude-sonnet-5-5", offering["id"])

		_, ok = snap.Offering("vertex", "global.anthropic.claude-sonnet-5-5")
		assert.False(t, ok, "another package's offerings")
	})

	t.Run("keeps numbers as written", func(t *testing.T) {
		t.Parallel()

		// 2^53+1 has no exact float64, so only a json.Number keeps it.
		offering, _ := snap.Offering("bedrock", "global.anthropic.claude-sonnet-5-5")
		raw, err := json.Marshal(offering["pricing"])
		require.NoError(t, err)
		assert.Contains(t, string(raw), "9007199254740993")
	})

	t.Run("rejects a malformed snapshot", func(t *testing.T) {
		t.Parallel()

		_, err := ParseSnapshot([]byte(`{"providers": [`))
		assert.Error(t, err)
	})
}

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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFireworksOfferingPattern(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		want bool
	}{
		{"fireworks:accounts/fireworks/models/gpt-oss-120b", true},
		{"fireworks:accounts/fireworks/models/deepseek-v4p1-flash-1017", true},
		{"fireworks:accounts/fireworks/routers/glm-5p3-fast", true},
		{"accounts/fireworks/models/gpt-oss-120b", false},
		{"openai:accounts/fireworks/models/gpt-oss-120b", false},
		{"fireworks:accounts/private/models/model", false},
		{"fireworks:accounts/fireworks/deployments/model", false},
		{"fireworks:accounts/fireworks/models/../model", false},
		{"fireworks:accounts/fireworks/models/model%2fother", false},
		{"fireworks:accounts/fireworks/models/model`markup", false},
		{"fireworks:accounts/fireworks/models/model\n", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, OfferingPattern.MatchString(tt.name))
		})
	}
}

func TestParseReport(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		wantErr bool
	}{
		{"all three arrays", `{"changes": [], "needs_human": [], "skipped": []}`, false},
		{"truncated json", `{"changes": [`, true},
		{"unknown field", `{"changes": [], "needs_human": [], "skipped": [], "extra": 1}`, true},
		{"null", `null`, true},
		{"empty object", `{}`, true},
		{"missing changes", `{"needs_human": [], "skipped": []}`, true},
		{"missing needs_human", `{"changes": [], "skipped": []}`, true},
		{"missing skipped", `{"changes": [], "needs_human": []}`, true},
		{"null array", `{"changes": null, "needs_human": [], "skipped": []}`, true},
		{"a question with both sides", `{"changes": [], "needs_human": [{"offering": "x", "catalogue": "65535", "page": "65536"}], "skipped": []}`, false},
		{"a skipped row with values", `{"changes": [], "needs_human": [], "skipped": [{"offering": "x", "reason": "no_catalog_field", "rule": "no-catalog-field", "values": ["12h cache write"]}]}`, false},
		{"both sides only on questions", `{"changes": [{"offering": "x", "catalogue": "65535"}], "needs_human": [], "skipped": []}`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			_, err := ParseReport([]byte(tt.input))
			if tt.wantErr {
				require.Error(t, err)
				return
			}

			require.NoError(t, err)
		})
	}
}

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
)

// TestClaudeStructuredOutputCapability pins which Claude families advertise
// structured output; every variant of a family must agree.
func TestClaudeStructuredOutputCapability(t *testing.T) {
	t.Parallel()

	want := map[string]bool{
		ModelClaudeFable51:  false,
		ModelClaudeFable5:   false,
		ModelClaudeOpus55:   false,
		ModelClaudeOpus5:    false,
		ModelClaudeOpus48:   false,
		ModelClaudeOpus47:   false,
		ModelClaudeOpus46:   true,
		ModelClaudeOpus45:   true,
		ModelClaudeSonnet55: false,
		ModelClaudeSonnet5:  false,
		ModelClaudeSonnet46: true,
		ModelClaudeSonnet45: true,
		ModelClaudeHaiku55:  false,
		ModelClaudeHaiku45:  true,
	}

	seen := make(map[string]bool, len(want))

	for _, o := range Catalog().All() {
		bare := o.ID
		if hasRegionPrefix(bare) {
			_, bare, _ = strings.Cut(bare, ".")
		}

		if !strings.HasPrefix(bare, "anthropic.") {
			continue
		}

		wantCap, ok := want[bare]
		if !assert.Truef(t, ok, "Claude family %s has no expected structured-output capability", bare) {
			continue
		}

		seen[bare] = true

		assert.Equalf(t, wantCap, o.Capabilities.StructuredOutput, "%s structured output", o.ID)
	}

	for bare := range want {
		assert.Truef(t, seen[bare], "no catalog variant found for %s", bare)
	}
}

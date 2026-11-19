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

package mcp

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/internal/toolname"
)

// A replayed invalid name must never equal the shortened name of a real MCP
// tool, or the history would claim a call to a tool that was never invoked.
func TestReplayNeverEqualsMangledName(t *testing.T) {
	t.Parallel()

	names := map[string]string{
		"65 chars":  strings.Repeat("a", 30) + "__" + strings.Repeat("b", 33),
		"200 chars": strings.Repeat("a", 100) + "__" + strings.Repeat("b", 98),
	}

	for label, n := range names {
		t.Run(label, func(t *testing.T) {
			t.Parallel()

			assert.NotEqual(t, mangleHeadIfTooLong(n, maxToolNameLen), toolname.Replay(n))
		})
	}
}

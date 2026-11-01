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
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
)

// TestBedrockClaudeHasNoToolSearch fails a Claude offering on Bedrock with
// tool search on. Anthropic's tool-search page serves it on Bedrock through
// InvokeModel only, not the Converse API this SDK calls, and the Bedrock
// provider sends no tool search: with the capability on, an agent would send
// every deferred tool's schema on every call. It lives here, outside the
// packages a reconcile run may edit (RULES.md, owned-test-update).
func TestBedrockClaudeHasNoToolSearch(t *testing.T) {
	t.Parallel()

	claude := 0

	for _, o := range bedrock.Catalog().All() {
		if o.Facts().Publisher != catalog.PublisherAnthropic {
			continue
		}

		claude++

		assert.Emptyf(t, o.Capabilities.ToolSearchProtocol,
			"%s: tool search is on, but Bedrock doesn't serve it through Converse", o.ID)
	}

	// Pin the premise, or the guard passes vacuously once no Claude copy is left.
	require.NotZero(t, claude, "bedrock must still carry Claude offerings for this guard to mean anything")
}

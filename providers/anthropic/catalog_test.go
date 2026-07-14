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

package anthropic

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestModelCatalogAcceptsOnlyDocumentedAnthropicIDShapes(t *testing.T) {
	t.Parallel()

	provider := &Provider{}
	for _, model := range []string{
		ModelClaudeOpus48,
		ModelClaudeSonnet45,
		"claude-sonnet-4-5-20250929",
		"claude-haiku-4-5-20251001",
		"claude-opus-4-1-20250805",
	} {
		_, ok := provider.ModelCatalog(model)
		require.True(t, ok, model)
	}

	for _, model := range []string{
		"claude-opus-4-8-custom",
		"claude-opus-4-8-20260528",
		"claude-sonnet-4-6-preview",
		"claude-sonnet-4-5-custom",
		"claude-sonnet-4-5-20250230",
	} {
		_, ok := provider.ModelCatalog(model)
		require.False(t, ok, model)
	}
}

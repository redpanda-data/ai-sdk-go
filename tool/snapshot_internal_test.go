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

package tool

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

type metadataProbe struct{}

func (metadataProbe) Definition() llm.ToolDefinition { return llm.ToolDefinition{Name: "metadata"} }

func (metadataProbe) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return nil, nil
}

func TestSnapshotCopiesRegistrationConfig(t *testing.T) {
	t.Parallel()

	policy := map[string]any{"scope": "original"}
	items := []any{map[string]any{"scope": "original"}}
	metadata := map[string]any{"version": "original", "policy": policy, "items": items}
	source := NewRegistry(RegistryConfig{})
	require.NoError(t, source.Register(metadataProbe{}, WithTimeout(time.Second), WithMetadata(metadata)))
	captured, err := Snapshot(t.Context(), source)
	require.NoError(t, err)

	native, ok := captured.(*registry)
	require.True(t, ok)

	const changed = "changed"

	metadata["version"] = changed
	policy["scope"] = changed
	items[0] = changed

	assert.Equal(t, map[string]any{
		"version": "original", "policy": map[string]any{"scope": "original"},
		"items": []any{map[string]any{"scope": "original"}},
	}, native.tools["metadata"].config.Metadata)
	assert.Equal(t, time.Second, native.tools["metadata"].config.Timeout)
}

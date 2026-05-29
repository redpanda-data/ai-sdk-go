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

package schemamap_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/internal/schemamap"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestToMap(t *testing.T) {
	t.Parallel()

	t.Run("nil schema yields nil map", func(t *testing.T) {
		t.Parallel()

		m, err := schemamap.ToMap(nil)
		require.NoError(t, err)
		assert.Nil(t, m)
	})

	// Regression: an empty jsonschema.Schema marshals to the JSON literal `true`,
	// not `{}`. A naive json.Unmarshal of `true` into map[string]any fails, which
	// previously broke every provider request mapper for no-argument tools.
	t.Run("empty schema yields nil map, not an error", func(t *testing.T) {
		t.Parallel()

		m, err := schemamap.ToMap(&llm.Schema{})
		require.NoError(t, err)
		assert.Nil(t, m)
	})

	// A schema parsed from `null` marshals to the boolean `false`; same hazard.
	t.Run("null/false schema yields nil map", func(t *testing.T) {
		t.Parallel()

		m, err := schemamap.ToMap(llm.MustSchema(`null`))
		require.NoError(t, err)
		assert.Nil(t, m)
	})

	t.Run("object schema decodes to a map", func(t *testing.T) {
		t.Parallel()

		m, err := schemamap.ToMap(&llm.Schema{
			Type:     "object",
			Required: []string{"q"},
			Properties: map[string]*llm.Schema{
				"q": {Type: "string"},
			},
		})
		require.NoError(t, err)
		assert.Equal(t, "object", m["type"])
		assert.Equal(t, []any{"q"}, m["required"])
		assert.Contains(t, m, "properties")
	})

	// A schema that cannot marshal must surface the error so the provider fails
	// loudly rather than silently sending a malformed request.
	t.Run("unmarshalable schema returns an error", func(t *testing.T) {
		t.Parallel()

		// Setting both Type and Types is rejected by Schema.MarshalJSON.
		_, err := schemamap.ToMap(&llm.Schema{Type: "object", Types: []string{"object"}})
		require.Error(t, err)
	})
}

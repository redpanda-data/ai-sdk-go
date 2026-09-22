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

package catalog_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func TestMustDeclarePublisherDeclaresOnEveryEntry(t *testing.T) {
	t.Parallel()

	got := catalog.MustDeclarePublisher("anthropic", []catalog.Entry{
		{ID: "claude-opus-5"},
		{ID: "claude-haiku-4-5", Attributes: map[string]string{"inference_geo": "us"}},
	})

	require.Len(t, got, 2)
	assert.Equal(t, "anthropic", got[0].Publisher)
	assert.Equal(t, "anthropic", got[1].Publisher)
	assert.Equal(t, "us", got[1].Attributes["inference_geo"], "existing attributes must survive")
}

func TestMustDeclarePublisherLeavesTheArgumentAlone(t *testing.T) {
	t.Parallel()

	entries := []catalog.Entry{{ID: "claude-opus-5"}}

	got := catalog.MustDeclarePublisher("anthropic", entries)

	assert.Equal(t, "anthropic", got[0].Publisher)
	assert.Empty(t, entries[0].Publisher, "the caller's slice must be untouched")
}

func TestMustDeclarePublisherGuards(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		publisher string
		entries   []catalog.Entry
		wantPanic string
	}{
		{
			name:      "conflicting declaration",
			publisher: "anthropic",
			entries:   []catalog.Entry{{ID: "gpt-5.6-sol", Publisher: "openai"}},
			wantPanic: "catalog: entry gpt-5.6-sol declares publisher openai, not anthropic",
		},
		{
			name:      "empty publisher",
			publisher: "",
			entries:   []catalog.Entry{{ID: "claude-opus-5"}},
			wantPanic: "catalog: MustDeclarePublisher needs a publisher",
		},
		{
			name:      "matching redeclaration is accepted",
			publisher: "anthropic",
			entries:   []catalog.Entry{{ID: "claude-opus-5", Publisher: "anthropic"}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			var got []catalog.Entry

			call := func() { got = catalog.MustDeclarePublisher(tt.publisher, tt.entries) }

			if tt.wantPanic == "" {
				require.NotPanics(t, call)

				for _, e := range got {
					assert.Equalf(t, tt.publisher, e.Publisher, "%s publisher", e.ID)
				}

				return
			}

			assert.PanicsWithError(t, tt.wantPanic, call)
		})
	}
}

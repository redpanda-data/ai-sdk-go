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

// TestMustDeclarePublisherDeclaresOnEveryEntry covers the two shapes a
// single-vendor catalog presents: entries with no attribute map at all
// (the common case) and entries that already carry provider attributes,
// which must keep them.
func TestMustDeclarePublisherDeclaresOnEveryEntry(t *testing.T) {
	t.Parallel()

	got := catalog.MustDeclarePublisher("anthropic", []catalog.Entry{
		{ID: "claude-opus-5"},
		{ID: "claude-haiku-4-5", Attributes: map[string]string{"inference_geo": "us"}},
	})

	require.Len(t, got, 2)
	assert.Equal(t, "anthropic", got[0].Attributes[catalog.AttributePublisher])
	assert.Equal(t, "anthropic", got[1].Attributes[catalog.AttributePublisher])
	assert.Equal(t, "us", got[1].Attributes["inference_geo"], "existing attributes must survive")
}

// TestMustDeclarePublisherGuards covers the authoring errors the helper
// refuses. Overwriting an authored publisher would hand a consumer the
// wrong brand mark, and writing an empty one would defer the failure to
// TestEveryOfferingDeclaresAPublisher two packages away; both fail at the
// authoring site instead. Re-declaring the same publisher is harmless, so
// the guard stays narrow.
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
			entries: []catalog.Entry{
				{ID: "gpt-5.6-sol", Attributes: map[string]string{catalog.AttributePublisher: "openai"}},
			},
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
			entries: []catalog.Entry{
				{ID: "claude-opus-5", Attributes: map[string]string{catalog.AttributePublisher: "anthropic"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			call := func() { catalog.MustDeclarePublisher(tt.publisher, tt.entries) }

			if tt.wantPanic == "" {
				require.NotPanics(t, call)

				// entries is mutated in place, so the declaration is
				// observable on the input slice.
				for _, e := range tt.entries {
					assert.Equalf(t, tt.publisher, e.Attributes[catalog.AttributePublisher], "%s publisher", e.ID)
				}

				return
			}

			assert.PanicsWithError(t, tt.wantPanic, call)
		})
	}
}

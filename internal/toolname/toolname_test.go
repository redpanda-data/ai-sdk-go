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

package toolname

import (
	"regexp"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	replayShape = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9_-]{0,63}$`)
	validShape  = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)
)

func name65() string { return strings.Repeat("a", 30) + "__" + strings.Repeat("b", 33) }

func TestValid(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		in   string
		want bool
	}{
		{"simple", "inventory__get_item", true},
		{"hyphen and digits", "get-item_2", true},
		{"empty", "", false},
		{"inner space", "a b", false},
		{"trailing space", "name ", false},
		{"dot", "functions.foo", false},
		{"non-ascii", "naïve", false},
		{"64 chars", strings.Repeat("a", 64), true},
		{"65 chars", name65(), false},
		{"newline", "a\n", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			assert.Equal(t, tt.want, Valid(tt.in))
			assert.Equal(t, tt.want, validShape.MatchString(tt.in))
		})
	}
}

func TestReplay(t *testing.T) {
	t.Parallel()

	inputs := []struct {
		name string
		in   string
	}{
		{"empty", ""},
		{"inner space", "a b"},
		{"trailing space", "name "},
		{"dot", "functions.foo"},
		{"65 chars all valid", name65()},
		{"200 chars", strings.Repeat("x", 200)},
		{"non-ascii", "naïve_café_ж"},
	}

	seen := make(map[string]string)

	for _, tt := range inputs {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := Replay(tt.in)
			assert.Regexp(t, replayShape, got)
			assert.True(t, Valid(got))
			assert.Equal(t, got, Replay(tt.in), "deterministic")
		})

		got := Replay(tt.in)
		if prev, dup := seen[got]; dup {
			t.Errorf("inputs %q and %q both map to %q", prev, tt.name, got)
		}

		seen[got] = tt.name
	}

	assert.NotEqual(t, "a_b", Replay("a b"), "a cleaned name must not collide with a real tool name")
	assert.NotEqual(t, Replay("a b"), Replay("a_b "))
}

func TestReplayAvoiding_Valid(t *testing.T) {
	t.Parallel()

	taken := map[string]struct{}{"inventory__get_item": {}}
	assert.Equal(t, "inventory__get_item", ReplayAvoiding("inventory__get_item", taken))
	assert.Equal(t, "inventory__get_item", ReplayAvoiding("inventory__get_item", nil))
}

func TestReplayAvoiding_RegisteredCollision(t *testing.T) {
	t.Parallel()

	for _, n := range []string{"inventory__get_item ", name65(), "a b", ""} {
		t.Run(n, func(t *testing.T) {
			t.Parallel()

			taken := map[string]struct{}{Replay(n): {}}
			got := ReplayAvoiding(n, taken)

			assert.NotEqual(t, Replay(n), got)
			assert.Regexp(t, replayShape, got)
			assert.True(t, Valid(got))
			assert.Equal(t, got, ReplayAvoiding(n, taken), "deterministic")
			assert.NotContains(t, taken, got)

			// Without a collision the plain replay name is used.
			assert.Equal(t, Replay(n), ReplayAvoiding(n, nil))
		})
	}
}

func TestReplayAvoiding_FallbackIsLong(t *testing.T) {
	t.Parallel()

	n := "inventory__get_item "
	got := ReplayAvoiding(n, map[string]struct{}{Replay(n): {}})

	require.Len(t, got, len("invalid_")+56)
	assert.True(t, strings.HasPrefix(got, "invalid_"))
}

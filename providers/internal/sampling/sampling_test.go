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

package sampling_test

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/internal/sampling"
)

func TestCoalesce(t *testing.T) {
	t.Parallel()

	a, b := 1, 2

	assert.Equal(t, &a, sampling.Coalesce(&a, &b), "override returned when non-nil")
	assert.Equal(t, &b, sampling.Coalesce[int](nil, &b), "fallback returned when override nil")
	assert.Nil(t, sampling.Coalesce[int](nil, nil), "both nil yields nil")
}

func TestCoalesceSlice(t *testing.T) {
	t.Parallel()

	override := []string{"a"}
	fallback := []string{"b"}

	assert.Equal(t, override, sampling.CoalesceSlice(override, fallback))
	assert.Equal(t, fallback, sampling.CoalesceSlice[string](nil, fallback))
	assert.Equal(t, fallback, sampling.CoalesceSlice([]string{}, fallback), "empty override falls through")
}

func TestValidateMaxOutputTokens(t *testing.T) {
	t.Parallel()

	v := func(n int) *int { return &n }

	cases := []struct {
		name     string
		resolved *int
		limit    int
		wantErr  bool
	}{
		{"nil resolved", nil, 100, false},
		{"no constraint", v(99999), 0, false},
		{"in range", v(500), 1000, false},
		{"at boundary", v(1000), 1000, false},
		{"over limit", v(1001), 1000, true},
		{"zero rejected", v(0), 1000, true},
		{"negative rejected", v(-1), 1000, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := sampling.ValidateMaxOutputTokens(tc.resolved, tc.limit)
			if tc.wantErr {
				require.Error(t, err)
				assert.ErrorIs(t, err, llm.ErrInvalidInput)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestValidateTemperature(t *testing.T) {
	t.Parallel()

	v := func(f float64) *float64 { return &f }

	cases := []struct {
		name     string
		resolved *float64
		rng      [2]float64
		wantErr  bool
	}{
		{"nil resolved", nil, [2]float64{0, 1}, false},
		{"zero range no-op", v(99), [2]float64{0, 0}, false},
		{"in range", v(0.5), [2]float64{0, 1}, false},
		{"min boundary", v(0), [2]float64{0, 2}, false},
		{"max boundary", v(2), [2]float64{0, 2}, false},
		{"below min", v(-0.1), [2]float64{0, 1}, true},
		{"above max", v(2.1), [2]float64{0, 2}, true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			err := sampling.ValidateTemperature(tc.resolved, tc.rng)
			if tc.wantErr {
				require.Error(t, err)
				assert.ErrorIs(t, err, llm.ErrInvalidInput)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

func TestRejectUnsupported(t *testing.T) {
	t.Parallel()

	t.Run("nil override no error", func(t *testing.T) {
		t.Parallel()

		assert.NoError(t, sampling.RejectUnsupported[int64]("seed", nil, "openai"))
	})

	t.Run("set override errors", func(t *testing.T) {
		t.Parallel()

		seed := int64(42)

		err := sampling.RejectUnsupported("seed", &seed, "openai")
		require.Error(t, err)
		assert.True(t, errors.Is(err, llm.ErrInvalidInput))
		assert.Contains(t, err.Error(), "seed")
		assert.Contains(t, err.Error(), "openai")
	})
}

func TestRejectUnsupportedSlice(t *testing.T) {
	t.Parallel()

	assert.NoError(t, sampling.RejectUnsupportedSlice[string]("stop", nil, "openai"))
	assert.NoError(t, sampling.RejectUnsupportedSlice[string]("stop", []string{}, "openai"))

	err := sampling.RejectUnsupportedSlice("stop", []string{"end"}, "openai")
	require.Error(t, err)
	assert.True(t, errors.Is(err, llm.ErrInvalidInput))
}

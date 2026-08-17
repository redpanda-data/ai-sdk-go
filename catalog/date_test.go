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

package catalog

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDateParse(t *testing.T) {
	t.Parallel()

	d, err := ParseDate("2026-06-15")
	require.NoError(t, err)
	assert.Equal(t, Date{Year: 2026, Month: time.June, Day: 15}, d)
	assert.Equal(t, "2026-06-15", d.String())

	for _, invalid := range []string{"2026-02-31", "2026-13-01", "June 2026", "2026-6-1", ""} {
		_, err := ParseDate(invalid)
		require.Error(t, err, invalid)
	}

	assert.Panics(t, func() { MustDate("not-a-date") })
}

func TestDateOrdering(t *testing.T) {
	t.Parallel()

	a := MustDate("2026-06-14")
	b := MustDate("2026-06-15")

	assert.True(t, a.Before(b))
	assert.False(t, b.Before(a))
	assert.True(t, b.After(a))
	assert.False(t, a.Before(a))
	assert.True(t, MustDate("2025-12-31").Before(MustDate("2026-01-01")))
	assert.True(t, MustDate("2026-05-31").Before(MustDate("2026-06-01")))
}

func TestDateZeroValue(t *testing.T) {
	t.Parallel()

	var zero Date
	assert.True(t, zero.IsZero())
	assert.Empty(t, zero.String())
	assert.False(t, MustDate("2026-01-01").IsZero())
}

func TestDateJSONRoundTrip(t *testing.T) {
	t.Parallel()

	type payload struct {
		When Date `json:"when"`
		Zero Date `json:"zero,omitzero"`
	}

	raw, err := json.Marshal(payload{When: MustDate("2026-06-15")})
	require.NoError(t, err)
	assert.JSONEq(t, `{"when":"2026-06-15"}`, string(raw))

	var back payload
	require.NoError(t, json.Unmarshal(raw, &back))
	assert.Equal(t, MustDate("2026-06-15"), back.When)
	assert.True(t, back.Zero.IsZero())

	var fromEmpty payload
	require.NoError(t, json.Unmarshal([]byte(`{"when":"2026-06-15","zero":""}`), &fromEmpty))
	assert.True(t, fromEmpty.Zero.IsZero())

	var bad payload
	assert.Error(t, json.Unmarshal([]byte(`{"when":"2026-02-31"}`), &bad))
}

func TestToday(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()
	d := Today()
	// Tolerate a midnight rollover between the two calls.
	ok := d == Date{Year: now.Year(), Month: now.Month(), Day: now.Day()}
	next := time.Now().UTC()
	ok = ok || d == Date{Year: next.Year(), Month: next.Month(), Day: next.Day()}
	assert.True(t, ok)
}

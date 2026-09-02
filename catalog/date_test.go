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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDateParse(t *testing.T) {
	t.Parallel()

	d, err := ParseDate("2026-06-15")
	require.NoError(t, err)
	assert.True(t, d.Equal(time.Date(2026, time.June, 15, 0, 0, 0, 0, time.UTC)))
	assert.Equal(t, "2026-06-15", dateString(d))

	for _, invalid := range []string{"2026-02-31", "2026-13-01", "June 2026", "2026-6-1", ""} {
		_, err := ParseDate(invalid)
		require.Error(t, err, invalid)
	}

	assert.Panics(t, func() { MustDate("not-a-date") })
}

func TestDateOnly(t *testing.T) {
	t.Parallel()

	assert.True(t, isDateOnly(time.Time{}), "zero means unset and is legal")
	assert.True(t, isDateOnly(MustDate("2026-06-15")))
	// Same instant expressed in another zone is still midnight UTC.
	assert.True(t, isDateOnly(MustDate("2026-06-15").In(time.FixedZone("CEST", 2*3600))))

	assert.False(t, isDateOnly(time.Date(2026, time.June, 15, 9, 30, 0, 0, time.UTC)), "intra-day precision")
	assert.False(t, isDateOnly(time.Date(2026, time.June, 15, 0, 0, 0, 0, time.FixedZone("CEST", 2*3600))), "local midnight is not UTC midnight")

	assert.Empty(t, dateString(time.Time{}))
}

func TestToday(t *testing.T) {
	t.Parallel()

	d := Today()
	assert.True(t, isDateOnly(d))
	// Tolerate a midnight rollover between the two calls.
	now := time.Now().UTC()
	assert.True(t, d.Equal(now.Truncate(24*time.Hour)) || d.Equal(now.Add(-time.Minute).Truncate(24*time.Hour)))
}

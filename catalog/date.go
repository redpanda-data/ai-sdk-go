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
	"fmt"
	"time"
)

// Lifecycle and fact dates are time.Time values pinned to midnight UTC.
// Providers publish lifecycle as calendar dates ("retired June 15,
// 2026"), and classification must not shift with the caller's clock or
// timezone — so dates carry no finer precision, and New rejects values
// that do. Construct with MustDate or ParseDate; the zero time.Time
// means "not set".

// ParseDate parses a "YYYY-MM-DD" string into a date-only time.Time
// (midnight UTC), rejecting non-calendar dates such as 2026-02-31.
func ParseDate(s string) (time.Time, error) {
	t, err := time.ParseInLocation(time.DateOnly, s, time.UTC)
	if err != nil {
		return time.Time{}, fmt.Errorf("catalog: invalid date %q: %w", s, err)
	}

	return t, nil
}

// MustDate is ParseDate that panics on malformed input. It is intended
// for authoring catalog literals, where the string is a compile-time
// constant and every entry is exercised by tests.
func MustDate(s string) time.Time {
	d, err := ParseDate(s)
	if err != nil {
		panic(err) //nolint:forbidigo // authoring error, not runtime
	}

	return d
}

// Today returns the current date in UTC, truncated to midnight.
func Today() time.Time {
	return time.Now().UTC().Truncate(24 * time.Hour)
}

// isDateOnly reports whether t is unset or exactly midnight UTC.
// Truncate rounds on the absolute timeline, so t's location cannot skew
// the check.
func isDateOnly(t time.Time) bool {
	return t.IsZero() || t.Equal(t.Truncate(24*time.Hour))
}

// dateString renders a date as "YYYY-MM-DD", or "" when unset — the
// snapshot and error-message form.
func dateString(t time.Time) string {
	if t.IsZero() {
		return ""
	}

	return t.UTC().Format(time.DateOnly)
}

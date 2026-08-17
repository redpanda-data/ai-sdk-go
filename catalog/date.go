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

// Date is a UTC calendar date with day granularity. The zero value means
// "unknown / not set" and is a valid state everywhere a Date appears in
// this package: an unset Retires means no retirement is announced, an
// unset Knowledge means the provider does not publish a training cutoff.
//
// Dates rather than timestamps are deliberate: every provider publishes
// model lifecycle as calendar dates ("retired June 15, 2026"), and
// classification must not flip within a day depending on the caller's
// clock or timezone.
//
//nolint:recvcheck // MarshalText is value, UnmarshalText is pointer — the standard encoding pair, like time.Time.
type Date struct {
	Year  int
	Month time.Month
	Day   int
}

// NewDate constructs a Date. It does not validate calendar correctness;
// use ParseDate or MustDate for validated construction from a string.
func NewDate(year int, month time.Month, day int) Date {
	return Date{Year: year, Month: month, Day: day}
}

// ParseDate parses a "YYYY-MM-DD" string into a Date, rejecting
// non-calendar dates such as 2026-02-31.
func ParseDate(s string) (Date, error) {
	t, err := time.ParseInLocation(time.DateOnly, s, time.UTC)
	if err != nil {
		return Date{}, fmt.Errorf("catalog: invalid date %q: %w", s, err)
	}

	return Date{Year: t.Year(), Month: t.Month(), Day: t.Day()}, nil
}

// MustDate is ParseDate that panics on malformed input. It is intended
// for authoring catalog literals, where the string is a compile-time
// constant and every entry is exercised by tests.
func MustDate(s string) Date {
	d, err := ParseDate(s)
	if err != nil {
		panic(err) //nolint:forbidigo // authoring error, not runtime
	}

	return d
}

// Today returns the current date in UTC.
func Today() Date {
	now := time.Now().UTC()
	return Date{Year: now.Year(), Month: now.Month(), Day: now.Day()}
}

// IsZero reports whether the date is unset.
func (d Date) IsZero() bool {
	return d == Date{}
}

// Before reports whether d is strictly earlier than other.
func (d Date) Before(other Date) bool {
	if d.Year != other.Year {
		return d.Year < other.Year
	}

	if d.Month != other.Month {
		return d.Month < other.Month
	}

	return d.Day < other.Day
}

// After reports whether d is strictly later than other.
func (d Date) After(other Date) bool {
	return other.Before(d)
}

// String renders the date as "YYYY-MM-DD", or "" for the zero value.
func (d Date) String() string {
	if d.IsZero() {
		return ""
	}

	return fmt.Sprintf("%04d-%02d-%02d", d.Year, int(d.Month), d.Day)
}

// MarshalText encodes the date as "YYYY-MM-DD"; the zero value encodes
// as an empty string. Implementing encoding.TextMarshaler makes Date
// work with encoding/json and any other text-based encoder.
func (d Date) MarshalText() ([]byte, error) {
	return []byte(d.String()), nil
}

// UnmarshalText decodes "YYYY-MM-DD"; an empty input yields the zero
// value.
func (d *Date) UnmarshalText(text []byte) error {
	if len(text) == 0 {
		*d = Date{}
		return nil
	}

	parsed, err := ParseDate(string(text))
	if err != nil {
		return err
	}

	*d = parsed

	return nil
}

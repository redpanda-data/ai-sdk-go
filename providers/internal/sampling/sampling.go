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

// Package sampling collects small, provider-agnostic helpers used by
// every provider's request mapper when merging per-request
// llm.SamplingParams over the model's Config defaults and validating
// the resulting values against per-model constraints.
//
// The helpers deliberately do not encode which knobs a particular
// provider supports. That decision lives in each provider's mapper so
// the knob-by-knob support matrix stays visible at the source of the
// API call.
package sampling

import (
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Coalesce returns override if non-nil, otherwise fallback. The generic
// variant subsumes the per-type llm.CoalesceFloat64/CoalesceInt helpers
// for use inside request mappers.
func Coalesce[T any](override, fallback *T) *T {
	if override != nil {
		return override
	}

	return fallback
}

// CoalesceSlice returns override if non-empty, otherwise fallback. A
// nil or zero-length override is treated as "use fallback" so callers
// can pass through SamplingParams.StopSequences unconditionally.
func CoalesceSlice[T any](override, fallback []T) []T {
	if len(override) > 0 {
		return override
	}

	return fallback
}

// ValidateMaxOutputTokens checks the resolved MaxOutputTokens value
// against the model's MaxOutputTokens constraint. resolved == nil and
// limit == 0 (no constraint declared) both no-op. Negative or zero
// resolved values are rejected with llm.ErrInvalidInput.
func ValidateMaxOutputTokens(resolved *int, limit int) error {
	if resolved == nil {
		return nil
	}

	v := *resolved
	if v <= 0 {
		return fmt.Errorf("%w: max_output_tokens %d must be positive", llm.ErrInvalidInput, v)
	}

	if limit > 0 && v > limit {
		return fmt.Errorf("%w: max_output_tokens %d exceeds model limit %d", llm.ErrInvalidInput, v, limit)
	}

	return nil
}

// ValidateTemperature checks the resolved Temperature value against the
// model's [min, max] constraint range. resolved == nil and a zero range
// (constraints not declared) both no-op.
func ValidateTemperature(resolved *float64, rng [2]float64) error {
	if resolved == nil {
		return nil
	}

	minT, maxT := rng[0], rng[1]
	if minT == 0 && maxT == 0 {
		return nil
	}

	v := *resolved
	if v < minT || v > maxT {
		return fmt.Errorf("%w: temperature %f out of range [%f, %f]", llm.ErrInvalidInput, v, minT, maxT)
	}

	return nil
}

// RejectUnsupported returns an error when override is set but the
// resolved provider does not support the named knob. Use it in request
// mappers to surface user-visible errors instead of silently dropping
// per-request overrides. The error wraps llm.ErrInvalidInput so
// callers can errors.Is against it.
//
// override is a *T (e.g. *int64, *float64) so the helper recognises
// "unset" via nil. A non-pointer override should be tested with the
// natural zero-value check at the call site instead.
func RejectUnsupported[T any](knob string, override *T, providerID llm.ProviderID) error {
	if override == nil {
		return nil
	}

	return fmt.Errorf("%w: %s sampling parameter is not supported by provider %q", llm.ErrInvalidInput, knob, providerID)
}

// RejectUnsupportedSlice mirrors RejectUnsupported for slice-typed
// knobs (StopSequences). A zero-length slice is treated as unset.
func RejectUnsupportedSlice[T any](knob string, override []T, providerID llm.ProviderID) error {
	if len(override) == 0 {
		return nil
	}

	return fmt.Errorf("%w: %s sampling parameter is not supported by provider %q", llm.ErrInvalidInput, knob, providerID)
}

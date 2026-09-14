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

package otel

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

type customError struct{}

func (customError) Error() string { return "custom" }

func TestSpanErrorType(t *testing.T) {
	t.Parallel()

	perr := &llm.ProviderError{Base: llm.ErrAPICall, Code: "guardrail_intervened", Message: "blocked"}

	tests := []struct {
		name   string
		err    error
		want   string
		wantOK bool
	}{
		{"provider error contributes its code", perr, "guardrail_intervened", true},
		{"provider code survives wrapping", fmt.Errorf("agent: model generation failed: %w", llm.WrapAPICall(perr)), "guardrail_intervened", true},
		{"provider error without a code falls through to its type", &llm.ProviderError{Base: llm.ErrServerError, Message: "x"}, "*llm.ProviderError", true},
		{"fmt wrap of a plain error names nothing", fmt.Errorf("ctx: %w", errors.New("boom")), "", false},
		{"multi-wrap names nothing", fmt.Errorf("%w: %w", llm.ErrAPICall, errors.New("boom")), "", false},
		{"errors.New names nothing", errors.New("boom"), "", false},
		{"errors.Join names nothing", errors.Join(errors.New("a"), errors.New("b")), "", false},
		{"concrete custom type keeps its name", customError{}, "otel.customError", true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got, ok := spanErrorType(tc.err)
			assert.Equal(t, tc.wantOK, ok)
			assert.Equal(t, tc.want, got)
		})
	}
}

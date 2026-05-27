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

package llm_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestCoalesceFloat64(t *testing.T) {
	t.Parallel()

	override := 0.9
	fallback := 0.5

	tests := []struct {
		name             string
		override         *float64
		fallback         *float64
		wantPtrValue     *float64
		wantNilFromFn    bool
		wantUseOverride  bool
		wantUseFallback  bool
		wantBothNilIsNil bool
	}{
		{name: "both nil", override: nil, fallback: nil, wantBothNilIsNil: true},
		{name: "override wins", override: &override, fallback: &fallback, wantUseOverride: true},
		{name: "fallback when override nil", override: nil, fallback: &fallback, wantUseFallback: true},
		{name: "override when fallback nil", override: &override, fallback: nil, wantUseOverride: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := llm.CoalesceFloat64(tc.override, tc.fallback)
			switch {
			case tc.wantBothNilIsNil:
				assert.Nil(t, got)
			case tc.wantUseOverride:
				assert.Equal(t, tc.override, got)
			case tc.wantUseFallback:
				assert.Equal(t, tc.fallback, got)
			}
		})
	}
}

func TestCoalesceInt(t *testing.T) {
	t.Parallel()

	override := 100
	fallback := 50

	assert.Nil(t, llm.CoalesceInt(nil, nil))
	assert.Equal(t, &override, llm.CoalesceInt(&override, &fallback))
	assert.Equal(t, &fallback, llm.CoalesceInt(nil, &fallback))
}

func TestCoalesceInt64(t *testing.T) {
	t.Parallel()

	override := int64(100)
	fallback := int64(50)

	assert.Nil(t, llm.CoalesceInt64(nil, nil))
	assert.Equal(t, &override, llm.CoalesceInt64(&override, &fallback))
	assert.Equal(t, &fallback, llm.CoalesceInt64(nil, &fallback))
}

func TestCoalesceStrings(t *testing.T) {
	t.Parallel()

	override := []string{"a"}
	fallback := []string{"b"}

	assert.Nil(t, llm.CoalesceStrings(nil, nil))
	assert.Equal(t, override, llm.CoalesceStrings(override, fallback))
	assert.Equal(t, fallback, llm.CoalesceStrings(nil, fallback))
	// Empty override falls through to fallback.
	assert.Equal(t, fallback, llm.CoalesceStrings([]string{}, fallback))
}

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

package llm

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestValidateTopP(t *testing.T) {
	t.Parallel()

	require.NoError(t, (&ModelConstraints{}).ValidateTopP(0))
	require.NoError(t, (&ModelConstraints{}).ValidateTopP(1))

	inclusive := &ModelConstraints{TopPRange: [2]float64{0.25, 0.75}}
	require.NoError(t, inclusive.ValidateTopP(0.25))
	require.NoError(t, inclusive.ValidateTopP(0.75))
	require.Error(t, inclusive.ValidateTopP(0.751))

	narrow := &ModelConstraints{
		TopPRange:        [2]float64{0.99, 1},
		TopPMaxExclusive: true,
	}
	require.NoError(t, narrow.ValidateTopP(0.99))
	require.Error(t, narrow.ValidateTopP(0.98))
	require.Error(t, narrow.ValidateTopP(1))
}

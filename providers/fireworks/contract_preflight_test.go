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

package fireworks_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func contractPreflight(t *testing.T, cases []contractCase, spend *contractSpend) {
	t.Helper()

	worksheet, err := contractPublishedRates()
	require.NoError(t, err)
	require.Equal(t, "https://docs.fireworks.ai/serverless/pricing", worksheet.Source)
	require.Equal(t, "usd_per_million_tokens", worksheet.Unit)
	require.NotEmpty(t, worksheet.Checked)
	var worst float64

	for _, c := range cases {
		rate, ok := worksheet.Models[c.Model]
		require.True(t, ok, "missing independently sourced rates for %s", c.Model)
		require.Positive(t, rate.Standard.Input)
		require.Positive(t, rate.Standard.Output)

		if c.Feature == contractInvalidAuth || c.Feature == contractInvalidModel || c.Feature == contractMalformed {
			continue
		}

		request := contractRequest(t, c)
		b, err := json.Marshal(request)
		require.NoError(t, err)

		output, _ := request["max_tokens"].(int)
		if value, ok := request["max_output_tokens"].(int); ok {
			output = value
		}

		worst += contractReservation(c, int64(len(b)+4096), int64(output))
		if c.Feature == contractTools || c.Feature == contractReasoningHistory || c.Feature == contractCacheRepeat {
			worst += contractReservation(c, 32*1024+4096, int64(output))
		}
	}

	t.Logf("preflight: %d required cases; theoretical full-matrix worst-case USD %.6f; cumulative prior held USD %.6f; approved cap USD %.2f. Sequential reservations reconcile only authoritative returned usage; exhaustion fails incomplete acceptance.", len(cases), worst, spend.spent+spend.reserved, spend.limit)
}

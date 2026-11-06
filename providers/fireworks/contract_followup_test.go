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
	"testing"

	"github.com/stretchr/testify/require"
)

//nolint:tparallel // Only the eight specifically approved corrective pairs may run.
func TestFireworksContractSemanticFollowup(t *testing.T) {
	t.Parallel()

	if !*contractSemanticFollowup {
		return
	}

	previous, source := contractReadReport(t, "testdata/contracts/semantic-controls-v5.json")
	var cases []contractCase
	jsonCases, adaptiveCases := 0, 0

	for _, row := range previous.Evidence {
		if !row.Failed {
			continue
		}

		if row.Case.Feature == contractJSON {
			cases = append(cases, row.Case)
			jsonCases++
		}

		if row.Case.Feature == contractAdaptive && row.Case.Stream {
			cases = append(cases, row.Case)
			adaptiveCases++
		}
	}

	require.Equal(t, 6, jsonCases)
	require.Equal(t, 2, adaptiveCases)
	contractCaptureRequests(t, "testdata/contracts/semantic-followup-v5.json", source, cases, func(c contractCase) contractRequestOptions {
		options := contractSemanticOptions(c)
		if c.Feature == contractJSON {
			options.UserPrompt = "What is the English word for an open field of grass and wildflowers? Answer with only the lowercase word meadow, without quotation marks, punctuation, or extra words."
			options.ExpectedPlain = "meadow"
		} else {
			options.MaxOutputTokens = 4096
		}

		return options
	})
}

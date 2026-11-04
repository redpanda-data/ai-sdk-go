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
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func contractSchemaStreamFixture(t *testing.T) (contractCase, contractExchange) {
	t.Helper()

	report, _ := contractReadReport(t, "testdata/contracts/semantic-controls-v5.json")
	for _, row := range report.Evidence {
		if row.Case.Feature == contractInvalidSchema && row.Case.API == catalog.WireProtocolOpenAIResponses && row.Case.Stream {
			require.Len(t, row.Exchanges, 1)
			return row.Case, row.Exchanges[0]
		}
	}

	t.Fatal("missing actual native streaming schema rejection")

	return contractCase{}, contractExchange{}
}

func TestFireworksContractSyntheticNativeSchemaRejection(t *testing.T) {
	t.Parallel()

	if mode := os.Getenv("FIREWORKS_SCHEMA_REJECTION_CHILD"); mode != "" {
		c, e := contractSchemaStreamFixture(t)
		if mode == "positive-parser" {
			contractParse(t, c, e)
		} else {
			e.Response = strings.ReplaceAll(e.Response, "response.failed", "response.completed")
			contractAssertControlRejection(t, c, e)
		}

		return
	}

	report, _ := contractReadReport(t, "testdata/contracts/semantic-controls-v5.json")
	count := 0

	for _, row := range report.Evidence {
		if row.Case.Feature != contractInvalidSchema || row.Case.API != catalog.WireProtocolOpenAIResponses || !row.Case.Stream {
			continue
		}

		count++

		t.Run(row.Case.Model, func(t *testing.T) {
			t.Parallel()
			require.Len(t, row.Exchanges, 1)
			contractAssertControlRejection(t, row.Case, row.Exchanges[0])
		})
	}

	require.Equal(t, 7, count)

	for _, mode := range []string{"positive-parser", "fake-completion"} {
		t.Run(mode, func(t *testing.T) {
			t.Parallel()
			cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestFireworksContractSyntheticNativeSchemaRejection$") //nolint:gosec // The current test executable runs one fixed offline regression.

			cmd.Env = append(os.Environ(), "FIREWORKS_SCHEMA_REJECTION_CHILD="+mode)
			output, err := cmd.CombinedOutput()
			require.Error(t, err, "invalid native stream must not qualify: %s", output)

			if mode == "positive-parser" {
				require.Contains(t, string(output), "upstream stream failed")
			} else {
				require.Contains(t, string(output), "a validation rejection cannot generate content or complete")
			}
		})
	}
}

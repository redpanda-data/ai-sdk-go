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
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

func contractOpenBudget(ctx context.Context, t *testing.T) *contractSpend {
	t.Helper()

	require.Positive(t, *contractBudget, "an explicit per-run budget is required")
	require.False(t, math.IsInf(*contractBudget, 0) || math.IsNaN(*contractBudget), "per-run budget must be finite")

	return &contractSpend{limit: *contractBudget, ctx: ctx}
}

func contractAtomicWrite(t *testing.T, path string, b []byte) {
	t.Helper()
	require.NoError(t, os.MkdirAll(filepath.Dir(path), 0o750))
	file, err := os.CreateTemp(filepath.Dir(path), ".fireworks-contract-*")
	require.NoError(t, err)

	name := file.Name()

	t.Cleanup(func() { err := os.Remove(name); require.True(t, err == nil || os.IsNotExist(err)) })

	_, err = file.Write(b)
	require.NoError(t, err)
	require.NoError(t, file.Sync())
	require.NoError(t, file.Close())
	require.NoError(t, os.Rename(name, path))
}

func contractCheckpoint(t *testing.T, manifest contractManifest, current *contractEvidence, spend *contractSpend) {
	t.Helper()

	if !*contractRecord {
		return
	}

	contractReportBudget(&manifest, spend)

	if current != nil {
		pending := *current
		pending.Failed = true
		manifest.Evidence = append(append([]contractEvidence{}, manifest.Evidence...), pending)
	}

	b, err := json.MarshalIndent(manifest, "", "  ")
	require.NoError(t, err)
	contractAtomicWrite(t, "testdata/contracts/incomplete.json", append(b, '\n'))
}

type contractDollarRates struct {
	Input       float64 `json:"input"`
	CachedInput float64 `json:"cached_input"`
	Output      float64 `json:"output"`
}
type contractRateWorksheet struct {
	Source  string `json:"source"`
	Checked string `json:"checked"`
	Unit    string `json:"unit"`
	Models  map[string]struct {
		Standard contractDollarRates `json:"standard"`
		Priority contractDollarRates `json:"priority"`
	} `json:"models"`
}

var contractPublishedRates = sync.OnceValues(func() (contractRateWorksheet, error) {
	var result contractRateWorksheet

	b, err := os.ReadFile("testdata/contracts/published-rates.json")
	if err != nil {
		return result, err
	}

	err = json.Unmarshal(b, &result)

	return result, err
})

func contractReservation(c contractCase, input, output int64) float64 {
	bound := contractCost(c, input, 0, output)
	for _, o := range fireworks.Catalog().All() {
		if o.ID == c.Model {
			rates := o.Pricing.Default.Base
			for _, override := range o.Pricing.Overrides {
				if c.Feature == contractPriority && override.Match.ServiceTier == contractPriority {
					rates = override.RateCard.Base
				}
			}

			catalogBound := (float64(input)*float64(max(rates.InputPerMillion, 0)) + float64(output)*float64(max(rates.OutputPerMillion, 0))) / 1e14
			bound = max(bound, catalogBound)
		}
	}

	return bound
}

func contractReportBudget(manifest *contractManifest, spend *contractSpend) {
	manifest.ObservedUSD = 0
	manifest.KnownRunUSD = spend.known
	manifest.PriorConservativeUSD = spend.prior
	manifest.CumulativeConservativeUSD = spend.spent + spend.reserved
	manifest.UnresolvedReservationsUSD = spend.reserved
}

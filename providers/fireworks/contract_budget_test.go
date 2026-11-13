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
	"testing"

	"github.com/stretchr/testify/require"
)

func contractOpenBudget(ctx context.Context, t *testing.T) *contractSpend {
	t.Helper()

	require.Positive(t, *contractBudget, "an explicit per-run budget is required")
	require.False(t, math.IsInf(*contractBudget, 0) || math.IsNaN(*contractBudget), "per-run budget must be finite")

	return &contractSpend{limit: *contractBudget, ctx: ctx}
}

//nolint:gosec // The local operator explicitly selects the evidence output path.
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
	contractAtomicWrite(t, *contractRecordOutput+".incomplete.json", append(b, '\n'))
}

func contractReportBudget(manifest *contractManifest, spend *contractSpend) {
	manifest.ObservedUSD = 0
	manifest.KnownRunUSD = spend.known
	manifest.PriorConservativeUSD = spend.prior
	manifest.CumulativeConservativeUSD = spend.spent + spend.reserved
	manifest.UnresolvedReservationsUSD = spend.reserved
}

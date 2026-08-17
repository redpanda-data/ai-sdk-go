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
	"bytes"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func snapshotFixture(t *testing.T) *Catalog {
	t.Helper()

	robin2 := validEntry("robin-2", "acme/robin-2")
	robin2.Aliases = []string{"robin-latest"}
	robin2.Capabilities.Reasoning = true
	robin2.Reasoning = ReasoningSupport{
		Efforts:  []llm.ReasoningEffort{"low", "high"},
		Adaptive: true,
	}
	robin2.Speeds = []llm.Speed{llm.SpeedStandard, llm.SpeedFast}
	robin2.Pricing = robin2.Pricing.WithOverride(
		pricing.Selector{Speed: llm.SpeedFast},
		pricing.RateCard{Base: pricing.NewRates(6.00, 30.00, 0.60)},
	)
	robin2.Life.RetirementNotBefore = MustDate("2027-01-01")
	robin2.Tuning = Tuning{DefaultMaxOutputTokens: 4096, CompactAtInputTokens: 150_000}
	robin2.Attributes = map[string]string{"zone": "a", "alpha": "1"}

	robin3 := validEntry("robin-3", "acme/robin-3")
	robin3.Life.Stage = StagePreview

	robin1 := validEntry("robin-1", "acme/robin-1")
	robin1.Life.Deprecated = MustDate("2026-01-15")
	robin1.Life.Retires = MustDate("2026-06-15")
	robin1.Life.ReplacedBy = "robin-2"

	return mustCatalog(t, robin2, robin3, robin1)
}

func TestEncodeSnapshotDeterministic(t *testing.T) {
	t.Parallel()

	c := snapshotFixture(t)

	var a, b bytes.Buffer
	require.NoError(t, EncodeSnapshot(&a, c))
	require.NoError(t, EncodeSnapshot(&b, c))
	assert.Equal(t, a.String(), b.String(), "snapshot must be byte-deterministic")
}

func TestEncodeSnapshotShape(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	require.NoError(t, EncodeSnapshot(&buf, snapshotFixture(t)))

	var snap map[string]any
	require.NoError(t, json.Unmarshal(buf.Bytes(), &snap))

	assert.InDelta(t, SnapshotSchemaVersion, snap["schema_version"], 0)

	facts, ok := snap["facts"].(map[string]any)
	require.True(t, ok)
	require.Contains(t, facts, "acme/robin-2")

	robin2Facts, ok := facts["acme/robin-2"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "Robin 2", robin2Facts["name"])
	assert.Equal(t, "robin", robin2Facts["series"])
	assert.Equal(t, "2025-08-01", robin2Facts["released"])

	providers, ok := snap["providers"].([]any)
	require.True(t, ok)
	require.Len(t, providers, 1)

	prov, ok := providers[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "acme", prov["provider"])

	offerings, ok := prov["offerings"].([]any)
	require.True(t, ok)
	require.Len(t, offerings, 3)

	// Offerings are sorted by ID.
	first, ok := offerings[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "robin-1", first["id"])

	// Time-independent derivations only: robin-1 is retired by date,
	// but the snapshot carries the DATES, never a computed stage — the
	// artifact must not change when the clock does.
	lifecycle, ok := first["lifecycle"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "ga", lifecycle["stage"])
	assert.Equal(t, "2026-06-15", lifecycle["retires"])
	assert.Equal(t, "robin-2", lifecycle["replaced_by"])

	derived, ok := first["derived"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "medium", derived["price_tier"])
	assert.Equal(t, "acme/robin-2", derived["successor"])

	// Attributes are sorted key/value pairs.
	second, ok := offerings[1].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "robin-2", second["id"])

	attrs, ok := second["attributes"].([]any)
	require.True(t, ok)
	require.Len(t, attrs, 2)
	firstAttr, ok := attrs[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "alpha", firstAttr["key"])

	tuning, ok := second["tuning"].(map[string]any)
	require.True(t, ok)
	assert.InDelta(t, 4096, tuning["default_max_output_tokens"], 0)

	// Zero-valued tuning is omitted entirely.
	third, ok := offerings[2].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "robin-3", third["id"])
	assert.NotContains(t, third, "tuning")
}

func TestEncodeSnapshotRejectsConflictingFacts(t *testing.T) {
	t.Parallel()

	regA := testRegistry()
	regB := testRegistry()
	regB["acme/robin-2"] = Facts{
		Name: "Robin 2 (divergent)", Series: "robin",
		Released: MustDate("2025-08-01"),
	}

	a, err := New("acme", []Entry{validEntry("robin-2", "acme/robin-2")}, WithRegistry(regA))
	require.NoError(t, err)

	b, err := New("emca", []Entry{validEntry("robin-2", "acme/robin-2")}, WithRegistry(regB))
	require.NoError(t, err)

	var buf bytes.Buffer
	err = EncodeSnapshot(&buf, a, b)
	require.Error(t, err)
	require.ErrorContains(t, err, "conflicting facts")
}

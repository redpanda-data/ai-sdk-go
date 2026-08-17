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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func ids(offerings []Offering) []string {
	out := make([]string, len(offerings))
	for i, o := range offerings {
		out[i] = o.ID
	}

	return out
}

func TestViewRetirementBoundary(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-1", "acme/robin-1")
	e.Life.Deprecated = MustDate("2026-01-01")
	e.Life.Retires = MustDate("2026-06-15")
	c := mustCatalog(t, e)

	// The day before: deprecated, not retired.
	v := c.At(MustDate("2026-06-14"))
	assert.True(t, v.IsDeprecated("robin-1"))
	assert.False(t, v.IsRetired("robin-1"))

	// The boundary is INCLUSIVE: retired ON the shutdown date, matching
	// provider wording ("requests to retired models will fail").
	v = c.At(MustDate("2026-06-15"))
	assert.True(t, v.IsRetired("robin-1"))
	assert.False(t, v.IsDeprecated("robin-1"), "retired outranks deprecated")

	stage, ok := v.Stage("robin-1")
	require.True(t, ok)
	assert.Equal(t, StageRetired, stage)

	// Unknown offerings classify as nothing.
	_, ok = v.Stage("robin-9")
	assert.False(t, ok)
	assert.False(t, v.IsRetired("robin-9"))
}

func TestViewRetirementFloorNeverRetires(t *testing.T) {
	t.Parallel()

	// A published "not sooner than" floor is a lower bound, not a
	// shutdown date: crossing it must not classify the offering as
	// retired.
	e := validEntry("robin-2", "acme/robin-2")
	e.Life.RetirementNotBefore = MustDate("2026-01-01")
	c := mustCatalog(t, e)

	v := c.At(MustDate("2030-01-01"))
	assert.False(t, v.IsRetired("robin-2"))

	stage, ok := v.Stage("robin-2")
	require.True(t, ok)
	assert.Equal(t, StageGA, stage)
}

func TestViewDeprecationBoundary(t *testing.T) {
	t.Parallel()

	e := validEntry("robin-2", "acme/robin-2")
	e.Life.Deprecated = MustDate("2026-06-01")
	c := mustCatalog(t, e)

	assert.False(t, c.At(MustDate("2026-05-31")).IsDeprecated("robin-2"))
	assert.True(t, c.At(MustDate("2026-06-01")).IsDeprecated("robin-2"), "deprecation boundary is inclusive")
}

func TestViewGenerationAndLifecycleAreOrthogonal(t *testing.T) {
	t.Parallel()

	// robin-3 is the newest robin and is DEPRECATED; robin-2 is an older
	// GA offering; robin-1 is retired. wren-1 is a single-member series.
	robin1 := validEntry("robin-1", "acme/robin-1")
	robin1.Life.Retires = MustDate("2026-01-01")

	robin2 := validEntry("robin-2", "acme/robin-2")

	robin3 := validEntry("robin-3", "acme/robin-3")
	robin3.Life.Deprecated = MustDate("2026-05-01")

	wren := validEntry("wren-1", "acme/wren-1")

	c := mustCatalog(t, robin1, robin2, robin3, wren)
	v := c.At(MustDate("2026-08-01"))

	// Current: the newest non-retired generation per series — the
	// deprecated robin-3 is STILL the current generation (render the
	// badge alongside), and wren-1 is current in its own series.
	assert.Equal(t, []string{"robin-3", "wren-1"}, ids(v.Current()))

	// Previous: available offerings of non-current generations,
	// regardless of stage. robin-2 is GA and previous; robin-1 is
	// retired so it is in neither Current nor Previous.
	assert.Equal(t, []string{"robin-2"}, ids(v.Previous()))

	// Deprecated and Retired are lifecycle views, orthogonal to the
	// generation split.
	assert.Equal(t, []string{"robin-3"}, ids(v.Deprecated()))
	assert.Equal(t, []string{"robin-1"}, ids(v.Retired()))
}

func TestViewCurrentFallsBackWhenNewestIsRetired(t *testing.T) {
	t.Parallel()

	// When every offering of the newest model is retired, the series is
	// represented by the next-newest model with a live offering.
	robin2 := validEntry("robin-2", "acme/robin-2")

	robin3 := validEntry("robin-3", "acme/robin-3")
	robin3.Life.Retires = MustDate("2026-06-15")

	c := mustCatalog(t, robin2, robin3)

	v := c.At(MustDate("2026-06-14"))
	assert.Equal(t, []string{"robin-3"}, ids(v.Current()))
	assert.Equal(t, []string{"robin-2"}, ids(v.Previous()))

	v = c.At(MustDate("2026-06-15"))
	assert.Equal(t, []string{"robin-2"}, ids(v.Current()), "series falls back to the newest live model")
	assert.Empty(t, v.Previous())
	assert.Equal(t, []string{"robin-3"}, ids(v.Retired()))
}

func TestViewMultiOfferingModel(t *testing.T) {
	t.Parallel()

	// Geo-variant shape: every offering of the current model is Current
	// — variants do not compete, because generation is computed per
	// logical ModelID.
	c := mustCatalog(t,
		validEntry("robin-2", "acme/robin-2"),
		validEntry("robin-3", "acme/robin-3"),
		validEntry("eu.robin-3", "acme/robin-3"),
		validEntry("us.robin-3", "acme/robin-3"),
	)

	v := c.At(MustDate("2026-08-01"))
	assert.Equal(t, []string{"eu.robin-3", "robin-3", "us.robin-3"}, ids(v.Current()))
	assert.Equal(t, []string{"robin-2"}, ids(v.Previous()))
}

func TestViewFutureAvailability(t *testing.T) {
	t.Parallel()

	// An offering whose Available date is in the future is not yet
	// servable; classification simply reflects the authored stage, and
	// generation math still counts it (the provider chose to publish
	// it). This pins the CURRENT behavior: availability gating is the
	// caller's concern, not View's.
	e := validEntry("robin-2", "acme/robin-2")
	e.Life.Available = MustDate("2026-09-01")
	c := mustCatalog(t, e)

	v := c.At(MustDate("2026-08-01"))
	stage, ok := v.Stage("robin-2")
	require.True(t, ok)
	assert.Equal(t, StageGA, stage)
}

func TestViewReturnsDeepCopies(t *testing.T) {
	t.Parallel()

	c := mustCatalog(t, validEntry("robin-2", "acme/robin-2"))

	got := c.Now().Current()
	require.Len(t, got, 1)
	got[0].Constraints.SupportedParams[0] = mutated

	again, ok := c.Lookup("robin-2")
	require.True(t, ok)
	assert.Equal(t, "max_tokens", again.Constraints.SupportedParams[0])
}

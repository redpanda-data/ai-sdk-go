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

package bedrock

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func TestSonnet45RegionalLifecycle(t *testing.T) {
	t.Parallel()
	const (
		govCloud  = "us-gov-west-1"
		virginia  = "us-east-1"
		frankfurt = "eu-central-1"
		sydney    = "ap-southeast-2"
		tokyo     = "ap-northeast-1"
		calgary   = "ca-west-1"
	)

	for _, tt := range []struct{ id, region string }{
		{ModelClaudeSonnet45US, virginia},
		{ModelClaudeSonnet45EU, frankfurt},
		{ModelClaudeSonnet45AU, sydney},
		{ModelClaudeSonnet45JP, tokyo},
		{ModelClaudeSonnet45Global, calgary},
	} {
		t.Run(tt.id, func(t *testing.T) {
			t.Parallel()

			life, ok := LifecycleForRegion(tt.id, tt.region)
			require.True(t, ok)
			assert.Equal(t, catalog.MustDate("2026-10-08"), life.Deprecated)
			assert.Equal(t, catalog.MustDate("2027-04-08"), life.Retires)

			o, ok := Catalog().Lookup(tt.id)
			require.True(t, ok)
			assert.True(t, o.Life.Retires.IsZero(), "region-specific date must not become worldwide EOL")
		})
	}

	for _, region := range []string{"me-south-1", govCloud, "", "unannounced-region"} {
		life, ok := LifecycleForRegion(ModelClaudeSonnet45Global, region)
		require.True(t, ok)
		assert.True(t, life.Deprecated.IsZero())
		assert.True(t, life.Retires.IsZero())
	}

	life, ok := LifecycleForRegion(ModelClaudeSonnet55Global, virginia)
	require.True(t, ok)
	assert.True(t, life.Retires.IsZero())

	_, ok = LifecycleForRegion("unpublished", virginia)
	assert.False(t, ok)
}

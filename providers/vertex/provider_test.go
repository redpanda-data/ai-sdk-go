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

package vertex_test

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

func TestProviderName(t *testing.T) {
	t.Parallel()

	p, err := vertex.NewProvider(context.Background())
	require.NoError(t, err)
	assert.Equal(t, llm.ProviderID("gcp.vertex"), p.Name())
}

// TestProviderCatalog checks the provider surfaces the same catalog the
// package builds, so a consumer holding the provider and one calling the
// package function see identical offerings.
func TestProviderCatalog(t *testing.T) {
	t.Parallel()

	p, err := vertex.NewProvider(context.Background())
	require.NoError(t, err)
	assert.Same(t, vertex.Catalog(), p.Catalog())
}

// TestWithLocation_Shape checks WithLocation takes the location shape
// cloudv2 stores (llm_provider.proto, VertexConfig.location) and refuses
// anything else.
func TestWithLocation_Shape(t *testing.T) {
	t.Parallel()

	for _, location := range []string{"global", "us", "eu", "us-east5", "northamerica-northeast1", "europe-west12", " US-East5 "} {
		_, err := vertex.NewProvider(context.Background(), vertex.WithLocation(location))
		require.NoError(t, err, location)
	}

	for _, location := range []string{"attacker.example#", "evil.com/", "us-central1@evil", "us-", "1abc", "us--east5", "a" + strings.Repeat("b", 63)} {
		_, err := vertex.NewProvider(context.Background(), vertex.WithLocation(location))
		assert.Error(t, err, location)
	}
}

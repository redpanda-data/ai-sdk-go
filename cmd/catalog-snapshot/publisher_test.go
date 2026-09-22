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

package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// catalogPublisher gives, for each catalog, the publisher every one of its
// offerings carries. An empty value marks a multi-vendor catalog.
var catalogPublisher = map[llm.ProviderID]string{
	"anthropic":   catalog.PublisherAnthropic,
	"gcp.gemini":  catalog.PublisherGoogle,
	"meta":        catalog.PublisherMeta,
	"openai":      catalog.PublisherOpenAI,
	"aws.bedrock": "",
	"gcp.vertex":  "",
}

func TestEveryOfferingDeclaresAPublisher(t *testing.T) {
	t.Parallel()

	for _, cat := range allCatalogs() {
		wantPublisher, listed := catalogPublisher[cat.Provider()]
		require.Truef(t, listed, "%s is not listed in catalogPublisher", cat.Provider())

		for _, o := range cat.All() {
			assert.NotEmptyf(t, o.Publisher, "%s/%s declares no publisher", cat.Provider(), o.ID)

			if wantPublisher != "" {
				assert.Equalf(t, wantPublisher, o.Publisher, "%s/%s publisher", cat.Provider(), o.ID)
			}
		}
	}
}

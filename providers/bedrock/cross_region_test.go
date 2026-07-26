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
)

func TestIsModelAllowedFromRegion(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		modelID string
		region  string
		want    bool
	}{
		// Bare model IDs are always allowed — AWS handles in-region availability.
		{"bare from us", ModelClaudeSonnet45, "us-east-1", true},
		{"bare from eu", ModelClaudeSonnet45, "eu-west-1", true},
		{"bare from unknown region", ModelClaudeSonnet45, "xx-fake-1", true},

		// Sonnet 5 — only us. and global. are published; us. honours geo rules.
		{"sonnet5 bare from eu", ModelClaudeSonnet5, "eu-west-1", true},
		{"sonnet5 us from us-east-1", ModelClaudeSonnet5US, "us-east-1", true},
		{"sonnet5 us from ca-central-1 (Canada is US Geo)", ModelClaudeSonnet5US, "ca-central-1", true},
		{"sonnet5 us from eu-west-1 (cross-geo)", ModelClaudeSonnet5US, "eu-west-1", false},
		{"sonnet5 global from me-central-1", ModelClaudeSonnet5Global, "me-central-1", true},

		// global.* is always allowed.
		{"global from us", ModelClaudeSonnet46Global, "us-east-1", true},
		{"global from eu", ModelClaudeSonnet46Global, "eu-west-1", true},
		{"global from me-central-1", ModelClaudeOpus46Global, "me-central-1", true},
		{"global from sa-east-1", ModelClaudeOpus47Global, "sa-east-1", true},

		// Matching geo — allowed.
		{"us from us-east-1", ModelClaudeSonnet46US, "us-east-1", true},
		{"us from us-west-2", ModelClaudeSonnet46US, "us-west-2", true},
		{"us from ca-central-1 (Canada is US Geo)", ModelClaudeSonnet46US, "ca-central-1", true},
		{"us from ca-west-1 (Calgary is US Geo)", ModelClaudeOpus46US, "ca-west-1", true},
		{"Haiku 4.5 US from Calgary", ModelClaudeHaiku45US, "ca-west-1", false},
		{"Opus 4.5 US from Calgary", ModelClaudeOpus45US, "ca-west-1", false},
		{"eu from eu-west-1", ModelClaudeSonnet46EU, "eu-west-1", true},
		{"eu from eu-central-2", ModelClaudeSonnet46EU, "eu-central-2", true},
		{"jp from ap-northeast-1 (Tokyo)", ModelClaudeSonnet45JP, "ap-northeast-1", true},
		{"jp from ap-northeast-3 (Osaka)", ModelClaudeSonnet45JP, "ap-northeast-3", true},
		{"au from ap-southeast-2 (Sydney)", ModelClaudeHaiku45AU, "ap-southeast-2", true},
		{"au from ap-southeast-4 (Melbourne)", ModelClaudeHaiku45AU, "ap-southeast-4", true},
		{"au from ap-southeast-6 (New Zealand)", ModelClaudeOpus46AU, "ap-southeast-6", true},
		{"Opus 4.7 AU from New Zealand", ModelClaudeOpus47AU, "ap-southeast-6", false},
		{"Opus 4.8 AU from New Zealand", ModelClaudeOpus48AU, "ap-southeast-6", false},

		// Amazon Nova 2 Lite — same prefix-based routing as Claude (us/eu/jp
		// geo profiles + global). Confirms the amazon. namespace routes like
		// anthropic.
		{"nova2 bare from us", ModelNova2Lite, "us-east-1", true},
		{"nova2 us from us-east-1", ModelNova2LiteUS, "us-east-1", true},
		{"nova2 eu from eu-west-1", ModelNova2LiteEU, "eu-west-1", true},
		{"nova2 jp from ap-northeast-1 (Tokyo)", ModelNova2LiteJP, "ap-northeast-1", true},
		{"nova2 global from me-central-1", ModelNova2LiteGlobal, "me-central-1", true},
		{"nova2 us from eu-west-1 (cross-geo)", ModelNova2LiteUS, "eu-west-1", false},
		{"nova2 eu from us-east-1 (cross-geo)", ModelNova2LiteEU, "us-east-1", false},

		// Cross-geography — rejected.
		{"eu from us-east-1", ModelClaudeSonnet46EU, "us-east-1", false},
		{"us from eu-west-1", ModelClaudeSonnet46US, "eu-west-1", false},
		{"jp from us-east-1", ModelClaudeSonnet45JP, "us-east-1", false},
		{"jp from eu-central-1", ModelClaudeSonnet45JP, "eu-central-1", false},
		{"au from us-east-1", ModelClaudeHaiku45AU, "us-east-1", false},
		{"au from eu-west-1", ModelClaudeOpus46AU, "eu-west-1", false},
		{"eu from ap-northeast-1", ModelClaudeSonnet46EU, "ap-northeast-1", false},
		{"jp from ap-southeast-2 (Sydney is AU, not JP)", ModelClaudeSonnet45JP, "ap-southeast-2", false},
		{"au from ap-northeast-1 (Tokyo is JP, not AU)", ModelClaudeHaiku45AU, "ap-northeast-1", false},

		// Global-only source regions — only global. and bare allowed.
		{"us from me-central-1 (UAE — global-only)", ModelClaudeSonnet46US, "me-central-1", false},
		{"eu from me-central-1", ModelClaudeSonnet46EU, "me-central-1", false},
		{"jp from sa-east-1", ModelClaudeSonnet45JP, "sa-east-1", false},
		{"au from ap-south-1 (Mumbai — global-only)", ModelClaudeHaiku45AU, "ap-south-1", false},
		{"us from ap-northeast-2 (Seoul — global-only)", ModelClaudeSonnet46US, "ap-northeast-2", false},
		{"global from me-central-1", ModelClaudeSonnet46Global, "me-central-1", true},

		// Unknown region — any geo prefix is rejected (we don't know what
		// AWS would do, so fail fast).
		{"us from unknown region", ModelClaudeSonnet46US, "xx-fake-1", false},
		{"Sonnet 4.5 US from GovCloud", ModelClaudeSonnet45US, "us-gov-east-1", true},
		{"Opus 4.8 US from GovCloud", ModelClaudeOpus48US, "us-gov-east-1", false},
		{"global from GovCloud", ModelClaudeSonnet45Global, "us-gov-east-1", false},
		{"global from unknown region", ModelClaudeSonnet46Global, "xx-fake-1", true},
		{"bare from unknown region (allowed)", ModelClaudeSonnet45, "xx-fake-1", true},

		// Vendor-namespaced IDs with a dotted version (e.g. "openai.gpt-5.5")
		// are bare, not region-prefixed, and must be allowed from any region —
		// regression for the dot-counting misclassification that blocked them
		// everywhere.
		{"openai dotted-version bare from us-east-2", "openai.gpt-5.5", "us-east-2", true},
		{"openai dotted-version bare from eu-west-1", "openai.gpt-5.5", "eu-west-1", true},
		{"openai dotted-version bare from unknown region", "openai.gpt-5.5", "xx-fake-1", true},
		// A real geo prefix on such an ID is still enforced.
		{"eu-prefixed openai from us-east-1 (cross-geo)", "eu.openai.gpt-5.5", "us-east-1", false},
		{"us-prefixed openai from us-east-1", "us.openai.gpt-5.5", "us-east-1", true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := IsModelAllowedFromRegion(tc.modelID, tc.region)
			if got != tc.want {
				t.Errorf("IsModelAllowedFromRegion(%q, %q) = %v, want %v",
					tc.modelID, tc.region, got, tc.want)
			}
		})
	}
}

func TestIsModelAllowedFromRegion_AllSupportedModels(t *testing.T) {
	t.Parallel()

	// Every cataloged model must be invokable from at least one source region
	// (its bare or global. variant from any region; geo variants from a
	// matching geo). This guards against typos in model constants or geo
	// profile lists and ensures the catalog and the rule stay in sync.
	for name := range supportedModels {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// Bare and global variants must be allowed from a US region.
			if !IsModelAllowedFromRegion(name, "us-east-1") &&
				!hasGeoPrefix(name, "eu", "au", "jp") {
				t.Errorf("model %q rejected from us-east-1 unexpectedly", name)
			}
		})
	}
}

func hasGeoPrefix(modelID string, prefixes ...string) bool {
	for _, p := range prefixes {
		if len(modelID) > len(p)+1 && modelID[:len(p)+1] == p+"." {
			return true
		}
	}

	return false
}

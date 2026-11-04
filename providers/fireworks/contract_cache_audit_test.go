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
	"flag"
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/fireworks"
)

var (
	contractCacheAudit   = flag.Bool("fireworks-cache-audit", false, "controlled, billable base-versus-1017 cache characterization only")
	contractCacheVariant = flag.String("fireworks-cache-variant", "affinity", "affinity, preferred, or preferred-isolated")
	contractCachePrefix  = flag.Int("fireworks-cache-prefix-bytes", 8192, "empirical cache prefix bytes; single nonstream diagnostic supports 8KiB or 64KiB")
)

//nolint:tparallel // Paid comparisons deliberately run sequentially with bounded reservations.
func TestFireworksContractCacheAudit(t *testing.T) {
	t.Parallel()

	if !*contractCacheAudit {
		return
	}

	require.Contains(t, []string{"affinity", "preferred", "preferred-isolated"}, *contractCacheVariant)
	require.GreaterOrEqual(t, *contractCachePrefix, 1024)
	require.LessOrEqual(t, *contractCachePrefix, 16*1024)
	_, source := contractReadReport(t, "testdata/contracts/full-matrix-v2-incomplete.json")
	inventory, _ := contractInventory(t)
	var cases []contractCase

	for _, c := range inventory {
		if c.Feature == contractCacheRepeat && (c.Model == fireworks.ModelDeepSeekV41Flash || c.Model == fireworks.ModelDeepSeekV41Flash1017) {
			cases = append(cases, c)
		}
	}

	require.Len(t, cases, 12, "comparison must include both active offerings, every declared API and both modes")
	slices.SortFunc(cases, func(a, b contractCase) int {
		left := fmt.Sprintf("%s|%t|%s", a.API, a.Stream, a.Model)

		right := fmt.Sprintf("%s|%t|%s", b.API, b.Stream, b.Model)
		if left < right {
			return -1
		}

		if left > right {
			return 1
		}

		return 0
	})

	cohort := fmt.Sprintf("cache-v4-%d-%s", *contractCachePrefix, *contractCacheVariant)
	path := "testdata/contracts/" + cohort + ".json"

	report := contractCaptureRequests(t, path, source, cases, func(c contractCase) contractRequestOptions {
		key := fmt.Sprintf("%s-%s-%t", cohort, c.API, c.Stream)

		opts := contractRequestOptions{CachePrefixBytes: *contractCachePrefix, CacheProbeID: key, CacheRepeatCount: 3, CacheSettleMS: 2000, MaxOutputTokens: 16, AffinityOverride: key}
		if c.API == catalog.WireProtocolOpenAIChatCompletions {
			opts.PerfMetrics = true
		}

		if *contractCacheVariant != "affinity" {
			switch c.API {
			case catalog.WireProtocolAnthropicMessages:
				opts.ClaudeSessionID = key
			case catalog.WireProtocolOpenAIChatCompletions:
				opts.PromptCacheKey = key
			case catalog.WireProtocolOpenAIResponses:
				opts.User = key
			case catalog.WireProtocolGeminiGenerateContent, catalog.WireProtocolBedrockConverse:
				t.Fatal("unsupported diagnostic protocol")
			}
		}

		if *contractCacheVariant == "preferred-isolated" {
			opts.CacheIsolation = key
		}

		return opts
	})
	for _, row := range report.Evidence {
		observed := row
		contractObserveEvidence(t, &observed)
		t.Logf("CACHE OBSERVATION model=%s api=%s stream=%t hit=%t fresh=%t exchanges=%d", row.Case.Model, row.Case.API, row.Case.Stream, observed.CacheHitObserved, observed.FreshInputObserved, len(row.Exchanges))
	}
}

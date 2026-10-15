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

package openai

import (
	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

// Explicit snapshots outrank family resolution, preserving snapshot-specific
// retirement while leaving movable aliases untouched. Rates and capabilities
// follow the family except the original May 2024 snapshot noted below.
// https://developers.openai.com/api/docs/deprecations (checked 2026-10-15)
func appendDatedSnapshots(entries []catalog.Entry) []catalog.Entry {
	const (
		deprecated = "2026-10-14"
		retires    = "2027-01-14"
	)

	byID := make(map[string]catalog.Entry, len(entries))
	for _, entry := range entries {
		byID[entry.ID] = entry
	}

	for _, snapshot := range []struct{ base, id, available, deprecated, retires, replacement string }{
		{ModelGPT4O, ModelGPT4O20240513, "2024-05-13", "2026-04-22", "2026-10-23", ModelGPT5_6Sol},
		{ModelGPT4O, ModelGPT4O20240806, "2024-08-06", deprecated, retires, ModelGPT6Luna},
		{ModelGPT4O, ModelGPT4O20241120, "2024-11-20", deprecated, retires, ModelGPT6Luna},
		{ModelGPT4OMini, ModelGPT4OMini20240718, "2024-07-18", deprecated, retires, ModelGPT6Luna},
		{ModelGPT41, ModelGPT4120250414, "2025-04-14", deprecated, retires, ModelGPT6Luna},
		{ModelGPT41Mini, ModelGPT41Mini20250414, "2025-04-14", deprecated, retires, ModelGPT6Luna},
	} {
		entry := byID[snapshot.base]
		entry.ID = snapshot.id
		entry.Aliases = nil

		entry.Life = catalog.Lifecycle{Available: catalog.MustDate(snapshot.available), Deprecated: catalog.MustDate(snapshot.deprecated), Retires: catalog.MustDate(snapshot.retires), ReplacedBy: snapshot.replacement}
		if entry.ID == ModelGPT4O20240513 {
			// The original snapshot predates structured response formats; keep a
			// conservative 4K SDK output bound, not the current alias's 16K cap.
			// https://openai.com/index/introducing-structured-outputs-in-the-api/
			entry.Capabilities.StructuredOutput = false
			entry.Constraints.MaxOutputTokens = 4096
			// The May snapshot has its own published legacy rates and no cache rate.
			// https://developers.openai.com/api/docs/pricing
			entry.Pricing = pricing.FlatInfo(5, 15, 0).
				WithOverride(batchTier, flatCard(2.50, 7.50, 0)).
				WithOverride(fastTier, flatCard(8.75, 26.25, 0))
		}

		entries = append(entries, entry)
	}

	return entries
}

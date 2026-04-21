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

package llm

import "strings"

// ServiceTier identifies the provider-reported processing tier for a request.
//
// Pricing is not determined by model ID alone. Several providers expose request
// variants whose rates differ even when the model stays the same: OpenAI has
// flex/priority/scale, Anthropic has batch/priority, and Bedrock has reserved
// and provisioned-throughput modes.
//
// ServiceTier is a string-backed type so that unknown provider tiers can be
// preserved verbatim while known tiers are comparable against constants in
// switch statements. The empty value means the provider did not report a
// tier; any non-empty non-constant value is a provider-native tier the SDK
// does not recognize yet (still lower-cased and dash-normalized).
//
// This type lives in llm rather than pricing because it describes response
// metadata emitted by providers. The pricing package consumes it as one
// selector dimension when choosing a rate card.
type ServiceTier string

const (
	// ServiceTierDefault is the provider's base rate card. Providers that
	// report "default", "standard", or "auto" all normalize here. The empty
	// ServiceTier (unreported) is distinct from Default.
	ServiceTierDefault ServiceTier = "default"

	// ServiceTierFlex is a discounted, best-effort tier.
	ServiceTierFlex ServiceTier = "flex"

	// ServiceTierPriority is a premium lower-latency tier.
	ServiceTierPriority ServiceTier = "priority"

	// ServiceTierBatch is an asynchronous discounted tier.
	ServiceTierBatch ServiceTier = "batch"

	// ServiceTierScale is OpenAI's scale tier.
	ServiceTierScale ServiceTier = "scale"

	// ServiceTierReserved is a reserved-capacity tier.
	ServiceTierReserved ServiceTier = "reserved"

	// ServiceTierProvisionedThroughput is a provisioned-throughput tier.
	ServiceTierProvisionedThroughput ServiceTier = "provisioned_throughput"
)

// NormalizeServiceTier maps a provider-native tier string onto the SDK's
// canonical ServiceTier values.
//
// Aliases ("standard", "auto") collapse to the canonical constant. Unknown
// non-empty tiers are preserved verbatim (lower-cased, trimmed, with dashes
// converted to underscores) so consumers can still branch on them. The empty
// input yields the empty ServiceTier, which callers should treat as "not
// reported" rather than as "default".
func NormalizeServiceTier(raw string) ServiceTier {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "" {
		return ""
	}

	normalized := strings.ReplaceAll(raw, "-", "_")

	switch normalized {
	case "default", "standard", "auto":
		return ServiceTierDefault

	case "flex":
		return ServiceTierFlex

	case "priority":
		return ServiceTierPriority

	case "batch":
		return ServiceTierBatch

	case "scale":
		return ServiceTierScale

	case "reserved":
		return ServiceTierReserved

	case "provisioned_throughput":
		return ServiceTierProvisionedThroughput

	default:
		return ServiceTier(normalized)
	}
}

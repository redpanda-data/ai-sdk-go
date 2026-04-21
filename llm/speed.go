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

// Speed identifies the provider-reported latency mode for a request,
// as a cross-provider semantic category rather than a literal provider
// string. Providers use different vocabulary for the same concept
// (Anthropic's "fast" vs Bedrock's "optimized") and NormalizeSpeed
// collapses them to the same Speed* constant so cross-provider
// consumers can branch on the concept without knowing which provider
// served the request. New dialects are added to the alias table as
// they show up.
//
// Unknown non-empty values are preserved verbatim so consumers can
// still branch on them. The empty value means the provider did not
// report a speed.
type Speed string

const (
	// SpeedStandard is the provider's default latency mode. Providers
	// that report "standard" or "default" explicitly normalize here.
	SpeedStandard Speed = "standard"

	// SpeedFast is a premium low-latency mode with a published price
	// premium. Anthropic reports this as "fast"; Bedrock reports it
	// as "optimized"; NormalizeSpeed collapses both to this constant.
	SpeedFast Speed = "fast"
)

// NormalizeSpeed maps a provider-native speed string onto the Speed
// type's canonical values. Synonyms across providers collapse to the
// same constant (see the Speed* docs for the current map). Unknown
// non-empty values are preserved verbatim (lower-cased, trimmed,
// dashes to underscores) so new provider dialects stay visible until
// we add them to the alias table. The empty input yields the empty
// Speed, which callers should treat as "not reported".
func NormalizeSpeed(raw string) Speed {
	raw = strings.ToLower(strings.TrimSpace(raw))
	if raw == "" {
		return ""
	}

	normalized := strings.ReplaceAll(raw, "-", "_")

	switch normalized {
	case "standard", "default":
		return SpeedStandard

	case "fast", "optimized":
		return SpeedFast

	default:
		return Speed(normalized)
	}
}

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

package reconcile

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"
	"slices"
)

// PackageProviders maps a provider package to the provider name the
// snapshot uses for it. cmd/catalog-snapshot holds it equal to each
// package's ProviderName.
var PackageProviders = map[string]string{
	"anthropic": "anthropic",
	"bedrock":   "aws.bedrock",
	"google":    "gcp.gemini",
	"meta":      "meta",
	"openai":    "openai",
	"vertex":    "gcp.vertex",
}

// Snapshot is catalog/snapshot.json as data: each model's facts and each
// provider's offerings, with numbers kept exactly as written.
type Snapshot struct {
	Facts     map[string]map[string]any `json:"facts"`
	Providers []struct {
		Provider  string           `json:"provider"`
		Offerings []map[string]any `json:"offerings"`
	} `json:"providers"`
}

// ParseSnapshot decodes a snapshot, keeping numbers as json.Number.
func ParseSnapshot(data []byte) (Snapshot, error) {
	var snap Snapshot

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()

	if err := decoder.Decode(&snap); err != nil {
		return Snapshot{}, fmt.Errorf("parse snapshot: %w", err)
	}

	return snap, nil
}

// Offerings lists a provider package's offerings.
func (s Snapshot) Offerings(pkg string) []map[string]any {
	var offerings []map[string]any

	for _, provider := range s.Providers {
		if provider.Provider == PackageProviders[pkg] {
			offerings = append(offerings, provider.Offerings...)
		}
	}

	return offerings
}

// Offering finds the offering with this ID in a provider package.
func (s Snapshot) Offering(pkg, id string) (map[string]any, bool) {
	offerings := s.Offerings(pkg)
	if i := slices.IndexFunc(offerings, func(o map[string]any) bool { return o["id"] == id }); i >= 0 {
		return offerings[i], true
	}

	return nil, false
}

// WithoutDerived is an offering without its derived values, which change with
// their neighbours, so comparing offerings ignores them.
func WithoutDerived(offering map[string]any) map[string]any {
	if offering == nil {
		return nil
	}

	offering = maps.Clone(offering)
	delete(offering, "derived")

	return offering
}

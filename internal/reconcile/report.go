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
	"errors"
	"fmt"
	"regexp"
	"strings"
)

// OfferingPattern is how a report names an offering: a model ID, or
// <package>:<model ID> outside the run's own package (RULES.md, Report values).
// A Vertex model ID may end in its date version, as claude-opus-4-5@20251101.
var OfferingPattern = regexp.MustCompile(`^[a-z0-9][a-z0-9._:-]{0,79}(@[0-9]{8})?$`)

// PackageOf names a report offering's package and ID: <package>:<id>, or an
// ID in the run's own package (RULES.md, Report values). Bedrock IDs contain
// ":", so only a known package counts as a prefix.
func PackageOf(offering, vendor string) (string, string) {
	if pkg, id, found := strings.Cut(offering, ":"); found && PackageProviders[pkg] != "" {
		return pkg, id
	}

	return vendor, offering
}

// Report is the agent job's structured output. It deliberately has no
// free-text fields: everything rendered into a PR is an ID, an enum value,
// a short value matching valuePattern, or a URL on a named-source host.
type Report struct {
	Changes    []Change     `json:"changes"`
	NeedsHuman []NeedsHuman `json:"needs_human"`
	Skipped    []Skipped    `json:"skipped"`
}

// Change is one value the agent changed, and the Default that changed it.
type Change struct {
	Offering  string `json:"offering"`
	Field     string `json:"field"`
	Old       string `json:"old"`
	New       string `json:"new"`
	SourceURL string `json:"source_url"`
	Rule      string `json:"rule"`
}

// NeedsHuman is a value the agent couldn't decide. Rule is the RULES.md rule
// that sent it to a person, or no-rule when no rule covers the case.
// Catalogue and Page, when the case has them, are what the catalogue has now
// and what the cited page says; Values holds any other details.
type NeedsHuman struct {
	Offering   string   `json:"offering"`
	Field      string   `json:"field"`
	Catalogue  string   `json:"catalogue"`
	Page       string   `json:"page"`
	Values     []string `json:"values"`
	SourceURLs []string `json:"source_urls"`
	Rule       string   `json:"rule"`
}

// Skipped is an offering the agent left alone, and the rule that left it out.
type Skipped struct {
	Offering string `json:"offering"`
	Reason   string `json:"reason"`
	Rule     string `json:"rule"`
}

// ParseReport decodes a report strictly: unknown fields are an error, and all three
// arrays must be present, so a missing or null report never reads as "no drift".
func ParseReport(data []byte) (Report, error) {
	// Pointers tell a missing or null array apart from an empty one.
	var raw struct {
		Changes    *[]Change     `json:"changes"`
		NeedsHuman *[]NeedsHuman `json:"needs_human"`
		Skipped    *[]Skipped    `json:"skipped"`
	}

	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(&raw); err != nil {
		return Report{}, fmt.Errorf("parse report: %w", err)
	}

	if raw.Changes == nil || raw.NeedsHuman == nil || raw.Skipped == nil {
		return Report{}, errors.New("parse report: changes, needs_human and skipped are required")
	}

	return Report{Changes: *raw.Changes, NeedsHuman: *raw.NeedsHuman, Skipped: *raw.Skipped}, nil
}

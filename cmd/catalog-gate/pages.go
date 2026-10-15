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
	"cmp"
	"encoding/json"
	"fmt"
	"regexp"
	"slices"
	"strings"
)

// page is one source the gate fingerprints, as named-sources.json lists it.
type page struct {
	ID   string `json:"id"`
	Type string `json:"type"`
	URL  string `json:"url"`
	// Fragment is the id of the element that holds the content an html
	// page's run owns, when the page covers other vendors too.
	Fragment string `json:"fragment,omitempty"`
	// MustContain and MinLines are the content floor: the page's text must
	// contain it and have at least that many lines. MustContain defaults to
	// the vendor's.
	MustContain string `json:"must_contain,omitempty"`
	MinLines    int    `json:"min_lines,omitempty"`
	// Match picks a price list's services.
	Match string `json:"match,omitempty"`
}

// safeID is what a page id may be: it names the page's file in the gate's output.
var safeID = regexp.MustCompile(`^[a-z0-9-]{1,80}$`)

// sourcesPath is named-sources.json in the repo: each vendor's packages,
// named hosts and the pages the gate fingerprints.
const sourcesPath = ".github/reconcile-models/named-sources.json"

// vendorSources is one vendor's entry in named-sources.json.
type vendorSources struct {
	Providers []string `json:"providers"`
	Hosts     []string `json:"hosts"`
	Gate      struct {
		// ModelPrefix picks the vendor's models out of the snapshot, and
		// MustContain is the text every page must contain.
		ModelPrefix string     `json:"model_prefix"`
		MustContain string     `json:"must_contain"`
		Pages       []page     `json:"pages"`
		PerModel    []template `json:"per_model"`
	} `json:"gate"`
}

// template is a page per model of one snapshot provider; "{slug}" in its id
// and URL is the model ID without the vendor's model prefix.
type template struct {
	page

	Provider string `json:"provider"`
}

func readSources(raw []byte) (map[string]vendorSources, error) {
	var vendors map[string]vendorSources
	if err := json.Unmarshal(raw, &vendors); err != nil {
		return nil, fmt.Errorf("%s: %w", sourcesPath, err)
	}

	return vendors, nil
}

// loadPages lists a vendor's pages: the listed ones, then one per model and
// template for each of the vendor's models the snapshot still offers on that
// provider, so a new model gets its pages without an edit here.
func loadPages(sources, snapshotJSON []byte, vendor, today string) ([]page, error) {
	vendors, err := readSources(sources)
	if err != nil {
		return nil, err
	}

	config, ok := vendors[vendor]
	if !ok {
		return nil, fmt.Errorf("%s has no vendor %q", sourcesPath, vendor)
	}

	var snap struct {
		Providers []struct {
			Provider  string `json:"provider"`
			Offerings []struct {
				Model     string `json:"model"`
				Lifecycle struct {
					Retires string `json:"retires"`
				} `json:"lifecycle"`
			} `json:"offerings"`
		} `json:"providers"`
	}

	if err := json.Unmarshal(snapshotJSON, &snap); err != nil {
		return nil, fmt.Errorf("snapshot: %w", err)
	}

	pages := slices.Clone(config.Gate.Pages)

	for _, tmpl := range config.Gate.PerModel {
		var slugs []string

		for _, provider := range snap.Providers {
			if provider.Provider != tmpl.Provider {
				continue
			}

			for _, offering := range provider.Offerings {
				retires := offering.Lifecycle.Retires
				if slug, ok := strings.CutPrefix(offering.Model, config.Gate.ModelPrefix); ok && (retires == "" || retires > today) {
					slugs = append(slugs, slug)
				}
			}
		}

		slices.Sort(slugs)

		for _, slug := range slices.Compact(slugs) {
			derived := tmpl.page
			derived.ID = strings.ReplaceAll(tmpl.ID, "{slug}", slug)
			derived.URL = strings.ReplaceAll(tmpl.URL, "{slug}", slug)
			pages = append(pages, derived)
		}
	}

	seen := map[string]bool{}

	for i, p := range pages {
		switch {
		case !safeID.MatchString(p.ID):
			return nil, fmt.Errorf("page id %q isn't safe as a file name", p.ID)
		case seen[p.ID]:
			return nil, fmt.Errorf("duplicate page id %q", p.ID)
		}

		seen[p.ID] = true
		pages[i].MustContain = cmp.Or(p.MustContain, config.Gate.MustContain)
	}

	return pages, nil
}

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

package catalog

import (
	"errors"
	"fmt"
	"slices"
)

// The publisher values in use across the catalogs today. A new vendor
// adds a const here and declares it in its catalog.
const (
	PublisherAmazon    = "amazon"
	PublisherAnthropic = "anthropic"
	PublisherGoogle    = "google"
	PublisherMeta      = "meta"
	PublisherMistral   = "mistral"
	PublisherOpenAI    = "openai"
)

// MustDeclarePublisher returns a copy of entries with [Entry.Publisher]
// set to publisher. It panics on an empty publisher, or on an entry that
// already declares a different one.
func MustDeclarePublisher(publisher string, entries []Entry) []Entry {
	if publisher == "" {
		panic(errors.New("catalog: MustDeclarePublisher needs a publisher")) //nolint:forbidigo // authoring error, not runtime
	}

	out := slices.Clone(entries)

	for i := range out {
		if existing := out[i].Publisher; existing != "" && existing != publisher {
			panic(fmt.Errorf("catalog: entry %s declares publisher %s, not %s", out[i].ID, existing, publisher)) //nolint:forbidigo // authoring error, not runtime
		}

		out[i].Publisher = publisher
	}

	return out
}

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
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// allCatalogs is every provider catalog the snapshot covers. This is the
// only package that may import all four (catalog's architecture test
// forbids it there), so cross-provider invariants live here.
func allCatalogs() []*catalog.Catalog {
	return []*catalog.Catalog{
		anthropic.Catalog(), bedrock.Catalog(), google.Catalog(), openai.Catalog(),
	}
}

// TestDeprecatedOfferingsNameAReplacement is a tripwire: an offering with a
// deprecation or retirement date must tell callers where to go. Missing
// ReplacedBy is how o3, o3-pro and o4-mini sat deprecated with no migration
// target.
//
// If a vendor's recommended replacement is genuinely not an offering we
// carry, CLAUDE.md says to skip ReplacedBy — add the ID here with a comment
// naming the uncarried replacement rather than deleting the assertion.
func TestDeprecatedOfferingsNameAReplacement(t *testing.T) {
	t.Parallel()

	noCarriedReplacement := map[string]string{
		// Google recommends Gemini 3.1 Flash-Lite or Gemma 4; the google
		// catalog carries neither.
		"gemini-2.5-flash-lite": "gemini-3.1-flash-lite / gemma-4",
	}

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			if o.Life.Deprecated.IsZero() && o.Life.Retires.IsZero() {
				continue
			}

			if _, ok := noCarriedReplacement[o.ID]; ok {
				continue
			}

			assert.NotEmptyf(t, o.Life.ReplacedBy,
				"%s/%s is deprecated or retiring but names no ReplacedBy", cat.Provider(), o.ID)
		}
	}
}

// TestReplacementsAreNotThemselvesRetired stops the catalog pointing callers
// at a dead model: a migration target that has already passed its own
// retirement date is worse than none.
func TestReplacementsAreNotThemselvesRetired(t *testing.T) {
	t.Parallel()

	now := time.Now().UTC()

	for _, cat := range allCatalogs() {
		for _, o := range cat.All() {
			if o.Life.ReplacedBy == "" {
				continue
			}

			target, ok := cat.Lookup(o.Life.ReplacedBy)
			if !ok {
				continue // New already rejects an unresolvable ReplacedBy.
			}

			if target.Life.Retires.IsZero() {
				continue
			}

			assert.Falsef(t, target.Life.Retires.Before(now),
				"%s/%s points at %s, which retired %s",
				cat.Provider(), o.ID, target.ID, target.Life.Retires.Format(time.DateOnly))
		}
	}
}

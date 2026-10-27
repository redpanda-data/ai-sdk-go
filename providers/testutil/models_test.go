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

package testutil

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

func TestRetiredReason(t *testing.T) {
	t.Parallel()

	// Hand-dated copies of one real entry, so the cases don't pin OpenAI's
	// real dates, which reconcile runs move.
	base, ok := openai.Catalog().Lookup(openai.ModelGPT5_4Mini)
	require.True(t, ok)

	entry := func(id string, retires time.Time, replacedBy string) catalog.Entry {
		e := base.Entry
		e.ID, e.Aliases = id, nil
		e.Life.Retires, e.Life.ReplacedBy = retires, replacedBy

		return e
	}

	shutdownDay := catalog.MustDate("2027-01-14")

	cat := catalog.MustNew(openai.ProviderName, []catalog.Entry{
		entry("current-model", time.Time{}, ""),
		entry("retiring-model", shutdownDay, "current-model"),
		entry("orphan-model", shutdownDay, ""),
	})

	tests := []struct {
		name       string
		offeringID string
		asOf       time.Time
		wantReason string
		wantErr    string
	}{
		{"no shutdown date runs", "current-model", shutdownDay, "", ""},
		{"day before the shutdown runs", "retiring-model", shutdownDay.AddDate(0, 0, -1), "", ""},
		{
			"shutdown day skips", "retiring-model", shutdownDay,
			"retiring-model retired on 2027-01-14, per the openai catalog; its replacement is current-model", "",
		},
		{
			"shutdown passed skips", "retiring-model", shutdownDay.AddDate(0, 0, 1),
			"retiring-model retired on 2027-01-14, per the openai catalog; its replacement is current-model", "",
		},
		{
			"no replacement recorded", "orphan-model", shutdownDay,
			"orphan-model retired on 2027-01-14, per the openai catalog, which records no replacement", "",
		},
		{
			// IsRetired reports false for an unknown ID, so the test would
			// still call the model after its shutdown.
			"unknown ID fails", "unknown-model", shutdownDay, "",
			"unknown-model is not an offering ID in the openai catalog, so SkipIfRetired could never skip it",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			reason, err := retiredReason(cat, tt.offeringID, tt.asOf)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				return
			}

			require.NoError(t, err)
			assert.Equal(t, tt.wantReason, reason)
		})
	}
}

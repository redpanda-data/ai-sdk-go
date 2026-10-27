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
	"fmt"
	"testing"
	"time"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

// SkipIfRetired skips t from the date the catalog records as the offering's
// shutdown, naming its replacement, so a test about a specific model keeps
// testing it until the vendor turns it off. It fails t when the catalog
// doesn't know the offering, because the skip could then never fire.
func SkipIfRetired(t *testing.T, cat *catalog.Catalog, offeringID string) {
	t.Helper()

	reason, err := retiredReason(cat, offeringID, catalog.Today())
	if err != nil {
		t.Fatal(err)
	}

	if reason != "" {
		t.Skip(reason)
	}
}

// retiredReason returns why a test about the offering should skip as of asOf,
// or "" when the offering is still served.
func retiredReason(cat *catalog.Catalog, offeringID string, asOf time.Time) (string, error) {
	o, ok := cat.Lookup(offeringID)
	if !ok {
		return "", fmt.Errorf("%s is not an offering ID in the %s catalog, so SkipIfRetired could never skip it", offeringID, cat.Provider())
	}

	if !cat.At(asOf).IsRetired(offeringID) {
		return "", nil
	}

	reason := fmt.Sprintf("%s retired on %s, per the %s catalog", offeringID, o.Life.Retires.Format(time.DateOnly), cat.Provider())

	if o.Life.ReplacedBy == "" {
		return reason + ", which records no replacement", nil
	}

	return reason + "; its replacement is " + o.Life.ReplacedBy, nil
}

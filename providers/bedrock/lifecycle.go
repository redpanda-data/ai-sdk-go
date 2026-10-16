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

package bedrock

import (
	"slices"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

// LifecycleForRegion returns an exact catalog offering's lifecycle with any
// region-specific AWS notice applied. It does not check model availability or
// account access or validate profile/source-region compatibility. Pass the exact
// AWS source-region ID. An unknown offering returns false. Unlisted regions retain
// the offering's unscoped lifecycle; absence of a date is not a promise of
// indefinite availability. Catalog().At cannot apply region-specific notices.
func LifecycleForRegion(modelID, region string) (catalog.Lifecycle, bool) {
	offering, ok := Catalog().Lookup(modelID)
	if !ok {
		return catalog.Lifecycle{}, false
	}

	life := offering.Life
	if offering.Model == catalog.ModelClaudeSonnet45 && slices.Contains(sonnet45LegacyRegions, region) {
		life.Deprecated = sonnet45LegacyDate
		life.Retires = sonnet45EOLDate
	}

	return life, true
}

// AWS's notice covers only these commercial regions. The global catalog's
// Life remains unscoped, rather than claiming an EOL in GovCloud or elsewhere.
// Extended access starts 2027-01-08; no future surcharge is invented here.
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle-legacy.html
var sonnet45LegacyRegions = strings.Fields(`
 af-south-1 ap-east-2 ap-northeast-1 ap-northeast-2 ap-northeast-3
 ap-south-1 ap-south-2 ap-southeast-1 ap-southeast-2 ap-southeast-3
 ap-southeast-4 ap-southeast-5 ap-southeast-6 ap-southeast-7
 ca-central-1 ca-west-1 eu-central-1 eu-central-2 eu-north-1
 eu-south-1 eu-south-2 eu-west-1 eu-west-2 eu-west-3
 il-central-1 mx-central-1 sa-east-1 us-east-1 us-east-2 us-west-1 us-west-2
`)

var (
	sonnet45LegacyDate = catalog.MustDate("2026-10-08")
	sonnet45EOLDate    = catalog.MustDate("2027-04-08")
)

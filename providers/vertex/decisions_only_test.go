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

package vertex

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

// TestNewModelRefusesDecisionsOnly feeds the constructor NewModel delegates to
// every catalogued offering turned decisions-only.
func TestNewModelRefusesDecisionsOnly(t *testing.T) {
	t.Parallel()

	for _, offering := range Catalog().All() {
		offering.WireAPIs = []catalog.WireAPI{{Protocol: catalog.WireProtocolOpenAIDecisions, Transport: catalog.WireTransportDirect}}
		_, err := (&Provider{}).newModelFromOffering(offering.ID, offering)
		require.ErrorIs(t, err, catalog.ErrDecisionsOnly, offering.ID)
	}
}

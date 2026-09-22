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
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

func TestExpandFamiliesRequiresPublisher(t *testing.T) {
	t.Parallel()

	declare := func(publisher string) func() {
		return func() {
			expandFamilies([]family{{
				BareID:        "anthropic.claude-opus-5",
				Publisher:     publisher,
				Model:         catalog.ModelID("claude-opus-5"),
				DisplayName:   "Claude Opus 5",
				BareInvokable: true,
			}})
		}
	}

	want := func(publisher string) string {
		return fmt.Sprintf("bedrock: family anthropic.claude-opus-5 declares Publisher %q, "+
			"want the vendor namespace of its BareID", publisher)
	}

	assert.PanicsWithValue(t, want(""), declare(""))
	assert.PanicsWithValue(t, want("amazon"), declare("amazon"))
	assert.NotPanics(t, declare("anthropic"))
}

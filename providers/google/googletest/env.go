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

package googletest

import (
	"os"
	"testing"
)

// GetAPIKeyOrSkipTest returns the API key if present,
// otherwise it skips the test.
func GetAPIKeyOrSkipTest(t *testing.T) string {
	t.Helper()

	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		t.Skip("skipping test: GOOGLE_API_KEY not set")
	}

	return key
}

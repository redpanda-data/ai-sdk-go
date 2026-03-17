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

package mcptest

import (
	"os"
	"testing"
)

// GetContext7APIKeyOrSkipTest returns the Context7 API key if present,
// otherwise it skips the test.
func GetContext7APIKeyOrSkipTest(t *testing.T) string {
	t.Helper()

	key := os.Getenv("CONTEXT7_API_KEY")
	if key == "" {
		t.Skip("skipping test: CONTEXT7_API_KEY not set")
	}

	return key
}

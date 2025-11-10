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

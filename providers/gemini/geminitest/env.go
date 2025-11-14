package geminitest

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

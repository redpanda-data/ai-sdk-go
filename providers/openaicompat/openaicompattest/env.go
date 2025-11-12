package openaicompattest

import (
	"os"
	"testing"
)

// GetAPIKeyOrSkipTest returns the API key if present,
// otherwise it skips the test.
func GetAPIKeyOrSkipTest(t *testing.T) string {
	t.Helper()

	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		t.Skip("skipping test: OPENAI_API_KEY not set")
	}

	return key
}

// GetDeepSeekAPIKeyOrSkipTest returns the DeepSeek API key if present,
// otherwise it skips the test.
func GetDeepSeekAPIKeyOrSkipTest(t *testing.T) string {
	t.Helper()

	key := os.Getenv("DEEPSEEK_API_KEY")
	if key == "" {
		t.Skip("DEEPSEEK_API_KEY not set, skipping DeepSeek test")
	}

	return key
}

// GetDeepSeekBaseURL returns the DeepSeek base URL from environment or the default.
func GetDeepSeekBaseURL() string {
	if baseURL := os.Getenv("DEEPSEEK_BASE_URL"); baseURL != "" {
		return baseURL
	}

	return DeepSeekDefaultBaseURL
}

// GetDeepSeekModel returns the DeepSeek model name from environment or the default.
func GetDeepSeekModel(defaultModel string) string {
	if model := os.Getenv("DEEPSEEK_MODEL"); model != "" {
		return model
	}

	return defaultModel
}

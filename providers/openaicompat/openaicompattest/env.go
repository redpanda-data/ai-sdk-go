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

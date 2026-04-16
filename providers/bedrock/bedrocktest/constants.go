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

package bedrocktest

const (
	// TestModelName is the model to use for integration tests.
	TestModelName = "anthropic.claude-sonnet-4-5-20250929-v1:0"
	// TestReasoningModelName is the model for reasoning tests.
	// Uses a model with forced (non-adaptive) extended thinking. Opus 4.7
	// supports adaptive thinking only, which Bedrock's WithThinking helper
	// cannot yet request.
	TestReasoningModelName = "anthropic.claude-opus-4-5-20251101-v1:0"
	// TestRegion is the default AWS region for tests, overridable via AWS_REGION.
	TestRegion = "us-east-1"
)

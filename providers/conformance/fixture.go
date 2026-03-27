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

package conformance

import (
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Fixture defines the interface that each provider must implement
// to participate in conformance testing. This allows the generic test suite
// to test any provider implementation against the llm.Model interface.
type Fixture interface {
	// Name returns the provider name for test reporting
	Name() string

	// NewStandardModel returns a fresh model instance suitable for standard testing
	// (basic generation, streaming, tools, structured output).
	NewStandardModel(t *testing.T) llm.Model

	// NewReasoningModel returns a fresh model instance that supports reasoning capabilities.
	// Implementations should call t.Skip when reasoning models are not available.
	NewReasoningModel(t *testing.T) llm.Model

	// Models returns the list of all models available from this provider
	// for discovery testing. Returns nil or empty slice to skip model discovery tests.
	Models() []llm.ModelDiscoveryInfo

	// NewModel creates a new model instance by name for testing.
	// Used by the TestAllSupportedModels test to verify all models work.
	// Returns an error if the model cannot be created.
	NewModel(modelName string) (llm.Model, error)
}

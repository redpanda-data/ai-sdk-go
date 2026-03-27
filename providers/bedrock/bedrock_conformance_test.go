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

package bedrock_test

import (
	"context"
	"os"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock/bedrocktest"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
)

// BedrockFixture implements the conformance.Fixture interface for the Bedrock provider.
type BedrockFixture struct {
	provider *bedrock.Provider
}

// NewBedrockFixture creates a new Bedrock test fixture.
func NewBedrockFixture(t *testing.T) *BedrockFixture {
	t.Helper()

	bedrocktest.SkipUnlessAWSCredentials(t)

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = bedrocktest.TestRegion
	}

	ctx := context.Background()

	provider, err := bedrock.NewProvider(ctx, bedrock.WithRegion(region))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return &BedrockFixture{
		provider: provider,
	}
}

func (f *BedrockFixture) Name() string {
	return "Bedrock"
}

func (f *BedrockFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(bedrocktest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	return retry.WrapModel(model)
}

func (f *BedrockFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(bedrocktest.TestReasoningModelName, bedrock.WithThinking(8192))
	if err != nil {
		t.Skipf("No reasoning model available: %v", err)
		return nil
	}

	return retry.WrapModel(model)
}

func (f *BedrockFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *BedrockFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestBedrockConformance_Integration runs the generic conformance test suite for the Bedrock provider.
func TestBedrockConformance_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewBedrockFixture(t)
	testsuite.Run(t, conformance.NewSuite(fixture))
}

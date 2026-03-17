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

	"github.com/stretchr/testify/suite"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock/bedrocktest"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
)

// BedrockFixture implements the conformance.Fixture interface for the Bedrock provider.
type BedrockFixture struct {
	provider       *bedrock.Provider
	standardModel  llm.Model
	reasoningModel llm.Model
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

	standardModel, err := provider.NewModel(bedrocktest.TestModelName)
	if err != nil {
		t.Fatalf("Failed to create standard model: %v", err)
	}

	reasoningModel, err := provider.NewModel(bedrocktest.TestReasoningModelName,
		bedrock.WithThinking(8192),
	)
	if err != nil {
		t.Logf("Failed to create reasoning model: %v", err)
	}

	return &BedrockFixture{
		provider:       provider,
		standardModel:  standardModel,
		reasoningModel: reasoningModel,
	}
}

func (f *BedrockFixture) Name() string {
	return "Bedrock"
}

func (f *BedrockFixture) StandardModel() llm.Model {
	return f.standardModel
}

func (f *BedrockFixture) ReasoningModel() llm.Model {
	return f.reasoningModel
}

func (f *BedrockFixture) Models() []llm.ModelDiscoveryInfo {
	return f.provider.Models()
}

func (f *BedrockFixture) NewModel(modelName string) (llm.Model, error) {
	return f.provider.NewModel(modelName)
}

// TestBedrockConformance runs the generic conformance test suite for the Bedrock provider.
//
//nolint:paralleltest // Test suite manages its own lifecycle
func TestBedrockConformance(t *testing.T) {
	fixture := NewBedrockFixture(t)
	suite.Run(t, conformance.NewSuite(fixture))
}

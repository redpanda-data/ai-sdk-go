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
	"errors"
	"os"
	"strings"
	"testing"
	"time"

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
	region   string
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
		region:   region,
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
	all := f.provider.Models()

	// The catalog exposes one entry per inference profile (global. and each
	// geo). Conformance runs in a single AWS region with limited IAM, so we
	// only iterate the variants reachable from the test region: the matching
	// geo profile (e.g. "us." when running in us-east-1). Other geo profiles
	// fail with ValidationException (cross-geography from a single entry-point
	// region) and the global. profile resolves to foundation-model ARNs that
	// the CI's SCP explicitly denies. Per-variant catalog correctness is
	// covered by unit tests in provider_test.go.
	geoPrefix := bedrock.InferenceProfileRegion(f.region) + "."

	filtered := make([]llm.ModelDiscoveryInfo, 0, len(all))

	for _, m := range all {
		// Fable 5 requires Bedrock provider data sharing. Keep the generic
		// all-model conformance loop on models that CI can fully generate with;
		// the dedicated Fable integration below still verifies Bedrock wiring by
		// accepting either a successful response or AWS's explicit Fable access gate.
		if modelRequiresProviderDataSharing(m) {
			continue
		}

		if strings.HasPrefix(m.Name, geoPrefix) {
			filtered = append(filtered, m)
		}
	}

	return filtered
}

func modelRequiresProviderDataSharing(model llm.ModelDiscoveryInfo) bool {
	return model.Metadata[bedrock.ModelMetadataRequiresProviderDataSharing] == "true"
}

func TestModelRequiresProviderDataSharing(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		model llm.ModelDiscoveryInfo
		want  bool
	}{
		{
			name: "Fable 5",
			model: llm.ModelDiscoveryInfo{
				Name: bedrock.ModelClaudeFable5US,
				Metadata: map[string]string{
					bedrock.ModelMetadataRequiresProviderDataSharing: "true",
				},
			},
			want: true,
		},
		{
			name:  "other Claude",
			model: llm.ModelDiscoveryInfo{Name: bedrock.ModelClaudeSonnet46US},
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := modelRequiresProviderDataSharing(tt.model)
			if got != tt.want {
				t.Fatalf("modelRequiresProviderDataSharing(%q) = %v, want %v", tt.model.Name, got, tt.want)
			}
		})
	}
}

func TestIsProviderDataSharingGate(t *testing.T) {
	t.Parallel()

	err := &llm.ProviderError{
		Base:    llm.ErrInvalidInput,
		Code:    "ValidationException",
		Message: "provider rejected request because data retention is not enabled",
	}

	if !isProviderDataSharingGate(err) {
		t.Fatal("expected provider data sharing gate to be detected")
	}
}

func TestIsFableAccessGate(t *testing.T) {
	t.Parallel()

	err := &llm.ProviderError{
		Base:    llm.ErrAPICall,
		Code:    "AccessDeniedException",
		Message: "access denied",
	}

	if !isFableAccessGate(err) {
		t.Fatal("expected Fable access gate to be detected")
	}
}

func TestBedrockFable5Invocation_Integration(t *testing.T) {
	t.Parallel()

	bedrocktest.SkipUnlessAWSCredentials(t)

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = bedrocktest.TestRegion
	}

	provider, err := bedrock.NewProvider(context.Background(), bedrock.WithRegion(region))
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	model, err := provider.NewModel(bedrock.ModelClaudeFable5, bedrock.WithMaxTokens(16))
	if err != nil {
		t.Fatalf("Failed to create Fable 5 model: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	resp, err := model.Generate(ctx, &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply with ok.")),
		},
	})
	if err != nil {
		if isFableAccessGate(err) {
			t.Skipf("Fable 5 reached Bedrock but account is not enabled for Fable access: %v", err)
		}

		t.Fatalf("Fable 5 Bedrock invocation failed: %v", err)
	}

	if resp == nil {
		t.Fatal("Fable 5 Bedrock invocation returned nil response")
	}
}

func TestBedrockAdaptiveThinking_Integration(t *testing.T) {
	t.Parallel()

	fixture := NewBedrockFixture(t)

	model, err := fixture.provider.NewModel(
		bedrock.ModelClaudeOpus47,
		bedrock.WithAdaptiveThinking(bedrock.EffortLow),
	)
	if err != nil {
		t.Fatalf("Failed to create adaptive thinking model: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()

	resp, err := retry.WrapModel(model).Generate(ctx, &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply with ok.")),
		},
	})
	if err != nil {
		t.Fatalf("Adaptive thinking Bedrock invocation failed: %v", err)
	}

	if resp == nil || resp.TextContent() == "" {
		t.Fatal("Adaptive thinking Bedrock invocation returned no text")
	}
}

func isProviderDataSharingGate(err error) bool {
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		return false
	}

	if providerErr.Code != "ValidationException" {
		return false
	}

	message := strings.ToLower(providerErr.Message)

	return strings.Contains(message, "data retention") || strings.Contains(message, "provider_data")
}

func isAccessDeniedGate(providerErr *llm.ProviderError) bool {
	return providerErr.Code == "AccessDeniedException"
}

func isFableAccessGate(err error) bool {
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		return false
	}

	return isProviderDataSharingGate(err) || isAccessDeniedGate(providerErr)
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

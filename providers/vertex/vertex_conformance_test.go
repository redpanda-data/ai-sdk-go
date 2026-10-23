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

package vertex_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
)

const (
	// credentialsJSONEnv names the variable holding the service-account key.
	credentialsJSONEnv = "VERTEX_CREDENTIALS_JSON" //nolint:gosec // a variable name, not a credential

	// testProject is the project set up for live Vertex tests.
	testProject = "cloudv2-testing-project"

	geminiTestModel          = vertex.ModelGemini36Flash
	claudeTestModel          = vertex.ModelClaudeHaiku45
	claudeTestReasoningModel = vertex.ModelClaudeSonnet5
)

// grantedClaudeModels are the Claude models enabled in testProject's Model
// Garden. Vertex answers any other Claude model with a 404 that the shared
// suite does not read as an access gate, so the Claude catalog holds only
// these, and NewModel refuses any other with a named reason.
var grantedClaudeModels = map[string]bool{
	vertex.ModelClaudeHaiku45: true,
	vertex.ModelClaudeSonnet5: true,
}

// VertexFixture implements the conformance.Fixture interface for one
// publisher's half of the Vertex provider: Gemini or Claude.
type VertexFixture struct {
	provider  *vertex.Provider
	publisher catalog.Publisher
}

func newVertexFixture(provider *vertex.Provider, publisher catalog.Publisher) *VertexFixture {
	return &VertexFixture{provider: provider, publisher: publisher}
}

// newLiveProvider returns a provider at the global location that signs its
// calls with the service-account key in credentialsJSONEnv. It skips the
// test when the key is not set, as it is not in pull-request CI.
func newLiveProvider(t *testing.T) *vertex.Provider {
	t.Helper()

	key := os.Getenv(credentialsJSONEnv)
	if key == "" {
		t.Skipf("skipping: %s is not set", credentialsJSONEnv)
	}

	creds, err := google.CredentialsFromJSONWithType(context.Background(), []byte(key), google.ServiceAccount, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		// The error can quote the key, so it is not printed.
		t.Fatalf("%s does not hold a usable service-account key", credentialsJSONEnv)
	}

	client := oauth2.NewClient(context.Background(), creds.TokenSource)
	client.Timeout = 2 * time.Minute

	provider, err := vertex.NewProvider(context.Background(),
		vertex.WithProject(testProject),
		vertex.WithLocation(vertex.LocationGlobal),
		vertex.WithHTTPClient(client),
	)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return provider
}

func (f *VertexFixture) Name() string {
	if f.publisher == catalog.PublisherAnthropic {
		return "Vertex Claude"
	}

	return "Vertex Gemini"
}

func (f *VertexFixture) NewStandardModel(t *testing.T) llm.Model {
	t.Helper()

	if f.publisher == catalog.PublisherGoogle {
		// Gemini 3 models think on every call, so the standard model sets an
		// effort, as the Gemini-API fixture does.
		return f.mustNewModel(t, geminiTestModel, vertex.WithReasoningEffort(vertex.ReasoningEffortLow))
	}

	model := f.mustNewModel(t, claudeTestModel)

	// The suite runs structured outputs on the standard model, and an
	// organization policy may deny that Claude feature on the project.
	if strings.HasSuffix(t.Name(), "/TestStructuredOutputs") {
		skipIfStructuredOutputsDenied(t, model)
	}

	return model
}

func (f *VertexFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	if f.publisher == catalog.PublisherGoogle {
		return f.mustNewModel(t, geminiTestModel, vertex.WithReasoningEffort(vertex.ReasoningEffortLow))
	}

	return f.mustNewModel(t, claudeTestReasoningModel, vertex.WithReasoningEffort(vertex.ReasoningEffortMedium))
}

func (f *VertexFixture) Catalog() *catalog.Catalog {
	all := f.provider.Catalog().All()
	entries := make([]catalog.Entry, 0, len(all))

	for _, o := range all {
		if o.Facts().Publisher != f.publisher {
			continue
		}
		// The sweep calls only the newest model of each series, so an
		// ungranted newer Claude model would hide a granted older one.
		if f.isUngranted(o.ID) {
			continue
		}

		entries = append(entries, o.Entry)
	}

	return catalog.MustNew(f.provider.Name(), entries)
}

func (f *VertexFixture) NewModel(modelName string) (llm.Model, error) {
	if f.isUngranted(modelName) {
		return nil, fmt.Errorf("vertex: %s is not granted in %s", modelName, testProject)
	}

	return f.provider.NewModel(modelName)
}

// isUngranted reports whether f is the Claude half and modelName is not one
// of the models testProject has enabled. It does not check what publisher
// modelName belongs to.
func (f *VertexFixture) isUngranted(modelName string) bool {
	return f.publisher == catalog.PublisherAnthropic && !grantedClaudeModels[modelName]
}

// ungrantedModels lists every model of this half that Catalog leaves out.
func (f *VertexFixture) ungrantedModels() []string {
	var names []string

	for _, o := range f.provider.Catalog().All() {
		if o.Facts().Publisher == f.publisher && f.isUngranted(o.ID) {
			names = append(names, o.ID)
		}
	}

	return names
}

// skipUngranted runs one skipped subtest per model Catalog leaves out, so
// the test output names each one and why.
func (f *VertexFixture) skipUngranted(t *testing.T) {
	t.Helper()

	for _, modelName := range f.ungrantedModels() {
		t.Run("ungranted_"+modelName, func(t *testing.T) {
			t.Parallel()

			_, err := f.NewModel(modelName)
			t.Skipf("Skipping model %s: %v", modelName, err)
		})
	}
}

func (f *VertexFixture) mustNewModel(t *testing.T, modelName string, opts ...vertex.Option) llm.Model {
	t.Helper()

	model, err := f.provider.NewModel(modelName, opts...)
	if err != nil {
		t.Fatalf("Failed to create model %s: %v", modelName, err)
	}

	return retry.WrapModel(model)
}

func newOfflineProvider(t *testing.T) *vertex.Provider {
	t.Helper()

	provider, err := vertex.NewProvider(context.Background(),
		vertex.WithProject(testProject),
		vertex.WithLocation(vertex.LocationGlobal),
		vertex.WithHTTPClient(http.DefaultClient),
	)
	require.NoError(t, err)

	return provider
}

func TestVertexFixtureCatalog_OnePublisherHalf(t *testing.T) {
	t.Parallel()

	provider := newOfflineProvider(t)

	claude := newVertexFixture(provider, catalog.PublisherAnthropic).Catalog()
	require.NotNil(t, claude)

	_, ok := claude.Lookup(vertex.ModelClaudeSonnet5)
	assert.True(t, ok, "Claude half must hold Sonnet 5")
	_, ok = claude.Lookup(vertex.ModelClaudeOpus55)
	assert.False(t, ok, "Claude half must not hold ungranted Opus 5.5")
	_, ok = claude.Lookup(vertex.ModelGemini36Flash)
	assert.False(t, ok, "Claude half must not hold Gemini 3.6 Flash")

	sweep := claude.Now().Current()
	current := make([]string, 0, len(sweep))

	for _, o := range sweep {
		current = append(current, o.ID)
	}

	assert.ElementsMatch(t, []string{vertex.ModelClaudeHaiku45, vertex.ModelClaudeSonnet5}, current,
		"the sweep must reach every granted Claude model")

	gemini := newVertexFixture(provider, catalog.PublisherGoogle).Catalog()
	require.NotNil(t, gemini)

	_, ok = gemini.Lookup(vertex.ModelGemini36Flash)
	assert.True(t, ok, "Gemini half must hold Gemini 3.6 Flash")
	_, ok = gemini.Lookup(vertex.ModelClaudeSonnet5)
	assert.False(t, ok, "Gemini half must not hold Sonnet 5")
}

func TestVertexFixtureNewModel_RefusesUngrantedClaude(t *testing.T) {
	t.Parallel()

	provider := newOfflineProvider(t)
	claude := newVertexFixture(provider, catalog.PublisherAnthropic)

	for _, granted := range []string{vertex.ModelClaudeHaiku45, vertex.ModelClaudeSonnet5} {
		_, err := claude.NewModel(granted)
		require.NoError(t, err, granted)
	}

	_, err := claude.NewModel(vertex.ModelClaudeOpus55)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not granted in cloudv2-testing-project")

	gemini := newVertexFixture(provider, catalog.PublisherGoogle)
	_, err = gemini.NewModel(vertex.ModelGemini36Flash)
	require.NoError(t, err)
}

func TestVertexFixtureUngrantedModels(t *testing.T) {
	t.Parallel()

	provider := newOfflineProvider(t)

	ungranted := newVertexFixture(provider, catalog.PublisherAnthropic).ungrantedModels()
	assert.Contains(t, ungranted, vertex.ModelClaudeOpus55)
	assert.Contains(t, ungranted, vertex.ModelClaudeSonnet45, "an older ungranted model must be named too")
	assert.NotContains(t, ungranted, vertex.ModelClaudeHaiku45)
	assert.NotContains(t, ungranted, vertex.ModelClaudeSonnet5)

	assert.Empty(t, newVertexFixture(provider, catalog.PublisherGoogle).ungrantedModels())
}

// skipIfStructuredOutputsDenied skips the test when the project's
// constraints/vertexai.allowedPartnerModelFeatures policy denies Claude
// structured outputs, which it does by default. A project
// granted an exception runs the test.
func skipIfStructuredOutputsDenied(t *testing.T, model llm.Model) {
	t.Helper()

	_, err := model.Generate(t.Context(), &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Reply with ok.")),
		},
		ResponseFormat: &llm.ResponseFormat{
			Type: llm.ResponseFormatJSONSchema,
			JSONSchema: &llm.JSONSchema{
				Name:   "reply",
				Schema: []byte(`{"type":"object","properties":{"reply":{"type":"string"}},"required":["reply"],"additionalProperties":false}`),
			},
		},
	})
	if isPartnerFeaturePolicyDenial(err) {
		t.Skipf("Claude structured outputs are denied by %s's allowedPartnerModelFeatures policy: %v", testProject, err)
	}
}

// isPartnerFeaturePolicyDenial reports whether err is Vertex refusing a
// Claude feature under the constraints/vertexai.allowedPartnerModelFeatures
// organization policy.
func isPartnerFeaturePolicyDenial(err error) bool {
	var providerErr *llm.ProviderError
	if !errors.As(err, &providerErr) {
		return false
	}

	return strings.Contains(strings.ToLower(providerErr.Message), "allowedpartnermodelfeatures")
}

func TestIsPartnerFeaturePolicyDenial(t *testing.T) {
	t.Parallel()

	denied := &llm.ProviderError{
		Base:    llm.ErrInvalidInput,
		Code:    "invalid_request_error",
		Message: "Request violates constraint constraints/vertexai.allowedPartnerModelFeatures",
	}
	assert.True(t, isPartnerFeaturePolicyDenial(fmt.Errorf("generate: %w", denied)))

	other := &llm.ProviderError{
		Base:    llm.ErrInvalidInput,
		Code:    "invalid_request_error",
		Message: "output_config.format.schema: invalid JSON schema",
	}
	assert.False(t, isPartnerFeaturePolicyDenial(other))
	assert.False(t, isPartnerFeaturePolicyDenial(errors.New("allowedPartnerModelFeatures")))
}

// denyingTransport answers every request the way Vertex AI does when the
// allowedPartnerModelFeatures policy denies a Claude feature.
type denyingTransport struct{}

func (denyingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	body := `{"type":"error","error":{"type":"invalid_request_error","message":"Request violates constraint constraints/vertexai.allowedPartnerModelFeatures"}}`

	return &http.Response{
		StatusCode: http.StatusBadRequest,
		Status:     "400 Bad Request",
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
		Request:    req,
	}, nil
}

func TestVertexClaudeStandardModel_SkipsStructuredOutputsWhenDenied(t *testing.T) {
	t.Parallel()

	provider, err := vertex.NewProvider(context.Background(),
		vertex.WithProject(testProject),
		vertex.WithLocation(vertex.LocationGlobal),
		vertex.WithHTTPClient(&http.Client{Transport: denyingTransport{}}),
	)
	require.NoError(t, err)

	fixture := newVertexFixture(provider, catalog.PublisherAnthropic)

	// NewStandardModel matches the suite's method by name, so this stops
	// the build if the suite renames it.
	_ = (*conformance.Suite).TestStructuredOutputs

	for name, wantSkip := range map[string]bool{"TestStructuredOutputs": true, "TestGenerate": false} {
		var skipped bool

		// The subtest is parallel, so it ends only after this function returns.
		t.Cleanup(func() {
			assert.Equal(t, wantSkip, skipped, "a denied policy must skip only the structured-outputs test: %s", name)
		})

		t.Run(name, func(t *testing.T) {
			t.Parallel()

			defer func() { skipped = t.Skipped() }()

			fixture.NewStandardModel(t)
		})
	}
}

// TestVertexConformance_Integration runs the generic conformance suite on
// both halves of the Vertex provider against live Vertex AI. It needs
// VERTEX_CREDENTIALS_JSON, which only the nightly workflow sets.
func TestVertexConformance_Integration(t *testing.T) {
	t.Parallel()

	for _, publisher := range []catalog.Publisher{catalog.PublisherGoogle, catalog.PublisherAnthropic} {
		t.Run(string(publisher), func(t *testing.T) {
			t.Parallel()

			fixture := newVertexFixture(newLiveProvider(t), publisher)
			fixture.skipUngranted(t)
			testsuite.Run(t, conformance.NewSuite(fixture))
		})
	}
}

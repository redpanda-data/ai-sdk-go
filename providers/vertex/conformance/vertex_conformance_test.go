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

package conformance_test

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/oauth2"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/internal/testsuite"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/plugins/retry"
	"github.com/redpanda-data/ai-sdk-go/providers/conformance"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex"
	"github.com/redpanda-data/ai-sdk-go/providers/vertex/vertextest"
)

const (
	// accessTokenEnv names the variable holding the OAuth 2.0 access token.
	accessTokenEnv = "VERTEX_ACCESS_TOKEN" //nolint:gosec // a variable name, not a credential

	// baseURLEnv names the variable holding the URL of an auth proxy.
	baseURLEnv = "VERTEX_BASE_URL"

	// projectEnv names the variable that overrides the test project. The
	// override project must have every model in grantedClaudeModels enabled,
	// or the sweep fails on a 404 for the one it lacks.
	projectEnv = "VERTEX_TEST_PROJECT"

	geminiTestModel          = vertextest.TestGeminiModelName
	claudeTestModel          = vertextest.TestClaudeModelName
	claudeTestReasoningModel = vertextest.TestClaudeReasoningModelName

	// claudeThinkingBudget is the Claude reasoning model's thinking budget in
	// tokens. Haiku 4.5 takes a budget, not an effort level.
	claudeThinkingBudget = 4096
)

// testProject is the project set up for live Vertex tests.
var testProject = projectFromEnv()

func projectFromEnv() string {
	if project := os.Getenv(projectEnv); project != "" {
		return project
	}

	return "cloudv2-testing-project"
}

// grantedClaudeModels are the Claude models enabled in testProject's Model
// Garden. Vertex answers any other Claude model with a 404.
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

func newLiveProvider(t *testing.T) *vertex.Provider {
	t.Helper()

	opts := []vertex.ProviderOption{
		vertex.WithProject(testProject),
		vertex.WithLocation(vertex.LocationGlobal),
	}

	if baseURL := os.Getenv(baseURLEnv); baseURL != "" {
		opts = append(opts,
			vertex.WithBaseURL(baseURL),
			vertex.WithHTTPClient(&http.Client{Timeout: 5 * time.Minute}),
		)
	} else {
		token := os.Getenv(accessTokenEnv)
		if token == "" {
			t.Skipf("skipping: %s and %s are not set", baseURLEnv, accessTokenEnv)
		}

		opts = append(opts, vertex.WithHTTPClient(newTokenClient(token)))
	}

	provider, err := vertex.NewProvider(context.Background(), opts...)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	return provider
}

// newTokenClient returns a client that sends token as a bearer token. The
// token is not refreshed, so it must outlast the test.
func newTokenClient(token string) *http.Client {
	client := oauth2.NewClient(context.Background(), oauth2.StaticTokenSource(&oauth2.Token{AccessToken: token}))
	client.Timeout = 5 * time.Minute

	return client
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
		// Gemini 3 models think on every call.
		return f.mustNewModel(t, geminiTestModel, vertex.WithReasoningEffort(vertex.ReasoningEffortLow))
	}

	model := f.mustNewModel(t, claudeTestModel)

	if inStructuredOutputsTest(t) {
		skipIfStructuredOutputsDenied(t, model)
	}

	return model
}

func (f *VertexFixture) NewReasoningModel(t *testing.T) llm.Model {
	t.Helper()

	if f.publisher == catalog.PublisherGoogle {
		// Gemini 3.8 Flash returns no thought summaries at the low level.
		return f.mustNewModel(t, geminiTestModel, vertex.WithReasoningEffort(vertex.ReasoningEffortMedium))
	}

	return f.mustNewModel(t, claudeTestReasoningModel, vertex.WithThinkingBudget(claudeThinkingBudget))
}

func (f *VertexFixture) Catalog() *catalog.Catalog {
	return f.halfCatalog(f.provider.Catalog())
}

func (f *VertexFixture) NewModel(modelName string) (llm.Model, error) {
	if f.isUngranted(modelName) {
		return nil, errUngranted(modelName)
	}

	return f.provider.NewModel(modelName)
}

// halfCatalog keeps the offerings of full that this half can call.
func (f *VertexFixture) halfCatalog(full *catalog.Catalog) *catalog.Catalog {
	all := full.All()
	entries := make([]catalog.Entry, 0, len(all))

	for _, o := range all {
		if o.Facts().Publisher != f.publisher {
			continue
		}

		if f.isUngranted(o.ID) {
			continue
		}

		e := o.Entry
		// MustNew refuses a replacement outside the subset.
		if f.isUngranted(e.Life.ReplacedBy) {
			e.Life.ReplacedBy = ""
		}

		entries = append(entries, e)
	}

	return catalog.MustNew(f.provider.Name(), entries)
}

// isUngranted does not check which publisher modelName belongs to.
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

// skipUngranted runs one skipped subtest per model of this half that Catalog
// leaves out.
func (f *VertexFixture) skipUngranted(t *testing.T) {
	t.Helper()

	for _, modelName := range f.ungrantedModels() {
		t.Run("ungranted_"+modelName, func(t *testing.T) {
			t.Parallel()

			t.Skipf("Skipping model %s: %v", modelName, errUngranted(modelName))
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

// structuredOutputsTest is the suite subtest NewStandardModel matches by
// name. The method expression below stops the build if the suite renames it.
const structuredOutputsTest = "TestStructuredOutputs"

var _ = (*conformance.Suite).TestStructuredOutputs

// inStructuredOutputsTest reports whether t is the suite's structured-outputs
// subtest or runs under it.
func inStructuredOutputsTest(t *testing.T) bool {
	t.Helper()

	return strings.Contains(t.Name()+"/", "/"+structuredOutputsTest+"/")
}

func errUngranted(modelName string) error {
	return fmt.Errorf("vertex: %s is not granted in %s", modelName, testProject)
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
	_, ok = claude.Lookup(vertex.ModelGemini38Flash)
	assert.False(t, ok, "Claude half must not hold Gemini 3.8 Flash")

	// A fixed date keeps this test from failing once Haiku 4.5 retires.
	sweep := claude.At(catalog.MustDate("2026-11-15")).Current()
	current := make([]string, 0, len(sweep))

	for _, o := range sweep {
		current = append(current, o.ID)
	}

	assert.ElementsMatch(t, []string{vertex.ModelClaudeHaiku45, vertex.ModelClaudeSonnet5}, current,
		"the sweep must reach every granted Claude model")

	gemini := newVertexFixture(provider, catalog.PublisherGoogle).Catalog()
	require.NotNil(t, gemini)

	_, ok = gemini.Lookup(vertex.ModelGemini38Flash)
	assert.True(t, ok, "Gemini half must hold Gemini 3.8 Flash")
	_, ok = gemini.Lookup(vertex.ModelClaudeSonnet5)
	assert.False(t, ok, "Gemini half must not hold Sonnet 5")
}

func TestVertexFixtureCatalog_BuildsWithUngrantedReplacement(t *testing.T) {
	t.Parallel()

	provider := newOfflineProvider(t)

	entries := make([]catalog.Entry, 0, provider.Catalog().Len())
	for _, o := range provider.Catalog().All() {
		switch o.ID {
		case vertex.ModelClaudeSonnet5:
			o.Life.ReplacedBy = vertex.ModelClaudeSonnet55
		case vertex.ModelClaudeHaiku45:
			o.Life.ReplacedBy = vertex.ModelClaudeSonnet5
		}

		entries = append(entries, o.Entry)
	}

	full := catalog.MustNew(provider.Name(), entries)

	var claude *catalog.Catalog

	require.NotPanics(t, func() {
		claude = newVertexFixture(provider, catalog.PublisherAnthropic).halfCatalog(full)
	})

	_, ok := claude.Lookup(vertex.ModelClaudeSonnet5)
	assert.True(t, ok, "Claude half must still hold Sonnet 5")

	haiku, ok := claude.Lookup(vertex.ModelClaudeHaiku45)
	require.True(t, ok, "Claude half must hold Haiku 4.5")
	assert.Equal(t, vertex.ModelClaudeSonnet5, haiku.Life.ReplacedBy, "a granted replacement must stay")
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
	assert.Contains(t, err.Error(), "not granted in "+testProject)

	gemini := newVertexFixture(provider, catalog.PublisherGoogle)
	_, err = gemini.NewModel(vertex.ModelGemini38Flash)
	require.NoError(t, err)
}

func TestVertexClaudeTestModelsAreGranted(t *testing.T) {
	t.Parallel()

	assert.True(t, grantedClaudeModels[claudeTestModel], "%s is not in grantedClaudeModels", claudeTestModel)
	assert.True(t, grantedClaudeModels[claudeTestReasoningModel], "%s is not in grantedClaudeModels", claudeTestReasoningModel)
}

func TestVertexGeminiTestModelIsInGeminiHalf(t *testing.T) {
	t.Parallel()

	gemini := newVertexFixture(newOfflineProvider(t), catalog.PublisherGoogle).Catalog()
	_, ok := gemini.Lookup(geminiTestModel)
	assert.True(t, ok, "%s is not in the Gemini half", geminiTestModel)
}

func TestVertexFixtureUngrantedModels(t *testing.T) {
	t.Parallel()

	provider := newOfflineProvider(t)

	ungranted := newVertexFixture(provider, catalog.PublisherAnthropic).ungrantedModels()
	assert.Contains(t, ungranted, vertex.ModelClaudeOpus55)
	assert.Contains(t, ungranted, vertex.ModelClaudeSonnet46, "an older ungranted model must be named too")
	assert.NotContains(t, ungranted, vertex.ModelClaudeHaiku45)
	assert.NotContains(t, ungranted, vertex.ModelClaudeSonnet5)
	assert.NotContains(t, ungranted, vertex.ModelGemini38Flash, "the Claude half must not name Gemini models")

	assert.Empty(t, newVertexFixture(provider, catalog.PublisherGoogle).ungrantedModels())
}

// skipIfStructuredOutputsDenied skips the test when the project's
// constraints/vertexai.allowedPartnerModelFeatures policy denies Claude
// structured outputs, which it does by default.
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

// newDenyingClaudeFixture returns the Claude half of a provider whose
// project policy denies structured outputs.
func newDenyingClaudeFixture(t *testing.T) *VertexFixture {
	t.Helper()

	provider, err := vertex.NewProvider(context.Background(),
		vertex.WithProject(testProject),
		vertex.WithLocation(vertex.LocationGlobal),
		vertex.WithHTTPClient(&http.Client{Transport: denyingTransport{}}),
	)
	require.NoError(t, err)

	return newVertexFixture(provider, catalog.PublisherAnthropic)
}

func TestVertexClaudeStandardModel_SkipsStructuredOutputsWhenDenied(t *testing.T) {
	t.Parallel()

	fixture := newDenyingClaudeFixture(t)

	for name, wantSkip := range map[string]bool{structuredOutputsTest: true, "TestGenerate": false} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			// t.Skipf ends the subtest with runtime.Goexit, which still runs
			// deferred calls.
			defer func() {
				assert.Equal(t, wantSkip, t.Skipped(), "a denied policy must skip only the structured-outputs test")
			}()

			fixture.NewStandardModel(t)
		})
	}
}

func TestVertexClaudeStandardModel_SkipsNestedStructuredOutputsWhenDenied(t *testing.T) {
	t.Parallel()

	fixture := newDenyingClaudeFixture(t)

	t.Run(structuredOutputsTest, func(t *testing.T) {
		t.Parallel()

		t.Run("nested", func(t *testing.T) {
			t.Parallel()

			defer func() {
				assert.True(t, t.Skipped(), "a denied policy must skip a subtest of the structured-outputs test")
			}()

			fixture.NewStandardModel(t)
		})
	})
}

// TestVertexConformance_Integration runs the generic conformance suite on
// both halves of the Vertex provider against live Vertex AI.
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

func TestNewTokenClient_SendsBearerToken(t *testing.T) {
	t.Parallel()

	var got string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.Header.Get("Authorization")

		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, server.URL, http.NoBody)
	require.NoError(t, err)

	resp, err := newTokenClient("ya29.test-token").Do(req)
	require.NoError(t, err)
	require.NoError(t, resp.Body.Close())

	assert.Equal(t, "Bearer ya29.test-token", got)
}

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

package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	signerv4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/pricing"
)

func TestIsMantleModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		modelID string
		want    bool
	}{
		{ModelGemma431B, true},
		{ModelGemma426BA4B, true},
		{ModelGemma4E2B, true},
		// Converse-API models are not mantle.
		{ModelClaudeSonnet45US, false},
		{ModelNova2LiteUS, false},
		{ModelMistralLarge3, false},
		// Unknown IDs are not mantle.
		{"google.gemma-4-31b-nonexistent", false},
		{"openai.gpt-5.5", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.modelID, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, IsMantleModel(tt.modelID))
		})
	}
}

func TestMantleModelsAreFlaggedAndBareID(t *testing.T) {
	t.Parallel()

	require.NotEmpty(t, mantleModelIDs)

	for id := range mantleModelIDs {
		t.Run(id, func(t *testing.T) {
			t.Parallel()
			// Mantle models are invoked by their bare ID — they must not carry a
			// geo/global inference-profile prefix (NewModel resolves them as-is).
			assert.False(t, hasRegionPrefix(id),
				"mantle model %s should be a bare ID with no geo prefix", id)

			_, ok := Catalog().Lookup(id)
			assert.True(t, ok, "mantle model %s must be a catalog offering", id)
		})
	}
}

func TestMantleBaseURL(t *testing.T) {
	t.Parallel()

	// Direct: real mantle host per region.
	assert.Equal(t, "https://bedrock-mantle.us-east-1.api.aws/openai/v1", mantleBaseURL("us-east-1", ""))
	assert.Equal(t, "https://bedrock-mantle.eu-central-1.api.aws/openai/v1", mantleBaseURL("eu-central-1", ""))
	// Proxy/gateway mode: route through the supplied base endpoint (trailing slash trimmed).
	assert.Equal(t, "https://proxy.example/openai/v1", mantleBaseURL("us-east-1", "https://proxy.example/"))
}

func TestNewModel_MantleRoutesToBedrockProvider(t *testing.T) {
	t.Parallel()

	// WithNoAuth uses anonymous credentials, so no real AWS credentials are
	// needed to construct the model (it is not invoked here).
	p, err := NewProvider(context.Background(), WithRegion("us-east-1"), WithNoAuth())
	require.NoError(t, err)

	m, err := p.NewModel(ModelGemma431B, WithTemperature(0.5), WithMaxTokens(1024))
	require.NoError(t, err)

	// It reports the Bedrock identity, not "openai".
	assert.Equal(t, llm.ProviderID("aws.bedrock"), m.Provider())
	assert.Equal(t, ModelGemma431B, m.Name())
	// Capabilities/constraints come from the Bedrock catalog entry.
	assert.True(t, m.Capabilities().Reasoning)
	assert.Equal(t, 256000, m.Constraints().MaxInputTokens)
}

func TestNewModel_GPT56ModelsUseMantleCatalog(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithRegion("us-east-1"), WithNoAuth())
	require.NoError(t, err)

	for _, modelID := range []string{
		ModelGPT56Sol,
		ModelGPT56Terra,
		ModelGPT56Luna,
	} {
		t.Run(modelID, func(t *testing.T) {
			t.Parallel()

			m, err := p.NewModel(modelID, WithMaxTokens(128_000))
			require.NoError(t, err)

			assert.True(t, IsMantleModel(modelID))
			assert.Equal(t, llm.ProviderID("aws.bedrock"), m.Provider())
			assert.Equal(t, modelID, m.Name())
			assert.True(t, m.Capabilities().Streaming)
			assert.True(t, m.Capabilities().Tools)
			assert.True(t, m.Capabilities().Reasoning)
			assert.Equal(t, 922_000, m.Constraints().MaxInputTokens)
			assert.Equal(t, 128_000, m.Constraints().MaxOutputTokens)
		})
	}
}

// TestNewModel_GPT6MantleEffortNone checks that NewModel accepts effort none
// on mantle for GPT-6 Sol, whose card lists it, and rejects it for GPT-6.1 Sol.
func TestNewModel_GPT6MantleEffortNone(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithRegion("us-east-1"), WithNoAuth())
	require.NoError(t, err)

	_, err = p.NewModel(ModelGPT6Sol, WithReasoningEffort(ReasoningEffortNone))
	require.NoError(t, err)

	_, err = p.NewModel(ModelGPT61Sol, WithReasoningEffort(ReasoningEffortNone))
	require.ErrorContains(t, err, "does not support reasoning effort")
}

// TestNewModel_GPT6MantleRegions checks that a bare GPT-6 ID reaches
// bedrock-mantle in the caller's region whether or not its model card
// publishes that region: MantleRegions never blocks a call.
func TestNewModel_GPT6MantleRegions(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct{ modelID, region string }{
		{ModelGPT6Sol, "us-east-1"},
		{ModelGPT6Sol, "eu-west-1"},
		{ModelGPT61Sol, "us-east-1"},
		{ModelGPT61Sol, "eu-west-1"},
	} {
		t.Run(tt.modelID+"/"+tt.region, func(t *testing.T) {
			t.Parallel()

			capture := &captureRoundTripper{}
			p, err := NewProvider(context.Background(), WithRegion(tt.region), WithNoAuth(), WithHTTPClient(&http.Client{Transport: capture}))
			require.NoError(t, err)

			m, err := p.NewModel(tt.modelID)
			require.NoError(t, err)

			// The canned "{}" body is not a valid response; only the request matters.
			_, _ = m.Generate(context.Background(), &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}})

			require.NotNil(t, capture.req)
			assert.Equal(t, "bedrock-mantle."+tt.region+".api.aws", capture.req.URL.Host)
			assert.Equal(t, "/openai/v1/responses", capture.req.URL.Path)
		})
	}
}

// TestMantleModelNotFoundRegionHint checks that a bedrock-mantle
// model-not-found error names the published regions when the caller's
// region is not among them, on both Generate and GenerateEvents, and is left
// alone from a published region or for a family without MantleRegions.
func TestMantleModelNotFoundRegionHint(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name, modelID, region, hint string
	}{
		{"unlisted region", ModelGPT6Sol, "eu-west-1", "(openai.gpt-6-sol is listed for bedrock-mantle in us-east-1 only; it may not be available in eu-west-1.)"},
		{"listed region", ModelGPT6Sol, "us-east-1", ""},
		{"no MantleRegions", ModelGPT56Sol, "eu-west-1", ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			p, err := NewProvider(context.Background(), WithRegion(tt.region), WithNoAuth(),
				WithHTTPClient(&http.Client{Transport: modelNotFoundRoundTripper{}}))
			require.NoError(t, err)

			m, err := p.NewModel(tt.modelID)
			require.NoError(t, err)

			req := &llm.Request{Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hi"))}}

			_, genErr := m.Generate(context.Background(), req)

			var streamErr error

			for _, err := range m.GenerateEvents(context.Background(), req) {
				if err != nil {
					streamErr = err
					break
				}
			}

			for _, err := range []error{genErr, streamErr} {
				require.Error(t, err)
				require.ErrorIs(t, err, llm.ErrAPICall)

				if tt.hint == "" {
					assert.NotContains(t, err.Error(), "is listed for bedrock-mantle")
				} else {
					assert.True(t, strings.HasSuffix(err.Error(), " "+tt.hint), err.Error())
				}
			}
		})
	}
}

// modelNotFoundRoundTripper answers every request with bedrock-mantle's
// reply for a model it does not serve in the called region, as captured
// from a signed eu-west-1 call on 2026-10-05, naming the requested model.
type modelNotFoundRoundTripper struct{}

func (modelNotFoundRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	var req struct {
		Model string `json:"model"`
	}

	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&req)
	}

	return &http.Response{
		StatusCode: http.StatusNotFound,
		Body: io.NopCloser(strings.NewReader(fmt.Sprintf(
			`{"error":{"code":"not_found_error","message":"The model '%s' does not exist","param":null,"type":"invalid_request_error"}}`,
			req.Model))),
		Header:  http.Header{"Content-Type": []string{"application/json"}},
		Request: r,
	}, nil
}

// TestIsMantleModelNotFound pins the signals the region hint keys on, so the
// code match is exercised on its own and not only through the message
// fallback.
func TestIsMantleModelNotFound(t *testing.T) {
	t.Parallel()

	for _, tt := range []struct {
		name string
		err  error
		want bool
	}{
		{"mantle code without the message", &llm.ProviderError{Base: llm.ErrAPICall, Code: "not_found_error", Message: "Not Found"}, true},
		{"OpenAI code", &llm.ProviderError{Base: llm.ErrAPICall, Code: "model_not_found", Message: "Not Found"}, true},
		{"message without a code", errors.New("The model 'openai.gpt-6-sol' does not exist"), true},
		{"other not-found", &llm.ProviderError{Base: llm.ErrAPICall, Code: "", Message: "route not found"}, false},
		{"rate limit", &llm.ProviderError{Base: llm.ErrRateLimitExceeded, Code: "rate_limit_exceeded", Message: "slow down"}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, isMantleModelNotFound(tt.err))
		})
	}
}

// TestExpandFamiliesRejectsMantleRegionsOutsideMantle checks the authoring
// guard: only a bare-invokable mantle family may declare MantleRegions, not
// (for example) the Converse sibling that shares its bare ID.
func TestExpandFamiliesRejectsMantleRegionsOutsideMantle(t *testing.T) {
	t.Parallel()

	for name, f := range map[string]family{
		"bare Converse": {
			BareID:        "openai.example",
			BareInvokable: true,
			Rates:         pricing.RateCard{Base: pricing.NewRates(1, 2, 0)},
			MantleRegions: []string{"us-east-1"},
		},
		"profile-only Converse": {
			BareID:        "openai.example",
			Profiles:      []string{"us"},
			Rates:         pricing.RateCard{Base: pricing.NewRates(1, 2, 0)},
			MantleRegions: []string{"us-east-1"},
		},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			assert.PanicsWithValue(t, "bedrock: family openai.example sets MantleRegions without Mantle and BareInvokable", func() {
				expandFamilies([]family{f})
			})
		})
	}
}

func TestNewModel_MantleRejectsUnserializedOptions(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithRegion("us-east-1"), WithNoAuth())
	require.NoError(t, err)

	// top_p and stop are NOT serialized by the Responses request mapper, so
	// Gemma does not advertise them — setting either must error rather than
	// silently do nothing.
	_, err = p.NewModel(ModelGemma431B, WithTopP(0.5))
	require.Error(t, err)

	_, err = p.NewModel(ModelGemma431B, WithStop("STOP"))
	require.Error(t, err)
}

func TestNewModel_MantleRejectsThinking(t *testing.T) {
	t.Parallel()

	p, err := NewProvider(context.Background(), WithRegion("us-east-1"), WithNoAuth())
	require.NoError(t, err)

	// WithThinking is Anthropic/Converse-shaped and has no mantle equivalent, so
	// it must error rather than being silently dropped.
	_, err = p.NewModel(ModelGemma431B, WithThinking(1024))
	require.Error(t, err)
}

func TestMantleTransport_EmptyAccessKeyErrors(t *testing.T) {
	t.Parallel()

	capture := &captureRoundTripper{}
	tr := &mantleTransport{
		base: capture,
		credProvider: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{}, nil // resolves, but with an empty access key
		}),
		signer: signerv4.NewSigner(),
		region: "us-east-1",
	}

	resp, err := tr.RoundTrip(newSignedRequest(t))
	require.Error(t, err)

	if resp != nil {
		_ = resp.Body.Close()
	}
	// It fails loudly instead of forwarding an unsigned request downstream.
	assert.Nil(t, capture.req)
}

func TestNewProvider_MantleHonorsAWSConfigHTTPClient(t *testing.T) {
	t.Parallel()

	// A client supplied via WithAWSConfig (not WithHTTPClient) must still be the
	// one the mantle transport wraps, matching the Converse client's behavior.
	custom := &http.Client{}

	p, err := NewProvider(context.Background(), WithAWSConfig(aws.Config{
		Region:      "us-east-1",
		HTTPClient:  custom,
		Credentials: aws.AnonymousCredentials{},
	}))
	require.NoError(t, err)
	assert.Same(t, custom, p.httpClient)
}

// captureRoundTripper records the request it receives and returns a canned 200.
type captureRoundTripper struct {
	req *http.Request
}

func (c *captureRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) {
	c.req = r

	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader("{}")),
		Header:     make(http.Header),
	}, nil
}

func newSignedRequest(t *testing.T) *http.Request {
	t.Helper()

	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost,
		"https://bedrock-mantle.us-east-1.api.aws/openai/v1/responses",
		strings.NewReader(`{"model":"google.gemma-4-31b","input":"hi"}`))
	require.NoError(t, err)
	// The OpenAI SDK sets a bearer token; the transport must strip it.
	req.Header.Set("Authorization", "Bearer bedrock-mantle-sigv4")
	req.Header.Set("Content-Type", "application/json")

	return req
}

func TestMantleTransport_SignsWithMantleService(t *testing.T) {
	t.Parallel()

	capture := &captureRoundTripper{}
	tr := &mantleTransport{
		base: capture,
		credProvider: aws.CredentialsProviderFunc(func(context.Context) (aws.Credentials, error) {
			return aws.Credentials{AccessKeyID: "AKIDEXAMPLE", SecretAccessKey: "secret"}, nil
		}),
		signer: signerv4.NewSigner(),
		region: "us-east-1",
	}

	resp, err := tr.RoundTrip(newSignedRequest(t))
	require.NoError(t, err)

	_ = resp.Body.Close()

	auth := capture.req.Header.Get("Authorization")
	assert.True(t, strings.HasPrefix(auth, "AWS4-HMAC-SHA256"),
		"expected a SigV4 Authorization header, got %q", auth)
	assert.Contains(t, auth, "/us-east-1/bedrock-mantle/aws4_request",
		"SigV4 credential scope should name the bedrock-mantle service")
	assert.NotContains(t, auth, "Bearer", "the placeholder bearer token must be stripped")

	// The body is preserved for the downstream send.
	body, err := io.ReadAll(capture.req.Body)
	require.NoError(t, err)
	assert.JSONEq(t, `{"model":"google.gemma-4-31b","input":"hi"}`, string(body))
}

func TestMantleTransport_AnonymousForwardsUnsigned(t *testing.T) {
	t.Parallel()

	capture := &captureRoundTripper{}
	tr := &mantleTransport{
		base:         capture,
		credProvider: aws.AnonymousCredentials{},
		signer:       signerv4.NewSigner(),
		region:       "us-east-1",
	}

	resp, err := tr.RoundTrip(newSignedRequest(t))
	require.NoError(t, err)

	_ = resp.Body.Close()

	// No signing, and the placeholder bearer is still stripped.
	assert.Empty(t, capture.req.Header.Get("Authorization"))
}

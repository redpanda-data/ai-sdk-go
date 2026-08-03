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
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	signerv4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
)

// mantleService is the AWS SigV4 signing name for the bedrock-mantle endpoint.
// It is deliberately distinct from the standard "bedrock" service that
// bedrock-runtime (Converse / InvokeModel) signs with — a request to
// bedrock-mantle signed as "bedrock" fails with a signature mismatch. See
// https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
const mantleService = "bedrock-mantle"

// mantlePlaceholderAPIKey is a non-empty stand-in required by the OpenAI SDK
// client. Authentication on the mantle endpoint is AWS SigV4, not a bearer
// token, so mantleTransport strips the Authorization header the SDK sets from
// this value before signing.
const mantlePlaceholderAPIKey = "bedrock-mantle-sigv4"

// IsMantleModel reports whether modelID is served on the bedrock-mantle
// endpoint (the OpenAI-compatible Responses / Chat Completions API) rather than
// the standard bedrock-runtime Converse API. It consults the catalog, so it is
// true only for registered mantle models — currently the Google Gemma 4 and
// OpenAI GPT-5.6 families.
//
// The Redpanda AI Gateway's Bedrock reverse proxy reuses this predicate to
// decide, per request, when to sign with the bedrock-mantle signing name and
// route to the mantle host instead of bedrock-runtime.
func IsMantleModel(modelID string) bool {
	def, ok := lookupModel(modelID)
	return ok && def.Mantle
}

// mantleBaseURL returns the OpenAI-compatible base URL for the bedrock-mantle
// endpoint. With no baseEndpoint it targets the real mantle host in the given
// region, e.g. "https://bedrock-mantle.us-east-1.api.aws/openai/v1". When a
// baseEndpoint is set (proxy/gateway mode, via WithAWSConfig) it routes through
// that host instead. Either way the OpenAI SDK appends "/responses" (or
// "/chat/completions"), producing the paths the Gemma 4 model cards document.
func mantleBaseURL(region, baseEndpoint string) string {
	if baseEndpoint != "" {
		return strings.TrimRight(baseEndpoint, "/") + "/openai/v1"
	}

	return fmt.Sprintf("https://bedrock-mantle.%s.api.aws/openai/v1", region)
}

// newMantleModel builds an llm.Model for a mantle-only model (Mantle: true in
// the catalog). It reuses the OpenAI provider's Responses transport, pointed at
// the mantle base URL with a SigV4-signing HTTP client, and wraps the result so
// it reports the Bedrock provider identity.
func newMantleModel(p *Provider, cfg *Config, def ModelDefinition) (llm.Model, error) {
	// WithThinking emits an Anthropic-shaped thinking document (Converse-only)
	// that the mantle Responses API does not accept, and there is no mantle
	// reasoning-budget lever to translate it to. Reject it rather than silently
	// dropping the caller's budget — Gemma's built-in reasoning runs in its
	// default mode when no thinking option is set.
	if cfg.EnableThinking {
		return nil, fmt.Errorf("bedrock-mantle: WithThinking is not supported for mantle model %s", cfg.ModelName)
	}

	baseURL := mantleBaseURL(p.region, p.baseEndpoint)

	transport := &mantleTransport{
		base:         baseTransport(p.httpClient),
		credProvider: p.credentials,
		signer:       p.signer,
		region:       p.region,
	}

	// Preserve the caller's client (Timeout, Jar, CheckRedirect, ...) and only
	// swap in the SigV4-signing transport, so mantle requests behave like the
	// Converse client rather than a bare default.
	httpClient := &http.Client{Transport: transport}
	if p.httpClient != nil {
		clone := *p.httpClient
		clone.Transport = transport
		httpClient = &clone
	}

	oaiProvider, err := openai.NewProvider(
		mantlePlaceholderAPIKey,
		openai.WithBaseURL(baseURL),
		openai.WithHTTPClient(httpClient),
	)
	if err != nil {
		return nil, fmt.Errorf("bedrock-mantle: build OpenAI transport: %w", err)
	}

	oaiDef := openai.ModelDefinition{
		Name:         def.Name,
		Label:        def.Label,
		Capabilities: def.Capabilities,
		Constraints:  def.Constraints,
		Pricing:      def.Pricing,
	}

	// cfg was already validated against the Bedrock constraints in NewModel;
	// forward the concrete parameters as OpenAI options (which re-validate
	// against the same constraints). WithThinking was rejected above.
	inner, err := oaiProvider.NewCompatModel(cfg.APIModelID, oaiDef, translateMantleOptions(cfg)...)
	if err != nil {
		return nil, fmt.Errorf("bedrock-mantle: build model %s: %w", cfg.ModelName, err)
	}

	return &mantleModel{Model: inner, name: cfg.ModelName, def: def}, nil
}

// translateMantleOptions maps the concrete parameters resolved on a Bedrock
// Config into the equivalent OpenAI options for the mantle transport. Only
// temperature and max_tokens are forwarded — those are the sampling controls
// the Responses request mapper serializes and the only ones the mantle model
// constraints advertise.
func translateMantleOptions(cfg *Config) []openai.Option {
	var opts []openai.Option

	if cfg.Temperature != nil {
		opts = append(opts, openai.WithTemperature(*cfg.Temperature))
	}

	if cfg.MaxTokens != nil {
		opts = append(opts, openai.WithMaxTokens(int(*cfg.MaxTokens)))
	}

	return opts
}

// mantleModel wraps the OpenAI-transport model so it reports the Bedrock
// provider identity ("aws.bedrock") and the user-facing model name, while
// delegating Generate/GenerateEvents to the embedded model.
type mantleModel struct {
	llm.Model // embedded OpenAI Responses model (Generate/GenerateEvents)

	name string
	def  ModelDefinition
}

func (m *mantleModel) Name() string                        { return m.name }
func (*mantleModel) Provider() string                      { return "aws.bedrock" }
func (m *mantleModel) Capabilities() llm.ModelCapabilities { return m.def.Capabilities }
func (m *mantleModel) Constraints() llm.ModelConstraints   { return m.def.Constraints }

// mantleTransport is an http.RoundTripper that SigV4-signs OpenAI-shaped
// requests for the bedrock-mantle endpoint. The OpenAI SDK has no notion of AWS
// signing, so this transport removes the placeholder bearer token, computes the
// payload hash, and signs with the bedrock-mantle service name before
// forwarding.
type mantleTransport struct {
	base         http.RoundTripper
	credProvider aws.CredentialsProvider
	signer       *signerv4.Signer
	region       string
}

func (t *mantleTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// SigV4 owns the Authorization header on the mantle endpoint; strip the
	// bearer placeholder the OpenAI SDK sets before signing (or before
	// forwarding unsigned in proxy mode).
	req.Header.Del("Authorization")

	// No credential provider or anonymous credentials (WithNoAuth / proxy
	// mode): forward unsigned and let the downstream proxy authenticate.
	// AnonymousCredentials.Retrieve errors rather than returning empty creds,
	// so this must be checked before Retrieve; IsCredentialsProvider also
	// unwraps the CredentialsCache the SDK wraps providers in.
	if t.credProvider == nil || aws.IsCredentialsProvider(t.credProvider, aws.AnonymousCredentials{}) {
		return t.base.RoundTrip(req)
	}

	creds, err := t.credProvider.Retrieve(req.Context())
	if err != nil {
		return nil, fmt.Errorf("bedrock-mantle: retrieve AWS credentials: %w", err)
	}

	// Anonymous is already handled above, so an empty access key here is a
	// misconfigured credential provider. Fail loudly rather than sending an
	// unsigned request the mantle endpoint rejects with an opaque 403.
	if creds.AccessKeyID == "" {
		return nil, errors.New("bedrock-mantle: resolved AWS credentials have an empty access key ID")
	}

	// Buffer the body to compute the exact SHA-256 SigV4 requires and to keep
	// it readable for the actual send.
	var body []byte
	if req.Body != nil {
		body, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, fmt.Errorf("bedrock-mantle: read request body: %w", err)
		}

		_ = req.Body.Close()
		req.Body = io.NopCloser(bytes.NewReader(body))
		req.ContentLength = int64(len(body))
	}

	sum := sha256.Sum256(body)
	payloadHash := hex.EncodeToString(sum[:])

	if err := t.signer.SignHTTP(req.Context(), creds, req, payloadHash, mantleService, t.region, time.Now()); err != nil {
		return nil, fmt.Errorf("bedrock-mantle: sign request: %w", err)
	}

	return t.base.RoundTrip(req)
}

// baseTransport returns the RoundTripper to forward signed requests with,
// preferring a caller-supplied client's Transport and falling back to the
// default.
func baseTransport(c *http.Client) http.RoundTripper {
	if c != nil && c.Transport != nil {
		return c.Transport
	}

	return http.DefaultTransport
}

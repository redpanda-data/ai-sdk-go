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

package vertex

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	anthropicsdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"google.golang.org/genai"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
)

// NewModel returns an llm.Model for a catalogued Vertex model at the
// provider's project and location. Gemini models reuse providers/google
// over genai's Vertex backend, and Claude models reuse providers/anthropic;
// both report gcp.vertex and this package's offering.
//
// It needs WithProject, WithLocation and WithHTTPClient. The HTTP client
// must authenticate requests itself: genai detects Application Default
// Credentials when no HTTP client is set, and in a multi-tenant host those
// bill one tenant's traffic to the host's own project.
func (p *Provider) NewModel(modelName string, opts ...Option) (llm.Model, error) {
	offering, ok := Catalog().Resolve(modelName)
	if !ok {
		return nil, fmt.Errorf("unsupported Vertex model: %s", modelName)
	}

	cfg := &modelConfig{}
	for _, opt := range opts {
		err := opt(cfg)
		if err != nil {
			return nil, fmt.Errorf("invalid option for %s: %w", modelName, err)
		}
	}

	err := p.checkModelConfig()
	if err != nil {
		return nil, err
	}

	switch publisher := offering.Facts().Publisher; publisher {
	case catalog.PublisherGoogle:
		return p.newGeminiModel(modelName, offering, cfg)
	case catalog.PublisherAnthropic:
		return p.newClaudeModel(modelName, offering, cfg)
	case catalog.PublisherAmazon, catalog.PublisherMeta, catalog.PublisherMistral, catalog.PublisherOpenAI:
		return nil, fmt.Errorf("vertex: model %s is published by %q, which this package cannot call", modelName, publisher)
	default:
		return nil, fmt.Errorf("vertex: model %s is published by %q, which this package cannot call", modelName, publisher)
	}
}

func (p *Provider) checkModelConfig() error {
	switch {
	case p.project == "":
		return errors.New("vertex: a model needs a project; use WithProject")
	case p.location == "":
		return errors.New("vertex: a model needs a location; use WithLocation")
	case p.httpClient == nil:
		return errors.New("vertex: a model needs an authenticating HTTP client; use WithHTTPClient")
	}

	return nil
}

// genaiAPIVersion pins the Vertex API version. genai defaults the Vertex
// backend to v1beta1; the AI Gateway's model-call path takes either, and
// v1 matches the Claude path below.
const genaiAPIVersion = "v1"

func (p *Provider) newGeminiModel(modelName string, offering catalog.Offering, cfg *modelConfig) (llm.Model, error) {
	ctx := context.Background()

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    p.project,
		Location:   p.location,
		HTTPClient: p.httpClient,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    p.endpoint(),
			APIVersion: genaiAPIVersion,
		},
	})
	if err != nil {
		return nil, fmt.Errorf("vertex: build genai client: %w", err)
	}

	provider, err := google.NewProviderWithClient(ctx, client)
	if err != nil {
		return nil, fmt.Errorf("vertex: %w", err)
	}

	// The request names the catalog ID, as the Claude path does, so a model
	// reached by alias or snapshot reaches the AI Gateway under one name.
	opts := []google.Option{google.WithCustomModelName(offering.ID)}
	if cfg.reasoningEffort != nil {
		opts = append(opts, google.WithReasoningEffort(*cfg.reasoningEffort))
	}

	return provider.NewModelFromOffering(modelName, offering, opts...)
}

// anthropicVersion is the anthropic_version Vertex requires in a Claude
// request body. It is not Bedrock's value; a body sent with that one is
// refused with "Invalid API version".
const anthropicVersion = "vertex-2023-10-16"

// anthropicMessagesPath is the path suffix the Anthropic SDK builds for a
// Messages call. The middleware matches it as a suffix, because a gateway
// base URL carries a path of its own, and count_tokens does not end in it.
const anthropicMessagesPath = "/v1/messages"

func (p *Provider) newClaudeModel(modelName string, offering catalog.Offering, cfg *modelConfig) (llm.Model, error) {
	client := anthropicsdk.NewClient(
		option.WithBaseURL(p.endpoint()+"/"),
		option.WithHTTPClient(p.httpClient),
		option.WithMiddleware(p.rawPredictMiddleware(offering.ID)),
	)

	provider, err := anthropic.NewProviderWithClient(&client)
	if err != nil {
		return nil, fmt.Errorf("vertex: %w", err)
	}

	var opts []anthropic.Option
	if cfg.reasoningEffort != nil {
		opts = append(opts, anthropic.WithReasoningEffort(*cfg.reasoningEffort))
	}

	return provider.NewModelFromOffering(modelName, offering, opts...)
}

// rawPredictMiddleware turns an Anthropic Messages request into a Vertex
// publisher request. It makes these path and body edits:
//
//   - the path becomes the publisher path, ending :rawPredict, or
//     :streamRawPredict when the body asks to stream, with no query;
//   - the model field is deleted, because Vertex takes the model from the
//     path and refuses it in the body;
//   - anthropic_version is set when absent.
//
// It also drops any credential header the Anthropic SDK took from the
// environment (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN). The provider's
// HTTP client sets the only credential, after this middleware runs.
//
// Any other request passes through, including /v1/messages/count_tokens,
// which has no Vertex equivalent for Claude.
func (p *Provider) rawPredictMiddleware(model string) option.Middleware {
	return func(r *http.Request, next option.MiddlewareNext) (*http.Response, error) {
		r.Header.Del("X-Api-Key")
		r.Header.Del("Authorization")

		prefix, matched := strings.CutSuffix(r.URL.Path, anthropicMessagesPath)
		if r.Body == nil || r.Method != http.MethodPost || !matched {
			return next(r)
		}

		body, err := io.ReadAll(r.Body)
		_ = r.Body.Close()

		if err != nil {
			return nil, fmt.Errorf("vertex: read request body: %w", err)
		}

		if !gjson.GetBytes(body, "anthropic_version").Exists() {
			body, err = sjson.SetBytes(body, "anthropic_version", anthropicVersion)
			if err != nil {
				return nil, fmt.Errorf("vertex: set anthropic_version: %w", err)
			}
		}

		verb := "rawPredict"
		if gjson.GetBytes(body, "stream").Bool() {
			verb = "streamRawPredict"
		}

		body, err = sjson.DeleteBytes(body, "model")
		if err != nil {
			return nil, fmt.Errorf("vertex: delete model from body: %w", err)
		}

		// The prefix is kept, so a gateway request stays under its
		// provider path.
		r.URL.Path = prefix + "/v1/projects/" + p.project + "/locations/" + p.location +
			"/publishers/anthropic/models/" + model + ":" + verb
		// The SDK's beta Messages path adds ?beta=true, which is not a Vertex
		// parameter; Vertex takes beta opt-ins in the anthropic-beta header.
		r.URL.RawQuery = ""

		// GetBody too, so a rewind by net/http, on a redirect or an HTTP/2
		// retry, resends the rewritten body and not the SDK's original one.
		r.Body = io.NopCloser(bytes.NewReader(body))
		r.GetBody = func() (io.ReadCloser, error) {
			return io.NopCloser(bytes.NewReader(body)), nil
		}
		r.ContentLength = int64(len(body))

		return next(r)
	}
}

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
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Provider is Google's Gemini Enterprise Agent Platform (formerly Vertex
// AI): its name, its validated model catalog, and models that build Vertex
// requests for one project and location. It implements catalog.Provider for
// the spending pipeline and the catalog snapshot.
type Provider struct {
	project    string
	location   string
	baseURL    string
	httpClient *http.Client
}

var _ catalog.Provider = (*Provider)(nil)

// ProviderOption configures a Provider.
type ProviderOption func(*Provider) error

// NewProvider returns a Provider. The options are needed only to build
// models; the catalog surface takes none.
func NewProvider(_ context.Context, opts ...ProviderOption) (*Provider, error) {
	p := &Provider{}

	for _, opt := range opts {
		err := opt(p)
		if err != nil {
			return nil, fmt.Errorf("provider configuration error: %w", err)
		}
	}

	return p, nil
}

// WithProject sets the Google Cloud project that serves and is billed for
// the models.
func WithProject(project string) ProviderOption {
	return func(p *Provider) error {
		if project == "" {
			return errors.New("project cannot be empty")
		}

		p.project = project

		return nil
	}
}

// WithLocation sets the Vertex location models are called at: global, a
// multi-region (us, eu), or a region such as us-east5.
func WithLocation(location string) ProviderOption {
	return func(p *Provider) error {
		location = normalizeLocation(location)
		if location == "" {
			return errors.New("location cannot be empty")
		}

		p.location = location

		return nil
	}
}

// WithBaseURL replaces the location's Vertex endpoint, for example with an
// AI Gateway provider URL. The Vertex path is appended to it.
func WithBaseURL(baseURL string) ProviderOption {
	return func(p *Provider) error {
		if baseURL == "" {
			return errors.New("base URL cannot be empty")
		}

		p.baseURL = strings.TrimSuffix(baseURL, "/")

		return nil
	}
}

// WithHTTPClient sets the HTTP client models send requests with. It must
// authenticate the requests itself: this package adds no credential, and
// never falls back to Application Default Credentials.
func WithHTTPClient(client *http.Client) ProviderOption {
	return func(p *Provider) error {
		if client == nil {
			return errors.New("HTTP client cannot be nil")
		}

		p.httpClient = client

		return nil
	}
}

// Name returns the provider identifier used in offerings and telemetry.
func (*Provider) Name() llm.ProviderID {
	return ProviderName
}

// Catalog implements catalog.Provider. See the package-level [Catalog].
func (*Provider) Catalog() *catalog.Catalog {
	return Catalog()
}

// endpoint is the base URL models send to: the configured one, or the
// location's Vertex host. The multi-regions us and eu are served at
// aiplatform.<location>.rep.googleapis.com, not the
// <location>-aiplatform.googleapis.com host genai derives.
func (p *Provider) endpoint() string {
	if p.baseURL != "" {
		return p.baseURL
	}

	switch p.location {
	case LocationGlobal:
		return "https://aiplatform.googleapis.com"
	case "us", "eu":
		return "https://aiplatform." + p.location + ".rep.googleapis.com"
	default:
		return "https://" + p.location + "-aiplatform.googleapis.com"
	}
}

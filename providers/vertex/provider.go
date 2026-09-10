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

	"github.com/redpanda-data/ai-sdk-go/catalog"
)

// TODO(maciej): add request transport (RFC-0014 M8).

// Provider is the catalog surface for Google's Gemini Enterprise Agent
// Platform (formerly Vertex AI): its name and validated model catalog. It
// implements catalog.Provider, consumed by the spending pipeline and the
// catalog snapshot.
//
// No request transport yet (RFC-0014 M8): the provider contributes its
// catalog and rates, and the AI Gateway forwards the customer's own native
// Vertex requests.
type Provider struct{}

var _ catalog.Provider = (*Provider)(nil)

// NewProvider returns a provider for Google's Gemini Enterprise Agent
// Platform (formerly Vertex AI).
func NewProvider(_ context.Context) (*Provider, error) {
	return &Provider{}, nil
}

// Name returns the provider identifier used in offerings and telemetry.
func (*Provider) Name() string {
	return providerName
}

// Catalog implements catalog.Provider: the validated Vertex model
// catalog, including pricing and lifecycle metadata.
func (*Provider) Catalog() *catalog.Catalog {
	return Catalog()
}

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

// Provider is the Google Vertex AI provider's catalog surface: its name
// and its validated model catalog. It satisfies catalog.Provider, which
// is what the spending pipeline and the catalog snapshot consume.
//
// It carries no request transport yet. Constructing Vertex requests is a
// later milestone (RFC-0014 M8); until then the provider contributes its
// catalog and rates, and the AI Gateway forwards the customer's own
// native Vertex requests.
type Provider struct{}

var _ catalog.Provider = (*Provider)(nil)

// NewProvider returns the Vertex provider. ctx is unused today and the
// returned error is always nil; the (context.Context) (*Provider, error)
// shape matches the sibling providers and reserves room for the transport
// milestone (RFC-0014 M8), which will load credentials at construction
// under the caller's context.
func NewProvider(ctx context.Context) (*Provider, error) {
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

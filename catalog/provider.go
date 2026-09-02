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

package catalog

// Provider is the discovery-and-pricing surface a model provider
// exposes. It lives in this package rather than llm because llm cannot
// import pricing (the dependency runs the other way), and a catalog
// without pricing would repeat the split this package exists to remove.
//
// Catalog may return nil for providers whose model space is inherently
// dynamic (openaicompat-style endpoints, where model names are
// caller-defined); consumers must treat a nil catalog as "no model
// metadata available", not as an error.
type Provider interface {
	// Name returns the provider identifier used in offerings and
	// telemetry (e.g. "openai", "anthropic", "aws.bedrock").
	Name() string

	// Catalog returns the provider's validated model catalog, or nil
	// when the provider has no static catalog.
	Catalog() *Catalog
}

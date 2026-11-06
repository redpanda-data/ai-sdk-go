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

package fireworks

import (
	"context"
	"fmt"
	"iter"
	"slices"

	"github.com/redpanda-data/ai-sdk-go/catalog"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

type model struct {
	llm.Model

	name     string
	offering catalog.Offering
}

var (
	_ llm.Model                 = (*model)(nil)
	_ llm.ReasoningEffortLister = (*model)(nil)
)

func (m *model) Name() string { return m.name }

func (*model) Provider() llm.ProviderID { return ProviderName }

func (m *model) Capabilities() llm.ModelCapabilities { return m.offering.Capabilities }

func (m *model) Constraints() llm.ModelConstraints {
	constraints := m.offering.Constraints
	constraints.SupportedParams = slices.Clone(constraints.SupportedParams)

	return constraints
}

func (m *model) SupportedReasoningEfforts() []llm.ReasoningEffort {
	return slices.Clone(m.offering.Reasoning.Efforts)
}

func (m *model) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	if err := m.validateRequest(req); err != nil {
		return nil, err
	}

	response, err := m.Model.Generate(ctx, req)
	if response != nil {
		response.InvokedModelID = m.offering.ID
	}

	return response, err
}

func (m *model) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		if err := m.validateRequest(req); err != nil {
			yield(nil, err)
			return
		}

		for event, err := range m.Model.GenerateEvents(ctx, req) {
			if end, ok := event.(llm.StreamEndEvent); ok && end.Response != nil {
				end.Response.InvokedModelID = m.offering.ID
				event = end
			}

			if !yield(event, err) {
				return
			}
		}
	}
}

func (m *model) validateRequest(req *llm.Request) error {
	if req == nil {
		return fmt.Errorf("%w: Fireworks request cannot be nil", llm.ErrInvalidInput)
	}

	caps := m.offering.Capabilities

	if req.Options != nil {
		return fmt.Errorf("%w: use Fireworks model options, not foreign request options", llm.ErrUnsupportedFeature)
	}

	if req.ToolSearch || (len(req.Tools) > 0 && !caps.Tools) {
		return fmt.Errorf("%w: Fireworks binding does not support the requested tools", llm.ErrUnsupportedFeature)
	}

	if req.ResponseFormat != nil {
		if req.ResponseFormat.Type == llm.ResponseFormatJSONObject && !caps.JSONMode {
			return fmt.Errorf("%w: selected Fireworks API requires a JSON schema", llm.ErrUnsupportedFeature)
		}

		if req.ResponseFormat.Type == llm.ResponseFormatJSONSchema && !caps.StructuredOutput {
			return fmt.Errorf("%w: selected Fireworks model configuration does not support structured output", llm.ErrUnsupportedFeature)
		}
	}

	return nil
}

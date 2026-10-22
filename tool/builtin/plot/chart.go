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

package plot

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
	"sync"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// ChartMediaType identifies validated, declarative charts for client-owned rendering.
const ChartMediaType = "application/vnd.redpanda.chart+json"

// MaxChartInputBytes bounds input before JSON decoding.
const MaxChartInputBytes = 100_000

// MaxChartOutputBytes keeps encoded artifacts within the registry's default
// 25,000 estimated-token budget. Registrations with a smaller budget must opt in
// to at least this many bytes / 4 tokens or handle their own stricter limit.
const MaxChartOutputBytes = 100_000

// ChartInput is the input contract of NewChart. It is separate from the legacy PNG Input.
type ChartInput struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Chart       json.RawMessage `json:"chart"`
}

// ChartOutput contains chart data, not generated HTML, JavaScript, or PNG bytes.
// Chart preserves supplied values; null optional object members are omitted.
// Consumers render Chart directly; they must not ask the model to rewrite it.
type ChartOutput struct {
	ArtifactID  string          `json:"artifact_id"`
	MimeType    string          `json:"mime_type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Chart       json.RawMessage `json:"chart"`
}

//go:embed chart.schema.json
var chartInputSchema json.RawMessage

var resolvedChartSchemas = sync.OnceValues(func() (map[string]*jsonschema.Resolved, error) {
	var schema jsonschema.Schema
	if err := json.Unmarshal(chartInputSchema, &schema); err != nil {
		return nil, err
	}

	root, err := schema.Resolve(nil)
	if err != nil {
		return nil, err
	}

	schemas := map[string]*jsonschema.Resolved{"": root}

	for _, branch := range schema.Properties["chart"].AnyOf {
		// Select the declared chart branch without weakening envelope validation.
		// This gives the caller the actual missing field, not an opaque anyOf error.
		specific := schema
		specific.Properties = maps.Clone(schema.Properties)
		specific.Properties["chart"] = branch

		resolved, err := specific.Resolve(nil)
		if err != nil {
			return nil, err
		}

		for _, rawKind := range branch.Properties["type"].Enum {
			kind, ok := rawKind.(string)
			if !ok {
				return nil, errors.New("chart schema type enum must contain strings")
			}

			schemas[kind] = resolved
		}
	}

	return schemas, nil
})

type chartTool struct{}

// NewChart opts the plot tool into validated chart-data output for interactive clients.
// Register it instead of New, not alongside it: both tools are named plot.
// New, Tool, Input and Output retain their existing PNG behavior unchanged.
// Validation checks structure and chart constraints, not the provenance or truth of data.
func NewChart() tool.Tool { return &chartTool{} }

func (*chartTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name: "plot",
		Description: `Create a chart from user-supplied or tool-verified data. The client renders the returned artifact directly with Chart, Data and Code views. Use the chart schema; all chart-specific fields belong inside chart.data. Do not emit chart code fences or repeat the artifact's JSON in your answer. Explain the finding in the user's requested response format.
Use only known values and relationships. Ask for missing information instead of inventing data, gauge bounds, quantiles, totals or flows. Synthetic data must be explicitly identified. Do not derive distributions from averages alone. Follow configured roles, output requirements, access and privacy limits. This tool validates chart structure, not data provenance.`,
		Parameters: append(json.RawMessage(nil), chartInputSchema...),
		Type:       llm.ToolTypeFunction,
		Metadata:   map[string]any{"category": "visualization"},
	}
}

func (*chartTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if len(args) > MaxChartInputBytes {
		return nil, fmt.Errorf("chart input exceeds %d bytes", MaxChartInputBytes)
	}

	if err := validateChartJSON(args); err != nil {
		return nil, fmt.Errorf("invalid chart JSON: %w", err)
	}

	var value any
	if err := json.Unmarshal(args, &value); err != nil {
		return nil, fmt.Errorf("invalid chart JSON: %w", err)
	}

	schemas, err := resolvedChartSchemas()
	if err != nil {
		return nil, fmt.Errorf("resolve chart schema: %w", err)
	}

	object, _ := value.(map[string]any)
	chart, _ := object["chart"].(map[string]any)
	kind, _ := chart["type"].(string)

	schema := schemas[kind]
	if schema == nil {
		schema = schemas[""]
	}

	if err := schema.Validate(value); err != nil {
		return nil, fmt.Errorf("invalid chart input: %w", err)
	}

	var input ChartInput
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("decode chart input: %w", err)
	}

	if strings.TrimSpace(input.Name) == "" || strings.TrimSpace(input.Description) == "" {
		return nil, errors.New("chart name and description must not be blank")
	}

	input.Chart, err = normalizeChart(input.Chart)
	if err != nil {
		return nil, fmt.Errorf("normalize chart: %w", err)
	}

	if err := validateChartSemantics(input.Chart); err != nil {
		return nil, fmt.Errorf("invalid chart: %w", err)
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	output, err := json.Marshal(ChartOutput{ArtifactID: "plot-" + xid.New().String(), MimeType: ChartMediaType, Name: input.Name, Description: input.Description, Chart: input.Chart})
	if err != nil {
		return nil, err
	}

	if len(output) > MaxChartOutputBytes {
		return nil, fmt.Errorf("encoded chart output exceeds %d bytes; reduce samples or label lengths", MaxChartOutputBytes)
	}

	return output, nil
}

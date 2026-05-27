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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/rs/xid"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/builtin"
)

// Tool implements the plot tool for generating charts.
type Tool struct{}

var _ tool.Tool = (*Tool)(nil)

// New creates a new plot tool instance.
func New() tool.Tool {
	return &Tool{}
}

// Definition returns the tool definition for LLM consumption.
func (*Tool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name: "plot",
		Description: `Generate data visualization charts as PNG images, returned as artifacts.

WHEN TO USE:
- Visualizing SQL query results (time series, distributions, comparisons)
- Showing trends and patterns in data
- Creating charts for data analysis and reporting
- Visualizing correlations and relationships

CHART TYPES:
- line: Time series, trends, continuous data with multiple series
- bar: Category comparisons, grouped data, multiple series side-by-side
- scatter: Point clouds, correlations, relationships between variables
- histogram: Frequency distributions, data binning

OUTPUT:
Returns an artifact ID reference. The chart PNG is stored as an artifact and not in message history.
You can reference the artifact by its ID in follow-up messages.

IMPORTANT:
- Must provide 'name' and 'description' for the artifact
- Each chart type requires specific data structure (line_data, bar_data, scatter_data, histogram_data)
- X and Y arrays must have matching lengths for line/scatter charts
- Bar chart values must match categories length

EXAMPLES:
Line: {"name": "User Growth", "description": "Daily active users over last 30 days", "chart_type": "line", "title": "Daily Users", "x_label": "Day", "y_label": "Count", "line_data": {"series": [{"name": "Users", "x": [1,2,3], "y": [100,150,120]}]}}
Bar: {"name": "Regional Sales", "description": "Q1 sales by region", "chart_type": "bar", "title": "Sales by Region", "bar_data": {"categories": ["North","South"], "series": [{"name": "Q1", "values": [100,150]}]}}
Scatter: {"name": "Transaction Analysis", "description": "Amount vs fraud score correlation", "chart_type": "scatter", "scatter_data": {"series": [{"name": "Transactions", "x": [10,20,30], "y": [0.1,0.5,0.9]}]}}
Histogram: {"name": "Response Time Distribution", "description": "API response time frequency", "chart_type": "histogram", "histogram_data": {"values": [12.3,45.2,23.1], "bins": 10}}`,
		Parameters: plotInputSchema,
		Type:       llm.ToolTypeFunction,
		Metadata: map[string]any{
			"category": "visualization",
		},
	}
}

// Execute performs the plot generation.
func (*Tool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var input Input
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("invalid plot input: %w", err)
	}

	// Validate required artifact metadata
	if input.Name == "" {
		return nil, errors.New("plot must have non-empty name")
	}

	if input.Description == "" {
		return nil, errors.New("plot must have non-empty description")
	}

	// Build the chart
	p, width, height, err := buildChart(input)
	if err != nil {
		return nil, fmt.Errorf("failed to build chart: %w", err)
	}

	// Render to PNG
	pngBytes, err := renderToPNG(p, width, height)
	if err != nil {
		return nil, fmt.Errorf("failed to render chart: %w", err)
	}

	// Encode to base64
	base64Data := base64.StdEncoding.EncodeToString(pngBytes)

	// Create output with artifact ID
	// The reconciler will extract PNGData and create an artifact,
	// then replace this response with just {artifactId} for the LLM
	output := Output{
		ArtifactID: "plot-" + xid.New().String(),
		PNGData:    base64Data,
		MimeType:   "image/png",
		Filename:   "chart.png",
		Width:      width,
		Height:     height,
	}

	return json.Marshal(output)
}

// Manual JSON schema for plot Input type. Parsed once at init time from the
// JSON literal below to keep the source readable; MustParseSchema panics on
// invalid schema, which is desirable for this compile-time constant.
var plotInputSchema = builtin.MustParseSchema(`{
	"type": "object",
	"properties": {
		"name": {"type": "string", "description": "Name for the plot artifact"},
		"description": {"type": "string", "description": "Description of what the plot shows"},
		"chart_type": {
			"type": "string",
			"enum": ["line", "bar", "scatter", "histogram"],
			"description": "Type of chart to generate"
		},
		"title": {"type": "string", "description": "Chart title"},
		"x_label": {"type": "string", "description": "X-axis label"},
		"y_label": {"type": "string", "description": "Y-axis label"},
		"options": {
			"type": "object",
			"description": "Chart rendering options",
			"properties": {
				"width": {"type": "integer", "description": "Chart width in pixels (default 800)"},
				"height": {"type": "integer", "description": "Chart height in pixels (default 600)"},
				"legend": {"type": "boolean", "description": "Show legend (default true)"},
				"grid": {"type": "boolean", "description": "Show grid lines (default true)"}
			},
			"additionalProperties": false
		},
		"line_data": {
			"type": "object",
			"description": "Data for line charts",
			"properties": {
				"series": {
					"type": "array",
					"description": "Data series for line chart",
					"minItems": 1,
					"items": {
						"type": "object",
						"properties": {
							"name": {"type": "string", "description": "Series name for legend"},
							"x": {"type": "array", "description": "X-axis values", "minItems": 1, "items": {"type": "number"}},
							"y": {"type": "array", "description": "Y-axis values", "minItems": 1, "items": {"type": "number"}}
						},
						"required": ["name", "x", "y"],
						"additionalProperties": false
					}
				}
			},
			"required": ["series"],
			"additionalProperties": false
		},
		"bar_data": {
			"type": "object",
			"description": "Data for bar charts",
			"properties": {
				"categories": {"type": "array", "description": "Category labels for X-axis", "minItems": 1, "items": {"type": "string"}},
				"series": {
					"type": "array",
					"description": "Data series for bar chart",
					"minItems": 1,
					"items": {
						"type": "object",
						"properties": {
							"name": {"type": "string", "description": "Series name for legend"},
							"values": {"type": "array", "description": "Data values", "minItems": 1, "items": {"type": "number"}}
						},
						"required": ["name", "values"],
						"additionalProperties": false
					}
				}
			},
			"required": ["categories", "series"],
			"additionalProperties": false
		},
		"scatter_data": {
			"type": "object",
			"description": "Data for scatter plots",
			"properties": {
				"series": {
					"type": "array",
					"description": "Data series for scatter plot",
					"minItems": 1,
					"items": {
						"type": "object",
						"properties": {
							"name": {"type": "string", "description": "Series name for legend"},
							"x": {"type": "array", "description": "X-axis values", "minItems": 1, "items": {"type": "number"}},
							"y": {"type": "array", "description": "Y-axis values", "minItems": 1, "items": {"type": "number"}}
						},
						"required": ["name", "x", "y"],
						"additionalProperties": false
					}
				}
			},
			"required": ["series"],
			"additionalProperties": false
		},
		"histogram_data": {
			"type": "object",
			"description": "Data for histograms",
			"properties": {
				"values": {"type": "array", "description": "Raw values to bin into histogram", "minItems": 1, "items": {"type": "number"}},
				"bins": {"type": "integer", "description": "Number of histogram bins (default 10)"}
			},
			"required": ["values"],
			"additionalProperties": false
		}
	},
	"required": ["name", "description", "chart_type"],
	"additionalProperties": false
}`)

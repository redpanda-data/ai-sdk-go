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

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Tool implements the plot tool for generating charts.
type Tool struct{}

var _ tool.Tool = (*Tool)(nil)

const plotDescription = `Generate data visualization charts as PNG images, returned as artifacts.

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
- Bar chart values must match categories length`

// New creates a new plot tool instance.
func New() tool.Tool {
	return &Tool{}
}

// Name implements tool.Tool.
func (*Tool) Name() string { return "plot" }

// Description implements tool.Tool.
func (*Tool) Description() string { return plotDescription }

// InputSchema implements tool.Tool.
func (*Tool) InputSchema() json.RawMessage {
	schemaBytes, err := json.Marshal(plotInputSchema)
	if err != nil {
		return json.RawMessage(`{"type":"object"}`)
	}

	return schemaBytes
}

// Execute performs the plot generation.
func (*Tool) Execute(_ context.Context, call tool.Call) (tool.Execution, error) {
	var input Input
	if err := json.Unmarshal(call.Request.Arguments, &input); err != nil {
		return tool.Execution{}, fmt.Errorf("invalid plot input: %w", err)
	}

	if input.Name == "" {
		return tool.Execution{}, errors.New("plot must have non-empty name")
	}

	if input.Description == "" {
		return tool.Execution{}, errors.New("plot must have non-empty description")
	}

	p, width, height, err := buildChart(input)
	if err != nil {
		return tool.Execution{}, fmt.Errorf("failed to build chart: %w", err)
	}

	pngBytes, err := renderToPNG(p, width, height)
	if err != nil {
		return tool.Execution{}, fmt.Errorf("failed to render chart: %w", err)
	}

	base64Data := base64.StdEncoding.EncodeToString(pngBytes)

	output := Output{
		ArtifactID: "plot-" + xid.New().String(),
		PNGData:    base64Data,
		MimeType:   "image/png",
		Filename:   "chart.png",
		Width:      width,
		Height:     height,
	}

	outputBytes, err := json.Marshal(output)
	if err != nil {
		return tool.Execution{}, fmt.Errorf("marshal plot output: %w", err)
	}

	return tool.Execution{
		Output: outputBytes,
		Actions: []tool.Action{{
			Kind: tool.ActionArtifact,
			Artifact: &tool.ArtifactAction{
				ID:          output.ArtifactID,
				Name:        input.Name,
				Description: input.Description,
				MediaType:   output.MimeType,
				Data:        pngBytes,
			},
		}},
	}, nil
}

// Manual JSON schema for plot Input type.
var plotInputSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"name": map[string]any{
			"type":        "string",
			"description": "Name for the plot artifact",
		},
		"description": map[string]any{
			"type":        "string",
			"description": "Description of what the plot shows",
		},
		"chart_type": map[string]any{
			"type":        "string",
			"enum":        []string{"line", "bar", "scatter", "histogram"},
			"description": "Type of chart to generate",
		},
		"title": map[string]any{
			"type":        "string",
			"description": "Chart title",
		},
		"x_label": map[string]any{
			"type":        "string",
			"description": "X-axis label",
		},
		"y_label": map[string]any{
			"type":        "string",
			"description": "Y-axis label",
		},
		"options": map[string]any{
			"type":        "object",
			"description": "Chart rendering options",
			"properties": map[string]any{
				"width": map[string]any{
					"type":        "integer",
					"description": "Chart width in pixels (default 800)",
				},
				"height": map[string]any{
					"type":        "integer",
					"description": "Chart height in pixels (default 600)",
				},
				"legend": map[string]any{
					"type":        "boolean",
					"description": "Show legend (default true)",
				},
				"grid": map[string]any{
					"type":        "boolean",
					"description": "Show grid lines (default true)",
				},
			},
			"additionalProperties": false,
		},
		"line_data": map[string]any{
			"type":        "object",
			"description": "Data for line charts",
			"properties": map[string]any{
				"series": map[string]any{
					"type":        "array",
					"description": "Data series for line chart",
					"minItems":    1,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Series name for legend",
							},
							"x": map[string]any{
								"type":        "array",
								"description": "X-axis values",
								"minItems":    1,
								"items": map[string]any{
									"type": "number",
								},
							},
							"y": map[string]any{
								"type":        "array",
								"description": "Y-axis values",
								"minItems":    1,
								"items": map[string]any{
									"type": "number",
								},
							},
						},
						"required":             []string{"name", "x", "y"},
						"additionalProperties": false,
					},
				},
			},
			"required":             []string{"series"},
			"additionalProperties": false,
		},
		"bar_data": map[string]any{
			"type":        "object",
			"description": "Data for bar charts",
			"properties": map[string]any{
				"categories": map[string]any{
					"type":        "array",
					"description": "Category labels for X-axis",
					"minItems":    1,
					"items": map[string]any{
						"type": "string",
					},
				},
				"series": map[string]any{
					"type":        "array",
					"description": "Data series for bar chart",
					"minItems":    1,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Series name for legend",
							},
							"values": map[string]any{
								"type":        "array",
								"description": "Data values",
								"minItems":    1,
								"items": map[string]any{
									"type": "number",
								},
							},
						},
						"required":             []string{"name", "values"},
						"additionalProperties": false,
					},
				},
			},
			"required":             []string{"categories", "series"},
			"additionalProperties": false,
		},
		"scatter_data": map[string]any{
			"type":        "object",
			"description": "Data for scatter plots",
			"properties": map[string]any{
				"series": map[string]any{
					"type":        "array",
					"description": "Data series for scatter plot",
					"minItems":    1,
					"items": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type":        "string",
								"description": "Series name for legend",
							},
							"x": map[string]any{
								"type":        "array",
								"description": "X-axis values",
								"minItems":    1,
								"items": map[string]any{
									"type": "number",
								},
							},
							"y": map[string]any{
								"type":        "array",
								"description": "Y-axis values",
								"minItems":    1,
								"items": map[string]any{
									"type": "number",
								},
							},
						},
						"required":             []string{"name", "x", "y"},
						"additionalProperties": false,
					},
				},
			},
			"required":             []string{"series"},
			"additionalProperties": false,
		},
		"histogram_data": map[string]any{
			"type":        "object",
			"description": "Data for histograms",
			"properties": map[string]any{
				"values": map[string]any{
					"type":        "array",
					"description": "Raw values to bin into histogram",
					"minItems":    1,
					"items": map[string]any{
						"type": "number",
					},
				},
				"bins": map[string]any{
					"type":        "integer",
					"description": "Number of histogram bins (default 10)",
				},
			},
			"required":             []string{"values"},
			"additionalProperties": false,
		},
	},
	"required":             []string{"name", "description", "chart_type"},
	"additionalProperties": false,
}

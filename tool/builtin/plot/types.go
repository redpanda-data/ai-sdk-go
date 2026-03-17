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

// Input represents the input parameters for creating a plot.
type Input struct {
	// Artifact metadata (required for all plots)
	Name        string `json:"name"        jsonschema:"Name for the plot artifact"`
	Description string `json:"description" jsonschema:"Description of what the plot shows"`

	// Chart configuration
	ChartType string   `json:"chart_type"        jsonschema:"Type of chart to generate (line, bar, scatter, histogram)"`
	Title     string   `json:"title,omitempty"   jsonschema:"Chart title"`
	XLabel    string   `json:"x_label,omitempty" jsonschema:"X-axis label"`
	YLabel    string   `json:"y_label,omitempty" jsonschema:"Y-axis label"`
	Options   *Options `json:"options,omitempty" jsonschema:"Chart rendering options"`

	// Type-specific data (only one should be set based on chart_type)
	LineData      *LineData      `json:"line_data,omitempty"      jsonschema:"Data for line charts"`
	BarData       *BarData       `json:"bar_data,omitempty"       jsonschema:"Data for bar charts"`
	ScatterData   *ScatterData   `json:"scatter_data,omitempty"   jsonschema:"Data for scatter plots"`
	HistogramData *HistogramData `json:"histogram_data,omitempty" jsonschema:"Data for histograms"`
}

// Options contains rendering options for the chart.
type Options struct {
	Width  int  `json:"width,omitempty"  jsonschema:"Chart width in pixels (default 800)"`
	Height int  `json:"height,omitempty" jsonschema:"Chart height in pixels (default 600)"`
	Legend bool `json:"legend,omitempty" jsonschema:"Show legend (default true)"`
	Grid   bool `json:"grid,omitempty"   jsonschema:"Show grid lines (default true)"`
}

// LineData contains data for line charts.
type LineData struct {
	Series []XYSeries `json:"series" jsonschema:"Data series for line chart (at least one series required)"`
}

// BarData contains data for bar charts.
type BarData struct {
	Categories []string      `json:"categories" jsonschema:"Category labels for X-axis (at least one category required)"`
	Series     []ValueSeries `json:"series"     jsonschema:"Data series for bar chart (at least one series required)"`
}

// ScatterData contains data for scatter plots.
type ScatterData struct {
	Series []XYSeries `json:"series" jsonschema:"Data series for scatter plot (at least one series required)"`
}

// HistogramData contains data for histograms.
type HistogramData struct {
	Values []float64 `json:"values"         jsonschema:"Raw values to bin into histogram (at least one value required)"`
	Bins   int       `json:"bins,omitempty" jsonschema:"Number of histogram bins (default 10)"`
}

// XYSeries represents a data series with X and Y coordinates.
type XYSeries struct {
	Name string    `json:"name" jsonschema:"Series name for legend"`
	X    []float64 `json:"x"    jsonschema:"X-axis values (at least one value required)"`
	Y    []float64 `json:"y"    jsonschema:"Y-axis values (at least one value required)"`
}

// ValueSeries represents a data series with only Y values (X is implicit from categories).
type ValueSeries struct {
	Name   string    `json:"name"   jsonschema:"Series name for legend"`
	Values []float64 `json:"values" jsonschema:"Data values (at least one value required)"`
}

// Output represents the output from the plot tool
// The PNGData field contains the base64-encoded PNG that will be extracted
// by the reconciler and stored as an artifact. Only ArtifactID is returned to the LLM.
type Output struct {
	ArtifactID string `json:"artifact_id"`
	PNGData    string `json:"png_data"`  // base64-encoded PNG (extracted by reconciler)
	MimeType   string `json:"mime_type"` // always "image/png"
	Filename   string `json:"filename"`  // default "chart.png"
	Width      int    `json:"width"`     // chart width in pixels
	Height     int    `json:"height"`    // chart height in pixels
}

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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTool_Definition(t *testing.T) {
	t.Parallel()

	tool := New()
	def := tool.Definition()

	assert.Equal(t, "plot", def.Name)
	assert.NotEmpty(t, def.Description)
	assert.NotEmpty(t, def.Parameters)

	// Verify schema is valid JSON
	var schema map[string]any

	err := json.Unmarshal(def.Parameters, &schema)
	require.NoError(t, err)
}

func TestTool_Execute_LineChart(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Test Line Chart",
		Description: "A test line chart visualization",
		ChartType:   "line",
		Title:       "Test Line Chart",
		XLabel:      "X Axis",
		YLabel:      "Y Axis",
		LineData: &LineData{
			Series: []XYSeries{
				{
					Name: "Series 1",
					X:    []float64{1, 2, 3, 4, 5},
					Y:    []float64{10, 20, 15, 30, 25},
				},
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	// Verify output structure
	assert.NotEmpty(t, output.ArtifactID)
	assert.Equal(t, "image/png", output.MimeType)
	assert.Equal(t, "chart.png", output.Filename)
	assert.Equal(t, defaultWidth, output.Width)
	assert.Equal(t, defaultHeight, output.Height)

	// Verify PNG data is valid base64
	pngData, err := base64.StdEncoding.DecodeString(output.PNGData)
	require.NoError(t, err)
	assert.NotEmpty(t, pngData)

	// Verify PNG magic bytes
	assert.Equal(t, []byte{0x89, 0x50, 0x4E, 0x47}, pngData[:4])
}

func TestTool_Execute_BarChart(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Test Bar Chart",
		Description: "A test bar chart visualization",
		ChartType:   "bar",
		Title:       "Test Bar Chart",
		BarData: &BarData{
			Categories: []string{"A", "B", "C"},
			Series: []ValueSeries{
				{
					Name:   "Series 1",
					Values: []float64{10, 20, 30},
				},
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	assert.NotEmpty(t, output.ArtifactID)
	assert.NotEmpty(t, output.PNGData)
}

func TestTool_Execute_ScatterChart(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Test Scatter Plot",
		Description: "A test scatter plot visualization",
		ChartType:   "scatter",
		Title:       "Test Scatter Plot",
		ScatterData: &ScatterData{
			Series: []XYSeries{
				{
					Name: "Points",
					X:    []float64{1, 2, 3, 4, 5},
					Y:    []float64{2, 4, 3, 5, 6},
				},
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	assert.NotEmpty(t, output.ArtifactID)
	assert.NotEmpty(t, output.PNGData)
}

func TestTool_Execute_Histogram(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Test Histogram",
		Description: "A test histogram visualization",
		ChartType:   "histogram",
		Title:       "Test Histogram",
		HistogramData: &HistogramData{
			Values: []float64{1.2, 2.3, 1.5, 3.4, 2.1, 2.8, 1.9, 3.1},
			Bins:   5,
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	assert.NotEmpty(t, output.ArtifactID)
	assert.NotEmpty(t, output.PNGData)
}

func TestTool_Execute_CustomDimensions(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Test Chart",
		Description: "A test chart with custom dimensions",
		ChartType:   "line",
		Options: &Options{
			Width:  1000,
			Height: 500,
		},
		LineData: &LineData{
			Series: []XYSeries{
				{
					Name: "Test",
					X:    []float64{1, 2, 3},
					Y:    []float64{1, 2, 3},
				},
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	assert.Equal(t, 1000, output.Width)
	assert.Equal(t, 500, output.Height)
}

func TestTool_Execute_ValidationErrors(t *testing.T) {
	t.Parallel()

	tool := New()

	tests := []struct {
		name  string
		input Input
	}{
		{
			name: "missing name",
			input: Input{
				Description: "Test",
				ChartType:   "line",
				LineData: &LineData{
					Series: []XYSeries{
						{
							Name: "Test",
							X:    []float64{1, 2, 3},
							Y:    []float64{1, 2, 3},
						},
					},
				},
			},
		},
		{
			name: "missing description",
			input: Input{
				Name:      "Test",
				ChartType: "line",
				LineData: &LineData{
					Series: []XYSeries{
						{
							Name: "Test",
							X:    []float64{1, 2, 3},
							Y:    []float64{1, 2, 3},
						},
					},
				},
			},
		},
		{
			name: "missing line data",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "line",
			},
		},
		{
			name: "missing bar data",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "bar",
			},
		},
		{
			name: "missing scatter data",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "scatter",
			},
		},
		{
			name: "missing histogram data",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "histogram",
			},
		},
		{
			name: "invalid chart type",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "invalid",
			},
		},
		{
			name: "mismatched x/y lengths",
			input: Input{
				Name:        "Test",
				Description: "Test",
				ChartType:   "line",
				LineData: &LineData{
					Series: []XYSeries{
						{
							Name: "Test",
							X:    []float64{1, 2, 3},
							Y:    []float64{1, 2},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			inputJSON, err := json.Marshal(tt.input)
			require.NoError(t, err)

			_, err = tool.Execute(context.Background(), inputJSON)
			assert.Error(t, err)
		})
	}
}

func TestTool_Execute_MultipleSeries(t *testing.T) {
	t.Parallel()

	tool := New()

	input := Input{
		Name:        "Multiple Series Chart",
		Description: "A chart with multiple data series",
		ChartType:   "line",
		Title:       "Multiple Series",
		LineData: &LineData{
			Series: []XYSeries{
				{
					Name: "Series 1",
					X:    []float64{1, 2, 3, 4},
					Y:    []float64{10, 20, 15, 25},
				},
				{
					Name: "Series 2",
					X:    []float64{1, 2, 3, 4},
					Y:    []float64{15, 25, 20, 30},
				},
				{
					Name: "Series 3",
					X:    []float64{1, 2, 3, 4},
					Y:    []float64{5, 10, 12, 8},
				},
			},
		},
	}

	inputJSON, err := json.Marshal(input)
	require.NoError(t, err)

	outputJSON, err := tool.Execute(context.Background(), inputJSON)
	require.NoError(t, err)

	var output Output

	err = json.Unmarshal(outputJSON.Output, &output)
	require.NoError(t, err)

	assert.NotEmpty(t, output.PNGData)
}

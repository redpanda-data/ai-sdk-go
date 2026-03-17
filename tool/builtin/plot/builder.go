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
	"bytes"
	"errors"
	"fmt"
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
)

const (
	defaultWidth  = 800
	defaultHeight = 600
	defaultBins   = 10
	minWidth      = 200
	maxWidth      = 4000
	minHeight     = 200
	maxHeight     = 4000
)

// buildChart creates a plot based on the input configuration.
func buildChart(input Input) (*plot.Plot, int, int, error) {
	// Validate input
	if err := validateInput(input); err != nil {
		return nil, 0, 0, err
	}

	// Extract dimensions
	width, height := extractDimensions(input.Options)

	// Create plot based on chart type
	var (
		p   *plot.Plot
		err error
	)

	switch input.ChartType {
	case "line":
		p, err = buildLineChart(input)
	case "bar":
		p, err = buildBarChart(input)
	case "scatter":
		p, err = buildScatterChart(input)
	case "histogram":
		p, err = buildHistogram(input)
	default:
		return nil, 0, 0, fmt.Errorf("unsupported chart type: %s", input.ChartType)
	}

	if err != nil {
		return nil, 0, 0, err
	}

	// Apply styling defaults
	applyDefaultStyling(p)

	// Apply common options
	if input.Title != "" {
		p.Title.Text = input.Title
	}

	if input.XLabel != "" {
		p.X.Label.Text = input.XLabel
	}

	if input.YLabel != "" {
		p.Y.Label.Text = input.YLabel
	}

	// Show grid if requested (default true)
	if input.Options == nil || input.Options.Grid {
		grid := plotter.NewGrid()
		grid.Horizontal.Color = color.RGBA{R: 220, G: 220, B: 220, A: 255}
		grid.Vertical.Color = color.RGBA{R: 220, G: 220, B: 220, A: 255}
		grid.Horizontal.Width = vg.Points(0.5)
		grid.Vertical.Width = vg.Points(0.5)
		p.Add(grid)
	}

	return p, width, height, nil
}

// buildLineChart creates a line chart from LineData.
func buildLineChart(input Input) (*plot.Plot, error) {
	if input.LineData == nil {
		return nil, errors.New("line_data is required for line charts")
	}

	p := plot.New()

	// Add each series
	for i, series := range input.LineData.Series {
		if len(series.X) != len(series.Y) {
			return nil, fmt.Errorf("series %q: X and Y arrays must have same length", series.Name)
		}

		pts := make(plotter.XYs, len(series.X))
		for j := range series.X {
			pts[j].X = series.X[j]
			pts[j].Y = series.Y[j]
		}

		line, err := plotter.NewLine(pts)
		if err != nil {
			return nil, fmt.Errorf("failed to create line for series %q: %w", series.Name, err)
		}

		// Set color and style
		line.Color = getSeriesColor(i)
		line.Width = vg.Points(2.5)

		p.Add(line)

		if shouldShowLegend(input.Options) {
			p.Legend.Add(series.Name, line)
		}
	}

	return p, nil
}

// buildBarChart creates a bar chart from BarData.
func buildBarChart(input Input) (*plot.Plot, error) {
	if input.BarData == nil {
		return nil, errors.New("bar_data is required for bar charts")
	}

	if len(input.BarData.Categories) == 0 {
		return nil, errors.New("at least one category is required")
	}

	p := plot.New()

	// Validate all series have correct length
	for _, series := range input.BarData.Series {
		if len(series.Values) != len(input.BarData.Categories) {
			return nil, fmt.Errorf("series %q: values length must match categories length", series.Name)
		}
	}

	// Create bars for each series
	barWidth := vg.Points(20)
	numSeries := len(input.BarData.Series)

	for seriesIdx, series := range input.BarData.Series {
		values := make(plotter.Values, len(series.Values))
		copy(values, series.Values)

		bars, err := plotter.NewBarChart(values, barWidth)
		if err != nil {
			return nil, fmt.Errorf("failed to create bar chart for series %q: %w", series.Name, err)
		}

		// Position bars side by side for multiple series
		bars.Color = getSeriesColor(seriesIdx)
		bars.Offset = vg.Points(float64(seriesIdx-numSeries/2) * float64(barWidth))

		p.Add(bars)

		if shouldShowLegend(input.Options) {
			p.Legend.Add(series.Name, bars)
		}
	}

	// Set category labels on X axis
	p.NominalX(input.BarData.Categories...)

	return p, nil
}

// buildScatterChart creates a scatter plot from ScatterData.
func buildScatterChart(input Input) (*plot.Plot, error) {
	if input.ScatterData == nil {
		return nil, errors.New("scatter_data is required for scatter plots")
	}

	p := plot.New()

	// Add each series
	for i, series := range input.ScatterData.Series {
		if len(series.X) != len(series.Y) {
			return nil, fmt.Errorf("series %q: X and Y arrays must have same length", series.Name)
		}

		pts := make(plotter.XYs, len(series.X))
		for j := range series.X {
			pts[j].X = series.X[j]
			pts[j].Y = series.Y[j]
		}

		scatter, err := plotter.NewScatter(pts)
		if err != nil {
			return nil, fmt.Errorf("failed to create scatter for series %q: %w", series.Name, err)
		}

		// Set color and style
		scatter.Color = getSeriesColor(i)
		scatter.Shape = draw.CircleGlyph{}
		scatter.Radius = vg.Points(4)
		scatter.Color = getSeriesColor(i)

		p.Add(scatter)

		if shouldShowLegend(input.Options) {
			p.Legend.Add(series.Name, scatter)
		}
	}

	return p, nil
}

// buildHistogram creates a histogram from HistogramData.
func buildHistogram(input Input) (*plot.Plot, error) {
	if input.HistogramData == nil {
		return nil, errors.New("histogram_data is required for histograms")
	}

	if len(input.HistogramData.Values) == 0 {
		return nil, errors.New("histogram values cannot be empty")
	}

	p := plot.New()

	bins := defaultBins
	if input.HistogramData.Bins > 0 {
		bins = input.HistogramData.Bins
	}

	values := make(plotter.Values, len(input.HistogramData.Values))
	copy(values, input.HistogramData.Values)

	hist, err := plotter.NewHist(values, bins)
	if err != nil {
		return nil, fmt.Errorf("failed to create histogram: %w", err)
	}

	hist.FillColor = getSeriesColor(0)
	hist.LineStyle.Width = vg.Points(0.5)
	hist.Color = color.RGBA{R: 100, G: 100, B: 100, A: 255}

	p.Add(hist)

	return p, nil
}

// renderToPNG renders a plot to PNG bytes.
func renderToPNG(p *plot.Plot, width, height int) ([]byte, error) {
	// Create image canvas
	img := vgimg.New(vg.Points(float64(width)), vg.Points(float64(height)))
	dc := draw.New(img)

	// Draw plot to canvas
	p.Draw(dc)

	// Encode to PNG
	var buf bytes.Buffer

	pngCanvas := vgimg.PngCanvas{Canvas: img}
	if _, err := pngCanvas.WriteTo(&buf); err != nil {
		return nil, fmt.Errorf("failed to encode PNG: %w", err)
	}

	return buf.Bytes(), nil
}

// validateInput validates the plot input.
func validateInput(input Input) error {
	// Validate chart type has corresponding data
	switch input.ChartType {
	case "line":
		if input.LineData == nil {
			return errors.New("line_data is required when chart_type is 'line'")
		}

		if len(input.LineData.Series) == 0 {
			return errors.New("at least one series is required")
		}
	case "bar":
		if input.BarData == nil {
			return errors.New("bar_data is required when chart_type is 'bar'")
		}

		if len(input.BarData.Series) == 0 {
			return errors.New("at least one series is required")
		}
	case "scatter":
		if input.ScatterData == nil {
			return errors.New("scatter_data is required when chart_type is 'scatter'")
		}

		if len(input.ScatterData.Series) == 0 {
			return errors.New("at least one series is required")
		}
	case "histogram":
		if input.HistogramData == nil {
			return errors.New("histogram_data is required when chart_type is 'histogram'")
		}
	default:
		return fmt.Errorf("unsupported chart type: %s", input.ChartType)
	}

	// Validate dimensions if provided
	if input.Options != nil {
		if input.Options.Width != 0 && (input.Options.Width < minWidth || input.Options.Width > maxWidth) {
			return fmt.Errorf("width must be between %d and %d pixels", minWidth, maxWidth)
		}

		if input.Options.Height != 0 && (input.Options.Height < minHeight || input.Options.Height > maxHeight) {
			return fmt.Errorf("height must be between %d and %d pixels", minHeight, maxHeight)
		}
	}

	return nil
}

// extractDimensions extracts width and height from options or uses defaults.
func extractDimensions(opts *Options) (int, int) {
	width := defaultWidth
	height := defaultHeight

	if opts != nil {
		if opts.Width > 0 {
			width = opts.Width
		}

		if opts.Height > 0 {
			height = opts.Height
		}
	}

	return width, height
}

// shouldShowLegend determines if legend should be shown.
func shouldShowLegend(opts *Options) bool {
	if opts == nil {
		return true // default to showing legend
	}

	return opts.Legend
}

// applyDefaultStyling applies consistent styling defaults to all plots.
func applyDefaultStyling(p *plot.Plot) {
	// Title styling
	p.Title.TextStyle.Font.Size = vg.Points(16)

	// Axis label styling
	p.X.Label.TextStyle.Font.Size = vg.Points(12)
	p.Y.Label.TextStyle.Font.Size = vg.Points(12)

	// Tick label styling
	p.X.Tick.Label.Font.Size = vg.Points(10)
	p.Y.Tick.Label.Font.Size = vg.Points(10)

	// Legend styling
	p.Legend.TextStyle.Font.Size = vg.Points(10)
	p.Legend.Top = true
	p.Legend.Left = false

	// Add padding for better appearance
	p.Title.Padding = vg.Points(10)
	p.X.Padding = vg.Points(5)
	p.Y.Padding = vg.Points(5)
}

// getSeriesColor returns a color for the given series index.
func getSeriesColor(idx int) color.Color {
	// Modern, vibrant color palette with good contrast
	colors := []color.Color{
		color.RGBA{R: 31, G: 119, B: 180, A: 255},  // Professional blue
		color.RGBA{R: 255, G: 127, B: 14, A: 255},  // Orange
		color.RGBA{R: 44, G: 160, B: 44, A: 255},   // Green
		color.RGBA{R: 214, G: 39, B: 40, A: 255},   // Red
		color.RGBA{R: 148, G: 103, B: 189, A: 255}, // Purple
		color.RGBA{R: 140, G: 86, B: 75, A: 255},   // Brown
		color.RGBA{R: 227, G: 119, B: 194, A: 255}, // Pink
		color.RGBA{R: 127, G: 127, B: 127, A: 255}, // Gray
		color.RGBA{R: 188, G: 189, B: 34, A: 255},  // Olive
		color.RGBA{R: 23, G: 190, B: 207, A: 255},  // Cyan
	}

	return colors[idx%len(colors)]
}

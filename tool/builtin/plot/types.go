package plot

// Input represents the input parameters for creating a plot.
type Input struct {
	// Artifact metadata (required for all plots)
	Name        string `json:"name"        jsonschema:"required,description=Name for the plot artifact"`
	Description string `json:"description" jsonschema:"required,description=Description of what the plot shows"`

	// Chart configuration
	ChartType string   `json:"chart_type"        jsonschema:"required,enum=line,enum=bar,enum=scatter,enum=histogram,description=Type of chart to generate"`
	Title     string   `json:"title,omitempty"   jsonschema:"description=Chart title"`
	XLabel    string   `json:"x_label,omitempty" jsonschema:"description=X-axis label"`
	YLabel    string   `json:"y_label,omitempty" jsonschema:"description=Y-axis label"`
	Options   *Options `json:"options,omitempty" jsonschema:"description=Chart rendering options"`

	// Type-specific data (only one should be set based on chart_type)
	LineData      *LineData      `json:"line_data,omitempty"      jsonschema:"description=Data for line charts"`
	BarData       *BarData       `json:"bar_data,omitempty"       jsonschema:"description=Data for bar charts"`
	ScatterData   *ScatterData   `json:"scatter_data,omitempty"   jsonschema:"description=Data for scatter plots"`
	HistogramData *HistogramData `json:"histogram_data,omitempty" jsonschema:"description=Data for histograms"`
}

// Options contains rendering options for the chart.
type Options struct {
	Width  int  `json:"width,omitempty"  jsonschema:"description=Chart width in pixels (default 800)"`
	Height int  `json:"height,omitempty" jsonschema:"description=Chart height in pixels (default 600)"`
	Legend bool `json:"legend,omitempty" jsonschema:"description=Show legend (default true)"`
	Grid   bool `json:"grid,omitempty"   jsonschema:"description=Show grid lines (default true)"`
}

// LineData contains data for line charts.
type LineData struct {
	Series []XYSeries `json:"series" jsonschema:"required,minItems=1,description=Data series for line chart"`
}

// BarData contains data for bar charts.
type BarData struct {
	Categories []string      `json:"categories" jsonschema:"required,minItems=1,description=Category labels for X-axis"`
	Series     []ValueSeries `json:"series"     jsonschema:"required,minItems=1,description=Data series for bar chart"`
}

// ScatterData contains data for scatter plots.
type ScatterData struct {
	Series []XYSeries `json:"series" jsonschema:"required,minItems=1,description=Data series for scatter plot"`
}

// HistogramData contains data for histograms.
type HistogramData struct {
	Values []float64 `json:"values"         jsonschema:"required,minItems=1,description=Raw values to bin into histogram"`
	Bins   int       `json:"bins,omitempty" jsonschema:"description=Number of histogram bins (default 10)"`
}

// XYSeries represents a data series with X and Y coordinates.
type XYSeries struct {
	Name string    `json:"name" jsonschema:"required,description=Series name for legend"`
	X    []float64 `json:"x"    jsonschema:"required,minItems=1,description=X-axis values"`
	Y    []float64 `json:"y"    jsonschema:"required,minItems=1,description=Y-axis values"`
}

// ValueSeries represents a data series with only Y values (X is implicit from categories).
type ValueSeries struct {
	Name   string    `json:"name"   jsonschema:"required,description=Series name for legend"`
	Values []float64 `json:"values" jsonschema:"required,minItems=1,description=Data values"`
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

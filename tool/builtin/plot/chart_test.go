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
	"context"
	"encoding/base64"
	"encoding/json"
	"image/png"
	"maps"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestChartToolReturnsValidatedDataWithoutPNG(t *testing.T) {
	t.Parallel()

	input := json.RawMessage(`{"name":"Requests","description":"Supplied regional request counts","chart":{"type":"bar","title":"Regional requests","data":{"labels":["US","EU"],"datasets":[{"label":"Requests","data":[120,80]}]}}}`)
	instance := NewChart()
	assert.Equal(t, "plot", instance.Definition().Name)
	raw, err := instance.Execute(t.Context(), input)
	require.NoError(t, err)
	var output struct {
		ArtifactID string          `json:"artifact_id"`
		MimeType   string          `json:"mime_type"`
		Chart      json.RawMessage `json:"chart"`
	}
	require.NoError(t, json.Unmarshal(raw, &output))
	assert.NotEmpty(t, output.ArtifactID)
	assert.Equal(t, "application/vnd.redpanda.chart+json", output.MimeType)
	assert.JSONEq(t, `{"type":"bar","title":"Regional requests","data":{"labels":["US","EU"],"datasets":[{"label":"Requests","data":[120,80]}]}}`, string(output.Chart))
	assert.NotContains(t, string(raw), "png_data")
}

func TestChartToolRejectsDuplicateFields(t *testing.T) {
	t.Parallel()
	_, err := NewChart().Execute(t.Context(), json.RawMessage(`{"name":"Gauge","description":"Supplied values","chart":{"type":"gauge","data":{"value":50,"min":40,"min":0,"max":100}}}`))
	require.Error(t, err, "duplicate fields must not silently change chart values")
}

func TestChartToolAcceptsStrictProviderArguments(t *testing.T) {
	t.Parallel()

	providers := map[string]func(*testing.T, *llm.Request) map[string]any{
		"openai responses": func(t *testing.T, request *llm.Request) map[string]any {
			t.Helper()

			mapped, err := openai.NewRequestMapper(&openai.Config{ModelName: "test-model"}).ToProvider(request)
			require.NoError(t, err)
			require.Len(t, mapped.Tools, 1)
			require.NotNil(t, mapped.Tools[0].OfFunction)
			require.True(t, mapped.Tools[0].OfFunction.Strict.Value)

			return mapped.Tools[0].OfFunction.Parameters
		},
		"openai compatible chat": func(t *testing.T, request *llm.Request) map[string]any {
			t.Helper()

			mapped, err := openaicompat.NewRequestMapper(&openaicompat.Config{ModelName: "test-model"}).ToProvider(request)
			require.NoError(t, err)
			require.Len(t, mapped.Tools, 1)
			require.NotNil(t, mapped.Tools[0].OfFunction)
			require.True(t, mapped.Tools[0].OfFunction.Function.Strict.Value)

			return mapped.Tools[0].OfFunction.Function.Parameters
		},
	}
	samples := map[string]struct{ chart, want string }{
		"bar with missing observation": {
			`{"type":"bar","title":null,"layout":null,"data":{"unit":null,"labels":[9007199254740993,"B"],"datasets":[{"label":"Count","data":[1.0000000000000002,null],"bounds":null}]}}`,
			`{"type":"bar","data":{"labels":[9007199254740993,"B"],"datasets":[{"label":"Count","data":[1.0000000000000002,null]}]}}`,
		},
		"gauge": {
			`{"type":"gauge","title":null,"data":{"value":65,"min":0,"max":100,"label":null}}`,
			`{"type":"gauge","data":{"value":65,"min":0,"max":100}}`,
		},
		"treemap": {
			`{"type":"treemap","title":null,"data":{"nodes":[{"id":"root","label":"All","parent":null,"value":null},{"id":"leaf","label":"A","parent":"root","value":5}]}}`,
			`{"type":"treemap","data":{"nodes":[{"id":"root","label":"All"},{"id":"leaf","label":"A","parent":"root","value":5}]}}`,
		},
		"sunburst": {
			`{"type":"sunburst","title":null,"data":{"nodes":[{"id":"root","label":"All","parent":null,"value":null},{"id":"leaf","label":"A","parent":"root","value":5}]}}`,
			`{"type":"sunburst","data":{"nodes":[{"id":"root","label":"All"},{"id":"leaf","label":"A","parent":"root","value":5}]}}`,
		},
		"timeline": {
			`{"type":"timeline","title":null,"data":{"unit":"ms","intervals":[{"label":"Tool","start":10,"end":80}]}}`,
			`{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"Tool","start":10,"end":80}]}}`,
		},
	}

	for name, chart := range map[string]string{
		"waterfall": `{"type":"waterfall","data":{"unit":"USD","steps":[{"label":"Before","kind":"total","value":0.1},{"label":"Added","kind":"delta","value":0.2},{"label":"After","kind":"total","value":0.3}]}}`,
		"ecdf":      `{"type":"ecdf","data":{"unit":"ms","datasets":[{"label":"Requests","data":[500,100,500]}]}}`,
		"nodeLink":  `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"Isolated"}],"links":[]}}`,
	} {
		samples[name] = struct{ chart, want string }{strings.Replace(chart, `"data":`, `"title":null,"data":`, 1), chart}
	}

	samples["bullet"] = struct{ chart, want string }{`{"type":"bullet","title":null,"data":{"unit":"ms","min":0,"max":1000,"measures":[{"label":"Search","value":250,"target":500},{"label":"Generate","value":750,"target":600}]}}`, `{"type":"bullet","data":{"unit":"ms","min":0,"max":1000,"measures":[{"label":"Search","value":250,"target":500},{"label":"Generate","value":750,"target":600}]}}`}
	samples["funnel"] = struct{ chart, want string }{`{"type":"funnel","title":null,"data":{"cohort":"100 requests submitted in this sample","stages":[{"label":"Submitted","count":100},{"label":"Authorized","count":80},{"label":"Completed","count":60},{"label":"Successful","count":50}]}}`, `{"type":"funnel","data":{"cohort":"100 requests submitted in this sample","stages":[{"label":"Submitted","count":100},{"label":"Authorized","count":80},{"label":"Completed","count":60},{"label":"Successful","count":50}]}}`}
	samples["violin"] = struct{ chart, want string }{`{"type":"violin","title":null,"data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`, `{"type":"violin","data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`}
	samples["ridgeline"] = struct{ chart, want string }{`{"type":"ridgeline","title":null,"data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`, `{"type":"ridgeline","data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`}

	samples["uncertaintyLine"] = struct{ chart, want string }{`{"type":"line","title":"Evaluation score (synthetic data)","data":{"labels":["A","B","C"],"datasets":[{"label":"Score","data":[60,80,70],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}],"unit":null},"layout":null}`, `{"type":"line","title":"Evaluation score (synthetic data)","data":{"labels":["A","B","C"],"datasets":[{"label":"Score","data":[60,80,70],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}]}}`}
	samples["uncertaintyScatter"] = struct{ chart, want string }{`{"type":"scatter","title":"Evaluation score (synthetic data)","data":{"datasets":[{"label":"Score","data":[{"x":1,"y":60},{"x":2,"y":80},{"x":3,"y":70}],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}],"unit":null},"layout":null}`, `{"type":"scatter","title":"Evaluation score (synthetic data)","data":{"datasets":[{"label":"Score","data":[{"x":1,"y":60},{"x":2,"y":80},{"x":3,"y":70}],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}]}}`}
	samples["smallMultiples"] = struct{ chart, want string }{`{"type":"bar","layout":"smallMultiples","title":"Token usage (synthetic data)","data":{"unit":"tokens","labels":["A","B"],"datasets":[{"label":"Agent one","data":[100,50],"bounds":null},{"label":"Agent two","data":[10,5],"bounds":null}]}}`, `{"type":"bar","layout":"smallMultiples","title":"Token usage (synthetic data)","data":{"unit":"tokens","labels":["A","B"],"datasets":[{"label":"Agent one","data":[100,50]},{"label":"Agent two","data":[10,5]}]}}`}

	for name, mapper := range providers {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			instance := NewChart()
			parameters := mapper(t, &llm.Request{Tools: []llm.ToolDefinition{instance.Definition()}})
			schemaJSON, err := json.Marshal(parameters)
			require.NoError(t, err)
			var schema jsonschema.Schema
			require.NoError(t, json.Unmarshal(schemaJSON, &schema))
			resolved, err := schema.Resolve(nil)
			require.NoError(t, err)

			for sample, input := range samples {
				t.Run(sample, func(t *testing.T) {
					t.Parallel()

					args := chartArguments(input.chart)
					var value any
					require.NoError(t, json.Unmarshal(args, &value))
					require.NoError(t, resolved.Validate(value), "arguments must satisfy the provider's strict schema")
					raw, err := instance.Execute(t.Context(), args)
					require.NoError(t, err, "strict-provider optional nulls must be accepted")
					var output ChartOutput
					require.NoError(t, json.Unmarshal(raw, &output))
					assert.JSONEq(t, input.want, string(output.Chart), "omit absent optional members, not missing array observations")

					if sample == "bar with missing observation" {
						assert.Contains(t, string(output.Chart), "9007199254740993", "normalization must not round supplied numbers")
					}
				})
			}
		})
	}
}

func chartArguments(chart string) json.RawMessage {
	return json.RawMessage(`{"name":"Supplied data","description":"Known example values","chart":` + chart + `}`)
}

func TestChartToolSupportedTypesPreserveData(t *testing.T) {
	t.Parallel()

	tabular := `{"labels":["A","B"],"datasets":[{"label":"Count","data":[5,3]}]}`

	cases := map[string]string{}
	for _, kind := range []string{"bar", "horizontalBar", "stackedBar", "percentBar", "line", "area", "stackedArea", "pie", "doughnut", "polarArea"} {
		cases[kind] = tabular
	}

	cases["radar"] = `{"labels":["A","B","C"],"datasets":[{"label":"Comparable counts","data":[5,3,4]}]}`
	cases["scatter"] = `{"datasets":[{"label":"Requests","data":[{"x":10,"y":2}]}]}`
	cases["bubble"] = `{"datasets":[{"label":"Requests","data":[{"x":10,"y":2,"size":5}]}]}`
	cases["histogram"] = `{"bins":[{"start":0,"end":10,"count":4},{"start":10,"end":20,"count":7}]}`
	cases["heatmap"] = `{"cells":[{"x":"Mon","y":"A","value":4}]}`
	cases["boxPlot"] = `{"boxes":[{"label":"A","min":1,"q1":2,"median":4,"q3":6,"max":9}]}`

	cases["gauge"] = `{"value":65,"min":0,"max":100,"label":"Budget (%)"}`
	for _, kind := range []string{"treemap", "sunburst"} {
		cases[kind] = `{"nodes":[{"id":"all","label":"All"},{"id":"a","parent":"all","label":"A","value":5}]}`
	}

	cases["sankey"] = `{"nodes":[{"id":"a","label":"Agent"},{"id":"b","label":"Tool"}],"links":[{"source":"a","target":"b","value":5}]}`

	cases["timeline"] = `{"unit":"ms","intervals":[{"label":"Search","start":10,"end":80},{"label":"Search","start":20,"end":60},{"label":"Done","start":80,"end":80}]}`
	cases["waterfall"] = `{"unit":"USD","steps":[{"label":"Before","kind":"total","value":100},{"label":"Compute","kind":"delta","value":30},{"label":"Savings","kind":"delta","value":-10},{"label":"After","kind":"total","value":120}]}`
	cases["ecdf"] = `{"unit":"ms","datasets":[{"label":"Requests","data":[500,100,500,900]}]}`

	cases["nodeLink"] = `{"nodes":[{"id":"a","label":"Agent"},{"id":"b","label":"Tool"},{"id":"c","label":"Isolated"}],"links":[{"source":"a","target":"b"},{"source":"b","target":"a"}]}`
	for kind, data := range cases {
		t.Run(kind, func(t *testing.T) {
			t.Parallel()

			chart := `{"type":"` + kind + `","data":` + data + `}`
			raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
			require.NoError(t, err)
			var output ChartOutput
			require.NoError(t, json.Unmarshal(raw, &output))
			assert.Equal(t, ChartMediaType, output.MimeType)
			assert.JSONEq(t, chart, string(output.Chart), "the backend must not invent, normalize or rewrite supplied observations")
		})
	}
}

func TestChartToolAcceptsIndependentCoordinateRanges(t *testing.T) {
	t.Parallel()

	for kind, point := range map[string]string{
		"scatter": `{"x":-1e308,"y":1e308}`,
		"bubble":  `{"x":-1e308,"y":1e308,"size":5}`,
	} {
		t.Run(kind, func(t *testing.T) {
			t.Parallel()

			chart := `{"type":"` + kind + `","data":{"datasets":[{"label":"Coordinates","data":[` + point + `]}]}}`
			raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
			require.NoError(t, err, "each axis has a finite range even though the combined x/y range overflows")
			var output ChartOutput
			require.NoError(t, json.Unmarshal(raw, &output))
			assert.JSONEq(t, chart, string(output.Chart))
		})
	}
}

func TestChartToolPreservesEmptyCompositionData(t *testing.T) {
	t.Parallel()

	for _, kind := range []string{"pie", "doughnut", "polarArea", "percentBar"} {
		for name, values := range map[string]string{"zero": "[0,0]", "missing": "[null,null]"} {
			t.Run(kind+"/"+name, func(t *testing.T) {
				t.Parallel()

				chart := `{"type":"` + kind + `","data":{"labels":["A","B"],"datasets":[{"label":"Count","data":` + values + `}]}}`
				raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
				require.NoError(t, err)
				var output ChartOutput
				require.NoError(t, json.Unmarshal(raw, &output))
				assert.JSONEq(t, chart, string(output.Chart), "empty composition data is preserved, not normalized into percentages")
			})
		}
	}
}

func TestChartToolRejectsInvalidCharts(t *testing.T) {
	t.Parallel()

	cases := map[string]string{
		"waterfall invented baseline":   `{"type":"waterfall","data":{"unit":"USD","steps":[{"label":"Change","kind":"delta","value":20}]}}`,
		"waterfall inconsistent total":  `{"type":"waterfall","data":{"unit":"USD","steps":[{"label":"Before","kind":"total","value":100},{"label":"Change","kind":"delta","value":20},{"label":"After","kind":"total","value":150}]}}`,
		"waterfall cumulative overflow": `{"type":"waterfall","data":{"unit":"USD","steps":[{"label":"Before","kind":"total","value":1e308},{"label":"Change","kind":"delta","value":1e308}]}}`,
		"waterfall range overflow":      `{"type":"waterfall","data":{"unit":"USD","steps":[{"label":"Before","kind":"total","value":-1e308},{"label":"Change","kind":"delta","value":1.5e308},{"label":"More","kind":"delta","value":1e308}]}}`,
		"ECDF null sample":              `{"type":"ecdf","data":{"unit":"ms","datasets":[{"label":"Requests","data":[100,null]}]}}`,
		"ECDF summary not samples":      `{"type":"ecdf","data":{"unit":"ms","datasets":[{"label":"Requests","data":{"p95":500}}]}}`,
		"ECDF range overflow":           `{"type":"ecdf","data":{"unit":"ms","datasets":[{"label":"Requests","data":[-1e308,1e308]}]}}`,
		"node-link duplicate ID":        `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"A"},{"id":"a","label":"B"}],"links":[]}}`,
		"node-link missing endpoint":    `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"A"}],"links":[{"source":"a","target":"b"}]}}`,
		"node-link self link":           `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"A"}],"links":[{"source":"a","target":"a"}]}}`,
		"node-link repeated edge":       `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"A"},{"id":"b","label":"B"}],"links":[{"source":"a","target":"b"},{"source":"a","target":"b"}]}}`,
		"node-link weighted flow":       `{"type":"nodeLink","data":{"nodes":[{"id":"a","label":"A"},{"id":"b","label":"B"}],"links":[{"source":"a","target":"b","value":100}]}}`,
		"missing timeline unit":         `{"type":"timeline","data":{"intervals":[{"label":"Task","start":0,"end":1}]}}`,
		"blank timeline unit":           `{"type":"timeline","data":{"unit":" ","intervals":[{"label":"Task","start":0,"end":1}]}}`,
		"blank timeline label":          `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":" ","start":0,"end":1}]}}`,
		"unfinished timeline interval":  `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"Task","start":0,"end":null}]}}`,
		"reversed timeline interval":    `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"Task","start":2,"end":1}]}}`,
		"string timeline offset":        `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"Task","start":"0","end":1}]}}`,
		"timeline range overflow":       `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"First","start":-1e308,"end":-1e308},{"label":"Last","start":1e308,"end":1e308}]}}`,
		"timeline unknown property":     `{"type":"timeline","data":{"unit":"ms","intervals":[{"label":"Task","start":0,"end":1,"color":"red"}]}}`,
		"null required gauge min":       `{"type":"gauge","data":{"value":65,"min":null,"max":100}}`,
		"unknown null property":         `{"type":"gauge","data":{"value":65,"min":0,"max":100,"script":null}}`,
		"null hierarchy leaf value":     `{"type":"treemap","data":{"nodes":[{"id":"all","label":"All","parent":null,"value":null}]}}`,
		"missing gauge min":             `{"type":"gauge","data":{"value":65,"max":100}}`,
		"gauge outside range":           `{"type":"gauge","data":{"value":101,"min":0,"max":100}}`,
		"gauge reversed bounds":         `{"type":"gauge","data":{"value":0,"min":10,"max":0}}`,
		"gauge range overflow":          `{"type":"gauge","data":{"value":0,"min":-1e308,"max":1e308}}`,
		"number overflow":               `{"type":"gauge","data":{"value":1e400,"min":0,"max":100}}`,
		"renderer script":               `{"type":"bar","data":{"labels":["A"],"datasets":[{"label":"Count","data":[5]}]},"script":"alert(1)"}`,
		"unknown type":                  `{"type":"plotly","data":{}}`,
		"wrong number type":             `{"type":"bar","data":{"labels":["A"],"datasets":[{"label":"Count","data":["5"]}]}}`,
		"missing dataset values":        `{"type":"bar","data":{"labels":["A","B"],"datasets":[{"label":"Count","data":[5]}]}}`,
		"negative pie":                  `{"type":"pie","data":{"labels":["A"],"datasets":[{"label":"Count","data":[-5]}]}}`,
		"multiple pie datasets":         `{"type":"pie","data":{"labels":["A"],"datasets":[{"label":"One","data":[5]},{"label":"Two","data":[3]}]}}`,
		"negative percentage":           `{"type":"percentBar","data":{"labels":["A"],"datasets":[{"label":"Count","data":[-5]}]}}`,
		"radar missing axes":            `{"type":"radar","data":{"labels":["A"],"datasets":[{"label":"Count","data":[5]}]}}`,
		"negative radar":                `{"type":"radar","data":{"labels":["A","B","C"],"datasets":[{"label":"Count","data":[5,-1,3]}]}}`,
		"total overflow":                `{"type":"bar","data":{"labels":["A","B"],"datasets":[{"label":"Count","data":[1e308,1e308]}]}}`,
		"bubble missing size":           `{"type":"bubble","data":{"datasets":[{"label":"Count","data":[{"x":1,"y":2}]}]}}`,
		"bubble nonpositive size":       `{"type":"bubble","data":{"datasets":[{"label":"Count","data":[{"x":1,"y":2,"size":0}]}]}}`,
		"x coordinate range overflow":   `{"type":"scatter","data":{"datasets":[{"label":"Count","data":[{"x":-1e308,"y":0},{"x":1e308,"y":1}]}]}}`,
		"y coordinate range overflow":   `{"type":"bubble","data":{"datasets":[{"label":"Count","data":[{"x":0,"y":-1e308,"size":1},{"x":1,"y":1e308,"size":1}]}]}}`,
		"histogram range overflow":      `{"type":"histogram","data":{"bins":[{"start":-1e308,"end":-5e307,"count":1},{"start":5e307,"end":1e308,"count":1}]}}`,
		"overlapping bins":              `{"type":"histogram","data":{"bins":[{"start":0,"end":10,"count":4},{"start":5,"end":15,"count":7}]}}`,
		"unequal bins":                  `{"type":"histogram","data":{"bins":[{"start":0,"end":10,"count":4},{"start":10,"end":30,"count":7}]}}`,
		"negative bin":                  `{"type":"histogram","data":{"bins":[{"start":0,"end":10,"count":-1}]}}`,
		"duplicate heatmap cell":        `{"type":"heatmap","data":{"cells":[{"x":"A","y":"B","value":1},{"x":"A","y":"B","value":2}]}}`,
		"unordered box":                 `{"type":"boxPlot","data":{"boxes":[{"label":"A","min":1,"q1":4,"median":2,"q3":6,"max":9}]}}`,
		"missing hierarchy parent":      `{"type":"treemap","data":{"nodes":[{"id":"all","label":"All","value":0},{"id":"a","parent":"unknown","label":"A","value":5}]}}`,
		"hierarchy double count":        `{"type":"treemap","data":{"nodes":[{"id":"all","label":"All","value":5},{"id":"a","parent":"all","label":"A","value":5}]}}`,
		"hierarchy missing leaf value":  `{"type":"treemap","data":{"nodes":[{"id":"all","label":"All"}]}}`,
		"hierarchy cycle":               `{"type":"sunburst","data":{"nodes":[{"id":"root","label":"Root","value":1},{"id":"a","label":"A","parent":"b"},{"id":"b","label":"B","parent":"a"}]}}`,
		"multiple roots":                `{"type":"treemap","data":{"nodes":[{"id":"a","label":"A","value":1},{"id":"b","label":"B","value":2}]}}`,
		"duplicate node":                `{"type":"sankey","data":{"nodes":[{"id":"a","label":"A"},{"id":"a","label":"Again"}],"links":[{"source":"a","target":"a","value":1}]}}`,
		"missing flow endpoint":         `{"type":"sankey","data":{"nodes":[{"id":"a","label":"A"}],"links":[{"source":"a","target":"b","value":1}]}}`,
		"flow cycle":                    `{"type":"sankey","data":{"nodes":[{"id":"a","label":"A"},{"id":"b","label":"B"}],"links":[{"source":"a","target":"b","value":1},{"source":"b","target":"a","value":1}]}}`,
	}
	for name, chart := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
			require.Error(t, err)
			assert.Empty(t, raw, "invalid input must not create a partial artifact")
		})
	}
}

func TestChartToolBoundsTimelineIntervals(t *testing.T) {
	t.Parallel()

	for _, size := range []int{500, 501} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			t.Parallel()

			intervals := make([]map[string]any, size)
			for i := range intervals {
				intervals[i] = map[string]any{"label": "Task", "start": i, "end": i + 1}
			}

			chart, err := json.Marshal(map[string]any{"type": "timeline", "data": map[string]any{"unit": "ms", "intervals": intervals}})
			require.NoError(t, err)

			raw, err := NewChart().Execute(t.Context(), chartArguments(string(chart)))
			if size == 500 {
				require.NoError(t, err)
				var output ChartOutput
				require.NoError(t, json.Unmarshal(raw, &output))
				assert.JSONEq(t, string(chart), string(output.Chart))
			} else {
				require.Error(t, err)
				assert.Empty(t, raw)
			}
		})
	}
}

func TestChartToolBoundsAndCancellation(t *testing.T) {
	t.Parallel()
	t.Run("payload", func(t *testing.T) {
		t.Parallel()
		_, err := NewChart().Execute(t.Context(), json.RawMessage(strings.Repeat(" ", MaxChartInputBytes+1)))
		require.Error(t, err)
	})
	t.Run("cancellation", func(t *testing.T) {
		t.Parallel()
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		_, err := NewChart().Execute(ctx, chartArguments(`{"type":"gauge","data":{"value":1,"min":0,"max":10}}`))
		require.ErrorIs(t, err, context.Canceled)
	})

	for _, size := range []int{100, 101} {
		t.Run(strconv.Itoa(size), func(t *testing.T) {
			t.Parallel()

			labels := make([]string, size)
			values := make([]int, size)

			for i := range labels {
				labels[i] = strconv.Itoa(i)
				values[i] = i
			}

			datasets := make([]any, 20)
			for i := range datasets {
				datasets[i] = map[string]any{"label": strconv.Itoa(i), "data": values}
			}

			chart, err := json.Marshal(map[string]any{"type": "bar", "data": map[string]any{"labels": labels, "datasets": datasets}})
			require.NoError(t, err)

			_, err = NewChart().Execute(t.Context(), chartArguments(string(chart)))
			if size == 100 {
				require.NoError(t, err, "100 labels × 20 series is the supported 2000-cell boundary")
			} else {
				require.Error(t, err, "101 labels × 20 series exceeds the 2000-cell boundary")
			}
		})
	}
}

func TestChartErrorsReachTheToolCallerAndAllowCorrection(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(NewChart()))
	failed, err := registry.Execute(t.Context(), &llm.ToolRequestPart{ID: "bad", Name: "plot", Arguments: chartArguments(`{"type":"gauge","data":{"value":65,"max":100}}`)})
	require.NoError(t, err)
	require.True(t, failed.IsError)
	var failure struct {
		Error string `json:"error"`
	}
	require.NoError(t, json.Unmarshal(failed.Result, &failure))
	assert.Contains(t, failure.Error, `"min"`, "the caller must know which gauge field to supply")
	assert.NotContains(t, string(failed.Result), "artifact_id")
	corrected, err := registry.Execute(t.Context(), &llm.ToolRequestPart{ID: "good", Name: "plot", Arguments: chartArguments(`{"type":"gauge","data":{"value":65,"min":0,"max":100}}`)})
	require.NoError(t, err)
	require.False(t, corrected.IsError)
	assert.Contains(t, string(corrected.Result), ChartMediaType)
}

func TestLegacyPNGContract(t *testing.T) {
	t.Parallel()

	samples := map[string]string{
		"line":      `"line_data":{"series":[{"name":"Count","x":[1,2],"y":[5,3]}]}`,
		"bar":       `"bar_data":{"categories":["A","B"],"series":[{"name":"Count","values":[5,3]}]}`,
		"scatter":   `"scatter_data":{"series":[{"name":"Count","x":[1,2],"y":[5,3]}]}`,
		"histogram": `"histogram_data":{"values":[1,2,3,4],"bins":2}`,
	}
	for kind, data := range samples {
		t.Run(kind, func(t *testing.T) {
			t.Parallel()

			instance := New()
			var schema struct {
				Properties map[string]json.RawMessage `json:"properties"`
			}
			require.NoError(t, json.Unmarshal(instance.Definition().Parameters, &schema))
			assert.Contains(t, schema.Properties, "chart_type")
			assert.NotContains(t, schema.Properties, "chart")

			args := json.RawMessage(`{"name":"Legacy plot","description":"Original contract","chart_type":"` + kind + `","options":{"width":600,"height":400},` + data + `}`)
			raw, err := instance.Execute(t.Context(), args)
			require.NoError(t, err)
			var fields map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(raw, &fields))
			assert.ElementsMatch(t, []string{"artifact_id", "mime_type", "filename", "png_data", "width", "height"}, slices.Collect(maps.Keys(fields)))
			var output Output
			require.NoError(t, json.Unmarshal(raw, &output))
			assert.Equal(t, "image/png", output.MimeType)
			assert.Equal(t, "chart.png", output.Filename)
			assert.NotEmpty(t, output.ArtifactID)
			pngBytes, err := base64.StdEncoding.DecodeString(output.PNGData)
			require.NoError(t, err)
			image, err := png.Decode(bytes.NewReader(pngBytes))
			require.NoError(t, err)
			assert.Equal(t, 600, output.Width)
			assert.Equal(t, 400, output.Height)
			assert.Positive(t, image.Bounds().Dx())
			assert.Positive(t, image.Bounds().Dy())
		})
	}
}

func TestChartToolNewKindCardinalityLimits(t *testing.T) {
	t.Parallel()

	steps := make([]map[string]any, 500)
	for index := range steps {
		kind := "delta"
		if index == 0 {
			kind = "total"
		}

		steps[index] = map[string]any{"label": "Step", "kind": kind, "value": 0}
	}

	nodes := make([]map[string]any, 100)
	for index := range nodes {
		nodes[index] = map[string]any{"id": strconv.Itoa(index), "label": "Node"}
	}

	links := make([]map[string]any, 500)
	for index := range links {
		links[index] = map[string]any{"source": strconv.Itoa(index % 100), "target": strconv.Itoa((index%100 + index/100 + 1) % 100)}
	}

	for _, tc := range []struct {
		name, kind, field string
		data              map[string]any
		extra             any
	}{
		{"steps", "waterfall", "steps", map[string]any{"unit": "USD", "steps": steps}, steps[0]},
		{"samples", "ecdf", "datasets", map[string]any{"unit": "ms", "datasets": []map[string]any{{"label": "One", "data": make([]float64, 1000)}, {"label": "Two", "data": make([]float64, 1000)}}}, map[string]any{"label": "Extra", "data": []float64{1}}},
		{"nodes", "nodeLink", "nodes", map[string]any{"nodes": nodes, "links": []any{}}, map[string]any{"id": "extra", "label": "Extra"}},
		{"links", "nodeLink", "links", map[string]any{"nodes": nodes, "links": links}, map[string]any{"source": "0", "target": "6"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			chart, err := json.Marshal(map[string]any{"type": tc.kind, "data": tc.data})
			require.NoError(t, err)
			_, err = NewChart().Execute(t.Context(), chartArguments(string(chart)))
			require.NoError(t, err, "documented cardinality boundary must work")
			var data map[string]any
			require.NoError(t, json.Unmarshal(chart, &data))
			fields, ok := data["data"].(map[string]any)
			require.True(t, ok)
			values, ok := fields[tc.field].([]any)
			require.True(t, ok)

			fields[tc.field] = append(values, tc.extra)
			oversized, err := json.Marshal(data)
			require.NoError(t, err)
			_, err = NewChart().Execute(t.Context(), chartArguments(string(oversized)))
			require.Error(t, err, "one beyond the documented boundary must reject")
		})
	}
}

func TestChartRegistryRejectsOversizeArtifactsAsToolErrors(t *testing.T) {
	t.Parallel()

	for name, label := range map[string]string{
		"large input":    strings.Repeat("x", 100_100),
		"escaped output": strings.Repeat("<", 20_000),
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			chart, err := json.Marshal(map[string]any{"type": "bar", "data": map[string]any{
				"labels": []string{label}, "datasets": []any{map[string]any{"label": "Count", "data": []int{1}}},
			}})
			require.NoError(t, err)
			// Keep literal HTML in input to cover growth during safe JSON encoding.
			chart = bytes.ReplaceAll(chart, []byte(`\u003c`), []byte("<"))
			registry := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, registry.Register(NewChart()))
			result, err := registry.Execute(t.Context(), &llm.ToolRequestPart{ID: "large", Name: "plot", Arguments: chartArguments(string(chart))})
			require.NoError(t, err)
			require.True(t, result.IsError, "oversize artifacts must be actionable errors, not silent successful fallback notes")
			assert.Contains(t, string(result.Result), "bytes")
			assert.NotContains(t, string(result.Result), "artifact_id")
			corrected, err := registry.Execute(t.Context(), &llm.ToolRequestPart{ID: "corrected", Name: "plot", Arguments: chartArguments(`{"type":"bar","data":{"labels":["A"],"datasets":[{"label":"Count","data":[1]}]}}`)})
			require.NoError(t, err)
			require.False(t, corrected.IsError)
			assert.Contains(t, string(corrected.Result), ChartMediaType)
		})
	}
}

func TestChartBulletPreservesTargets(t *testing.T) {
	t.Parallel()

	chart := `{"type":"bullet","data":{"unit":"ms","min":0,"max":1000,"measures":[{"label":"Latency","value":250,"target":500}]}}`
	raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
	require.NoError(t, err)
	var output ChartOutput
	require.NoError(t, json.Unmarshal(raw, &output))
	assert.JSONEq(t, chart, string(output.Chart))
}

func TestChartComparisonsPreserveSuppliedFacts(t *testing.T) {
	t.Parallel()

	for name, chart := range map[string]string{
		"funnel":             `{"type":"funnel","title":"Request completion (synthetic data)","data":{"cohort":"100 requests submitted in this sample","stages":[{"label":"Submitted","count":100},{"label":"Authorized","count":80},{"label":"Completed","count":60},{"label":"Successful","count":50}]}}`,
		"violin":             `{"type":"violin","title":"Latency distribution (synthetic data)","data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`,
		"ridgeline":          `{"type":"ridgeline","title":"Latency distribution (synthetic data)","data":{"unit":"ms","thresholds":[0,100,200,300,400],"datasets":[{"label":"Before","data":[20,100,140,160,210,230,270,400]},{"label":"After","data":[10,40,60,80,120,150,200,250]}]}}`,
		"uncertaintyLine":    `{"type":"line","title":"Evaluation score (synthetic data)","data":{"labels":["A","B","C"],"datasets":[{"label":"Score","data":[60,80,70],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}]}}`,
		"uncertaintyScatter": `{"type":"scatter","title":"Evaluation score (synthetic data)","data":{"datasets":[{"label":"Score","data":[{"x":1,"y":60},{"x":2,"y":80},{"x":3,"y":70}],"bounds":{"label":"Measured min/max","lower":[50,75,60],"upper":[70,85,90]}}]}}`,
		"smallMultiples":     `{"type":"bar","layout":"smallMultiples","title":"Token usage (synthetic data)","data":{"unit":"tokens","labels":["A","B"],"datasets":[{"label":"Agent one","data":[100,50]},{"label":"Agent two","data":[10,5]}]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			raw, err := NewChart().Execute(t.Context(), chartArguments(chart))
			require.NoError(t, err)
			var out ChartOutput
			require.NoError(t, json.Unmarshal(raw, &out))
			assert.JSONEq(t, chart, string(out.Chart))
		})
	}
}

func TestChartComparisonsRejectMisleadingData(t *testing.T) {
	t.Parallel()

	for name, chart := range map[string]string{
		"bounds and unbounded estimate range": `{"type":"line","data":{"labels":["A","B"],"datasets":[{"label":"A","data":[-1e+308,null],"bounds":{"label":"Range","lower":[-1e+308,null],"upper":[-1e+308,null]}},{"label":"B","data":[null,1e+308]}]}}`,
		"blank unicode label":                 `{"type":"bullet","data":{"unit":"ms","min":0,"max":10,"measures":[{"label":"\u00a0","value":1,"target":2}]}}`,
		"bounds combined range":               `{"type":"line","data":{"labels":["A"],"datasets":[{"label":"A","data":[0],"bounds":{"label":"Range","lower":[-1e+308],"upper":[0]}},{"label":"B","data":[0],"bounds":{"label":"Range","lower":[0],"upper":[1e+308]}}]}}`,
		"bullet outside target":               `{"type":"bullet","data":{"unit":"ms","min":0,"max":100,"measures":[{"label":"A","value":10,"target":101}]}}`,
		"funnel increase":                     `{"type":"funnel","data":{"cohort":"same requests","stages":[{"label":"A","count":10},{"label":"B","count":11}]}}`,
		"funnel fractional":                   `{"type":"funnel","data":{"cohort":"same requests","stages":[{"label":"A","count":0.5}]}}`,
		"funnel unsafe count":                 `{"type":"funnel","data":{"cohort":"same requests","stages":[{"label":"A","count":9007199254740992}]}}`,
		"duplicate edges":                     `{"type":"violin","data":{"unit":"ms","thresholds":[0,100,100],"datasets":[{"label":"A","data":[10]}]}}`,
		"outside samples":                     `{"type":"ridgeline","data":{"unit":"ms","thresholds":[0,100],"datasets":[{"label":"A","data":[101]}]}}`,
		"nonfinite density":                   `{"type":"violin","data":{"unit":"ms","thresholds":[0,5e-324],"datasets":[{"label":"A","data":[0]}]}}`,
		"bounds wrong length":                 `{"type":"line","data":{"labels":["A"],"datasets":[{"label":"A","data":[10],"bounds":{"label":"Measured range","lower":[],"upper":[20]}}]}}`,
		"bounds wrong order":                  `{"type":"scatter","data":{"datasets":[{"label":"A","data":[{"x":1,"y":10}],"bounds":{"label":"Measured range","lower":[11],"upper":[20]}}]}}`,
		"bounds missing sample":               `{"type":"line","data":{"labels":["A"],"datasets":[{"label":"A","data":[null],"bounds":{"label":"Measured range","lower":[0],"upper":[20]}}]}}`,
		"bounds on unsupported kind":          `{"type":"bar","data":{"labels":["A"],"datasets":[{"label":"A","data":[10],"bounds":{"label":"Measured range","lower":[0],"upper":[20]}}]}}`,
		"facets missing units":                `{"type":"bar","layout":"smallMultiples","data":{"labels":["A"],"datasets":[{"label":"A","data":[10]}]}}`,
		"facets unsupported kind":             `{"type":"area","layout":"smallMultiples","data":{"unit":"ms","labels":["A"],"datasets":[{"label":"A","data":[10]}]}}`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := NewChart().Execute(t.Context(), chartArguments(chart))
			require.Error(t, err)
		})
	}
}

func TestChartEncodedOutputSizeBoundary(t *testing.T) {
	t.Parallel()

	makeArgs := func(label string) json.RawMessage {
		chart, err := json.Marshal(map[string]any{"type": "bar", "data": map[string]any{"labels": []string{label}, "datasets": []any{map[string]any{"label": "Count", "data": []int{1}}}}})
		require.NoError(t, err)

		return chartArguments(string(chart))
	}
	base, err := NewChart().Execute(t.Context(), makeArgs(""))
	require.NoError(t, err)

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(NewChart()))

	for _, size := range []int{100_000 - 1, 100_000, 100_000 + 1} {
		response, err := registry.Execute(t.Context(), &llm.ToolRequestPart{ID: "boundary", Name: "plot", Arguments: makeArgs(strings.Repeat("x", size-len(base)))})
		require.NoError(t, err)

		if size > 100_000 {
			require.True(t, response.IsError)
			continue
		}

		require.False(t, response.IsError)
		assert.Len(t, response.Result, size)
		assert.Contains(t, string(response.Result), ChartMediaType)
	}
}

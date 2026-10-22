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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
)

const (
	chartTypeLine      = "line"
	chartTypeBar       = "bar"
	chartTypeScatter   = "scatter"
	chartTypeHistogram = "histogram"
	chartTypeRadar     = "radar"
)

type chartEnvelope struct {
	Layout string          `json:"layout"`
	Type   string          `json:"type"`
	Data   json.RawMessage `json:"data"`
}

type chartNode struct {
	ID     string   `json:"id"`
	Parent string   `json:"parent"`
	Value  *float64 `json:"value"`
}

type chartLink struct {
	Source string  `json:"source"`
	Target string  `json:"target"`
	Value  float64 `json:"value"`
}

// normalizeChart runs only after schema validation, so null object members can
// only be declared optional fields. Required and unknown null fields still fail.
// Preserve null array observations and numeric tokens, including large integers.
func normalizeChart(raw json.RawMessage) (json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()

	var value any
	if err := decoder.Decode(&value); err != nil {
		return nil, err
	}

	omitNullChartMembers(value)

	return json.Marshal(value)
}

func omitNullChartMembers(value any) {
	switch node := value.(type) {
	case map[string]any:
		for key, child := range node {
			if child == nil {
				delete(node, key)
			} else {
				omitNullChartMembers(child)
			}
		}
	case []any:
		for _, child := range node {
			omitNullChartMembers(child)
		}
	}
}

func decodeChartData[T any](raw json.RawMessage) (T, error) {
	var data T
	err := json.Unmarshal(raw, &data)

	return data, err
}

func validateChartSemantics(raw json.RawMessage) error {
	envelope, err := decodeChartData[chartEnvelope](raw)
	if err != nil {
		return err
	}

	if err := validateComparisonOptions(envelope.Type, envelope.Layout, envelope.Data); err != nil {
		return err
	}

	if validate := comparisonValidator(envelope.Type); validate != nil {
		return validate(envelope.Data)
	}

	switch envelope.Type {
	case chartTypeBar, "horizontalBar", "stackedBar", "percentBar", chartTypeLine, "area", "stackedArea", "pie", "doughnut", chartTypeRadar, "polarArea":
		return validateTabularChart(envelope.Type, envelope.Data)
	case chartTypeScatter, "bubble":
		data, err := decodeChartData[struct {
			Datasets []struct {
				Data []struct {
					X float64 `json:"x"`
					Y float64 `json:"y"`
				} `json:"data"`
			} `json:"datasets"`
		}](envelope.Data)
		if err != nil {
			return err
		}
		var xs, ys []float64

		count := 0
		for _, dataset := range data.Datasets {
			count += len(dataset.Data)
			for _, point := range dataset.Data {
				xs = append(xs, point.X)
				ys = append(ys, point.Y)
			}
		}

		if count > 2000 || !finiteRange(xs) || !finiteRange(ys) {
			return errors.New("coordinates require at most 2000 points and a finite range on each axis")
		}
	case chartTypeHistogram:
		return validateHistogramChart(envelope.Data)
	case "heatmap":
		return validateHeatmapChart(envelope.Data)
	case "boxPlot":
		return validateBoxPlotChart(envelope.Data)
	case "gauge":
		data, err := decodeChartData[struct {
			Value float64 `json:"value"`
			Min   float64 `json:"min"`
			Max   float64 `json:"max"`
		}](envelope.Data)
		if err != nil {
			return err
		}

		if !(data.Min < data.Max && data.Value >= data.Min && data.Value <= data.Max) || !finite(data.Max-data.Min) {
			return errors.New("gauge requires finite min < max with value inside those bounds")
		}
	case "waterfall":
		return validateWaterfallChart(envelope.Data)
	case "ecdf":
		return validateECDFChart(envelope.Data)
	case "nodeLink":
		return validateNodeLinkChart(envelope.Data)
	case "timeline":
		return validateTimelineChart(envelope.Data)
	case "treemap", "sunburst":
		data, err := decodeChartData[struct {
			Nodes []chartNode `json:"nodes"`
		}](envelope.Data)
		if err != nil {
			return err
		}

		return validateChartHierarchy(data.Nodes)
	case "sankey":
		data, err := decodeChartData[struct {
			Nodes []chartNode `json:"nodes"`
			Links []chartLink `json:"links"`
		}](envelope.Data)
		if err != nil {
			return err
		}

		return validateChartGraph(data.Nodes, data.Links)
	default:
		return fmt.Errorf("unsupported chart type %q", envelope.Type)
	}

	return nil
}

func validateBoxPlotChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Boxes []struct {
			Min    float64 `json:"min"`
			Q1     float64 `json:"q1"`
			Median float64 `json:"median"`
			Q3     float64 `json:"q3"`
			Max    float64 `json:"max"`
		} `json:"boxes"`
	}](raw)
	if err != nil {
		return err
	}

	for _, box := range data.Boxes {
		if !(box.Min <= box.Q1 && box.Q1 <= box.Median && box.Median <= box.Q3 && box.Q3 <= box.Max) || !finite(box.Max-box.Min) {
			return errors.New("box summaries must be ordered and have a finite range")
		}
	}

	return nil
}

func validateTimelineChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Unit      string `json:"unit"`
		Intervals []struct {
			Label string  `json:"label"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"intervals"`
	}](raw)
	if err != nil {
		return err
	}

	if strings.TrimSpace(data.Unit) == "" {
		return errors.New("timeline unit must not be blank")
	}

	endpoints := make([]float64, 0, len(data.Intervals)*2)
	for i, interval := range data.Intervals {
		if strings.TrimSpace(interval.Label) == "" || interval.Start > interval.End {
			return fmt.Errorf("timeline interval %d requires a nonblank label and start <= end", i+1)
		}

		endpoints = append(endpoints, interval.Start, interval.End)
	}

	if !finiteRange(endpoints) {
		return errors.New("timeline endpoints must have a finite overall range")
	}

	return nil
}

func validateHeatmapChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Cells []struct {
			X     string  `json:"x"`
			Y     string  `json:"y"`
			Value float64 `json:"value"`
		} `json:"cells"`
	}](raw)
	if err != nil {
		return err
	}

	seen := make(map[[2]string]bool)
	var values []float64

	for _, cell := range data.Cells {
		pair := [2]string{cell.X, cell.Y}
		if seen[pair] {
			return errors.New("heatmap x/y pairs must be unique")
		}

		seen[pair] = true

		values = append(values, cell.Value)
	}

	if !finiteRange(values) {
		return errors.New("heatmap range must be finite")
	}

	return nil
}

func validateHistogramChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Bins []struct {
			Start float64 `json:"start"`
			End   float64 `json:"end"`
			Count float64 `json:"count"`
		} `json:"bins"`
	}](raw)
	if err != nil {
		return err
	}

	total, width := 0.0, 0.0

	for i, bin := range data.Bins {
		current := bin.End - bin.Start
		if current <= 0 || !finite(current) {
			return fmt.Errorf("bin %d must have positive finite width", i+1)
		}

		if i == 0 {
			width = current
		}

		if i > 0 && bin.Start < data.Bins[i-1].End {
			return errors.New("bins must be ordered and nonoverlapping")
		}

		if math.Abs(current-width) > math.Max(current, width)*1e-9 {
			return errors.New("bins must have equal widths")
		}

		total += bin.Count
	}

	if !finite(total) {
		return errors.New("histogram total must be finite")
	}

	if !finite(data.Bins[len(data.Bins)-1].End - data.Bins[0].Start) {
		return errors.New("histogram axis range must be finite")
	}

	return nil
}

func validateTabularChart(kind string, raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Labels   []json.RawMessage `json:"labels"`
		Datasets []struct {
			Data []*float64 `json:"data"`
		} `json:"datasets"`
	}](raw)
	if err != nil {
		return err
	}

	if len(data.Labels)*len(data.Datasets) > 2000 {
		return errors.New("tabular charts support at most 2000 rendered values")
	}

	single := kind == "pie" || kind == "doughnut" || kind == "polarArea"
	if single && len(data.Datasets) != 1 {
		return fmt.Errorf("%s requires one dataset", kind)
	}

	if kind == chartTypeRadar && len(data.Labels) < 3 {
		return errors.New("radar requires at least three axes")
	}

	totals := make([]float64, len(data.Labels))
	for _, dataset := range data.Datasets {
		if len(dataset.Data) != len(data.Labels) {
			return errors.New("each dataset must have one value per label; use null for missing values")
		}

		total := 0.0

		for i, value := range dataset.Data {
			if value == nil {
				continue
			}

			if (single || kind == chartTypeRadar || kind == "percentBar") && *value < 0 {
				return fmt.Errorf("%s values must be nonnegative", kind)
			}

			total += math.Abs(*value)

			totals[i] += math.Abs(*value)
			if !finite(total) || !finite(totals[i]) {
				return errors.New("chart totals must be finite")
			}
		}
	}

	return nil
}

func validateChartHierarchy(nodes []chartNode) error {
	children := make(map[string]int)
	links := make([]chartLink, 0, len(nodes))
	roots := 0

	for _, node := range nodes {
		if node.Parent == "" {
			roots++
		} else {
			children[node.Parent]++
			links = append(links, chartLink{Source: node.Parent, Target: node.ID})
		}
	}

	if roots != 1 {
		return errors.New("hierarchy requires exactly one root")
	}

	total := 0.0

	for _, node := range nodes {
		if children[node.ID] > 0 {
			if node.Value != nil {
				return errors.New("internal hierarchy nodes must omit value")
			}
		} else {
			if node.Value == nil {
				return errors.New("hierarchy leaves require value")
			}

			total += *node.Value
		}
	}

	if !finite(total) {
		return errors.New("hierarchy leaf total must be finite")
	}

	return validateChartGraph(nodes, links)
}

func validateChartGraph(nodes []chartNode, links []chartLink) error {
	indegree := make(map[string]int, len(nodes))
	for _, node := range nodes {
		if _, exists := indegree[node.ID]; exists {
			return errors.New("node ids must be unique")
		}

		indegree[node.ID] = 0
	}

	outgoing := make(map[string][]string, len(nodes))
	total := 0.0

	for _, edge := range links {
		if _, exists := indegree[edge.Source]; !exists {
			return fmt.Errorf("missing source node %q", edge.Source)
		}

		if _, exists := indegree[edge.Target]; !exists {
			return fmt.Errorf("missing target node %q", edge.Target)
		}

		indegree[edge.Target]++
		outgoing[edge.Source] = append(outgoing[edge.Source], edge.Target)
		total += edge.Value
	}

	if !finite(total) {
		return errors.New("flow total must be finite")
	}
	var ready []string

	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}

	visited := 0

	for len(ready) > 0 {
		id := ready[len(ready)-1]
		ready = ready[:len(ready)-1]
		visited++

		for _, next := range outgoing[id] {
			indegree[next]--
			if indegree[next] == 0 {
				ready = append(ready, next)
			}
		}
	}

	if visited != len(nodes) {
		return errors.New("chart graph must not contain cycles")
	}

	return nil
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func finiteRange(values []float64) bool {
	if len(values) == 0 {
		return true
	}

	low, high := values[0], values[0]
	for _, value := range values {
		low = math.Min(low, value)
		high = math.Max(high, value)
	}

	return finite(high - low)
}

// validateChartJSON rejects ambiguous duplicate keys before a decoder can discard them.
func validateChartJSON(raw json.RawMessage) error {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var walk func(int) error

	walk = func(depth int) error {
		if depth > 64 {
			return errors.New("chart JSON nesting exceeds 64 levels")
		}

		token, err := decoder.Token()
		if err != nil {
			return err
		}

		switch token {
		case json.Delim('{'):
			keys := make(map[string]bool)

			for decoder.More() {
				token, err := decoder.Token()
				if err != nil {
					return err
				}

				key, ok := token.(string)
				if !ok {
					return errors.New("object key must be a string")
				}

				if keys[key] {
					return fmt.Errorf("duplicate object key %q", key)
				}

				keys[key] = true

				if err := walk(depth + 1); err != nil {
					return err
				}
			}

			_, err = decoder.Token()

			return err
		case json.Delim('['):
			for decoder.More() {
				if err := walk(depth + 1); err != nil {
					return err
				}
			}

			_, err = decoder.Token()

			return err
		}

		return nil
	}
	if err := walk(0); err != nil {
		return err
	}

	if _, err := decoder.Token(); err != io.EOF {
		return errors.New("chart input must contain exactly one JSON value")
	}

	return nil
}

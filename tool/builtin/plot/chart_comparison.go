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
	"encoding/json"
	"errors"
	"math"
	"sort"
	"strings"
)

func validateBulletChart(raw json.RawMessage) error {
	if err := validateComparisonText(raw); err != nil {
		return err
	}

	data, err := decodeChartData[struct {
		Min      float64 `json:"min"`
		Max      float64 `json:"max"`
		Measures []struct {
			Value  float64 `json:"value"`
			Target float64 `json:"target"`
		} `json:"measures"`
	}](raw)
	if err != nil {
		return err
	}

	if data.Min >= data.Max || !finite(data.Max-data.Min) {
		return errors.New("bullet requires min < max with a finite range")
	}

	for _, measure := range data.Measures {
		if measure.Value < data.Min || measure.Value > data.Max || measure.Target < data.Min || measure.Target > data.Max {
			return errors.New("bullet values and targets must lie inside the explicit min/max domain")
		}
	}

	return nil
}

func validateFunnelChart(raw json.RawMessage) error {
	if err := validateComparisonText(raw); err != nil {
		return err
	}

	data, err := decodeChartData[struct {
		Stages []struct {
			Count float64 `json:"count"`
		} `json:"stages"`
	}](raw)
	if err != nil {
		return err
	}

	previous := math.Inf(1)
	for _, stage := range data.Stages {
		if stage.Count > previous {
			return errors.New("funnel stage counts must not increase within the same cohort")
		}

		previous = stage.Count
	}

	return nil
}

func validateDistributionChart(raw json.RawMessage) error {
	if err := validateComparisonText(raw); err != nil {
		return err
	}

	data, err := decodeChartData[struct {
		Thresholds []float64 `json:"thresholds"`
		Datasets   []struct {
			Data []float64 `json:"data"`
		} `json:"datasets"`
	}](raw)
	if err != nil {
		return err
	}

	if !finiteRange(data.Thresholds) {
		return errors.New("distribution bin edges require a finite overall range")
	}

	for i := 1; i < len(data.Thresholds); i++ {
		if data.Thresholds[i] <= data.Thresholds[i-1] {
			return errors.New("distribution bin edges must be strictly increasing")
		}
	}

	total := 0
	for _, series := range data.Datasets {
		total += len(series.Data)
		if total > 2000 {
			return errors.New("distributions support at most 2000 total samples")
		}

		counts := make([]int, len(data.Thresholds)-1)
		for _, value := range series.Data {
			if value < data.Thresholds[0] || value > data.Thresholds[len(data.Thresholds)-1] {
				return errors.New("distribution samples must lie inside the explicit bin edges")
			}

			bin := sort.Search(len(data.Thresholds)-1, func(i int) bool { return value < data.Thresholds[i+1] })
			if bin == len(counts) {
				bin--
			}

			counts[bin]++
		}

		for i, count := range counts {
			if len(series.Data) > 0 && !finite((float64(count)/float64(len(series.Data)))/(data.Thresholds[i+1]-data.Thresholds[i])) {
				return errors.New("distribution density must remain finite; widen the bin edges")
			}
		}
	}

	return nil
}

type chartBounds struct {
	Label string     `json:"label"`
	Lower []*float64 `json:"lower"`
	Upper []*float64 `json:"upper"`
}

func validateComparisonOptions(kind, layout string, raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Unit     string `json:"unit"`
		Datasets []struct {
			Data   []json.RawMessage `json:"data"`
			Bounds *chartBounds      `json:"bounds"`
		} `json:"datasets"`
	}](raw)
	if err != nil {
		return err
	}

	if layout != "" && (kind != chartTypeBar && kind != chartTypeLine && kind != chartTypeScatter || strings.TrimSpace(data.Unit) == "" || len(data.Datasets) > 12) {
		return errors.New("small multiples require bar/line/scatter, a nonblank unit and at most 12 datasets")
	}

	var endpoints []float64

	hasBounds := layout != ""
	if hasBounds {
		endpoints = append(endpoints, 0)
	}

	for _, series := range data.Datasets {
		if kind == chartTypeLine || kind == chartTypeBar || kind == chartTypeScatter {
			for _, rawValue := range series.Data {
				value, err := chartObservation(rawValue, kind)
				if err != nil {
					return err
				}

				if value != nil {
					endpoints = append(endpoints, *value)
				}
			}
		}

		if series.Bounds == nil {
			continue
		}

		hasBounds = true

		if kind != chartTypeLine && kind != chartTypeScatter {
			return errors.New("uncertainty bounds are only supported on line and scatter charts")
		}

		if err := validateChartBounds(series.Bounds, series.Data, kind); err != nil {
			return err
		}

		for _, value := range series.Bounds.Lower {
			if value != nil {
				endpoints = append(endpoints, *value)
			}
		}

		for _, value := range series.Bounds.Upper {
			if value != nil {
				endpoints = append(endpoints, *value)
			}
		}
	}

	if hasBounds && !finiteRange(endpoints) {
		return errors.New("uncertainty bounds require a finite overall range across datasets")
	}

	return nil
}

func validateChartBounds(bounds *chartBounds, values []json.RawMessage, kind string) error {
	if strings.TrimSpace(bounds.Label) == "" {
		return errors.New("uncertainty bounds label must not be blank")
	}

	if len(bounds.Lower) != len(values) || len(bounds.Upper) != len(values) {
		return errors.New("uncertainty bounds require exactly one lower/upper value per observation")
	}
	var endpoints []float64

	for i, raw := range values {
		value, err := chartObservation(raw, kind)
		if err != nil {
			return err
		}

		low, high := bounds.Lower[i], bounds.Upper[i]
		if value == nil {
			if low != nil || high != nil {
				return errors.New("missing observations require null uncertainty bounds")
			}
		} else {
			if low == nil || high == nil || *low > *value || *high < *value || !finite(*high-*low) {
				return errors.New("uncertainty bounds must be finite and contain the observation")
			}

			endpoints = append(endpoints, *low, *high)
		}
	}

	if !finiteRange(endpoints) {
		return errors.New("uncertainty bounds require a finite overall range")
	}

	return nil
}

func comparisonValidator(kind string) func(json.RawMessage) error {
	switch kind {
	case "bullet":
		return validateBulletChart
	case "funnel":
		return validateFunnelChart
	case "violin", "ridgeline":
		return validateDistributionChart
	default:
		return nil
	}
}

func validateComparisonText(raw json.RawMessage) error {
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return err
	}
	var walk func(any) error
	walk = func(node any) error {
		switch child := node.(type) {
		case map[string]any:
			for _, v := range child {
				if err := walk(v); err != nil {
					return err
				}
			}
		case []any:
			for _, v := range child {
				if err := walk(v); err != nil {
					return err
				}
			}
		case string:
			if strings.TrimSpace(child) == "" {
				return errors.New("chart labels, cohort and unit must not be blank")
			}
		}

		return nil
	}

	return walk(value)
}

func chartObservation(raw json.RawMessage, kind string) (*float64, error) {
	if kind == chartTypeScatter {
		var point struct {
			Y *float64 `json:"y"`
		}
		err := json.Unmarshal(raw, &point)

		return point.Y, err
	}
	var value *float64
	err := json.Unmarshal(raw, &value)

	return value, err
}

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
)

func validateWaterfallChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Steps []struct {
			Kind  string  `json:"kind"`
			Value float64 `json:"value"`
		} `json:"steps"`
	}](raw)
	if err != nil {
		return err
	}

	total := 0.0
	endpoints := []float64{0}

	for index, step := range data.Steps {
		if index == 0 && step.Kind != "total" {
			return errors.New("waterfall first step must be an explicit total")
		}

		if step.Kind == "delta" {
			total += step.Value
		} else {
			if index > 0 && math.Abs(step.Value-total) > 1e-9*math.Max(1, math.Max(math.Abs(total), math.Abs(step.Value))) {
				return errors.New("waterfall total must equal preceding total plus deltas")
			}

			total = step.Value
		}

		if !finite(total) {
			return errors.New("waterfall cumulative total must remain finite")
		}

		endpoints = append(endpoints, total)
	}

	if !finiteRange(endpoints) {
		return errors.New("waterfall endpoints must have a finite range")
	}

	return nil
}

func validateECDFChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Datasets []struct {
			Data []float64 `json:"data"`
		} `json:"datasets"`
	}](raw)
	if err != nil {
		return err
	}

	var values []float64
	for _, dataset := range data.Datasets {
		values = append(values, dataset.Data...)
	}

	if len(values) > 2000 || !finiteRange(values) {
		return errors.New("ECDF requires at most 2000 samples with a finite overall range")
	}

	return nil
}

func validateNodeLinkChart(raw json.RawMessage) error {
	data, err := decodeChartData[struct {
		Nodes []chartNode `json:"nodes"`
		Links []chartLink `json:"links"`
	}](raw)
	if err != nil {
		return err
	}

	ids := make(map[string]bool, len(data.Nodes))
	for _, node := range data.Nodes {
		if ids[node.ID] {
			return errors.New("node-link node IDs must be unique")
		}

		ids[node.ID] = true
	}

	edges := make(map[[2]string]bool, len(data.Links))
	for _, link := range data.Links {
		if !ids[link.Source] || !ids[link.Target] {
			return errors.New("node-link endpoints must reference existing nodes")
		}

		edge := [2]string{link.Source, link.Target}
		if link.Source == link.Target || edges[edge] {
			return errors.New("node-link self-links and repeated directed edges are not supported")
		}

		edges[edge] = true
	}

	return nil
}

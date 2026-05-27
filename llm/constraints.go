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

package llm

import "fmt"

// ModelConstraints defines validation rules and limitations for a specific model.
// This shared type is used by all providers to ensure consistent validation behavior.
type ModelConstraints struct {
	// TemperatureRange defines the valid range for temperature parameter [min, max]
	TemperatureRange [2]float64

	// MaxInputTokens is the maximum context window size (input tokens)
	MaxInputTokens int

	// MaxOutputTokens is the maximum number of tokens the model can generate in a single response
	MaxOutputTokens int
}

// ValidateTemperature checks if a temperature value is valid for these constraints.
func (c *ModelConstraints) ValidateTemperature(temp float64) error {
	minTemp, maxTemp := c.TemperatureRange[0], c.TemperatureRange[1]
	if temp < minTemp || temp > maxTemp {
		return fmt.Errorf("temperature %f out of range [%f, %f]", temp, minTemp, maxTemp)
	}

	return nil
}

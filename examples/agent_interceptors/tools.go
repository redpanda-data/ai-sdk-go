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

package main

import (
	"context"
	"fmt"
	"math/rand"
	"time"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TemperatureSensorInput is the model-facing argument schema, inferred
// automatically by tool.Func.
type TemperatureSensorInput struct {
	Unit string `json:"unit,omitempty" jsonschema:"Temperature unit: 'celsius' or 'fahrenheit' (default: celsius)"`
}

type TemperatureSensorOutput struct {
	Temperature float64 `json:"temperature"`
	Unit        string  `json:"unit"`
	Timestamp   int64   `json:"timestamp"`
}

// NewTemperatureSensorTool simulates reading from a temperature sensor.
// The model cannot predict sensor readings, making this a genuine external
// dependency.
func NewTemperatureSensorTool() tool.Tool {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	return tool.Must(tool.Func(
		tool.Spec{
			Name:        "read_temperature_sensor",
			Description: "Reads the current temperature from the system's temperature sensor. Returns real-time sensor data.",
		},
		func(_ context.Context, in TemperatureSensorInput) (tool.Result[TemperatureSensorOutput], error) {
			unit := in.Unit
			if unit == "" {
				unit = "celsius"
			}

			// Simulate sensor reading with some randomness (18-24°C range with noise)
			baseTempC := 18.0 + rng.Float64()*6.0
			noise := (rng.Float64() - 0.5) * 0.5
			tempC := baseTempC + noise

			temp := tempC
			if unit == "fahrenheit" {
				temp = tempC*9.0/5.0 + 32.0
			}

			return tool.Done(TemperatureSensorOutput{
				Temperature: temp,
				Unit:        unit,
				Timestamp:   time.Now().Unix(),
			}), nil
		},
	))
}

// GetSecretValueInput is the model-facing argument schema.
type GetSecretValueInput struct {
	SecretName string `json:"secret_name" jsonschema:"Name of the secret to retrieve"`
}

type GetSecretValueOutput struct {
	SecretName  string `json:"secret_name"`
	SecretValue string `json:"secret_value"`
}

// NewGetSecretValueTool retrieves a secret value from memory. The model
// cannot know runtime secrets without accessing them, making this a genuine
// external dependency.
func NewGetSecretValueTool(secrets map[string]string) tool.Tool {
	return tool.Must(tool.Func(
		tool.Spec{
			Name:        "get_secret_value",
			Description: "Retrieves a secret value from the secure secrets store. Use this when you need to access API keys, passwords, or other sensitive configuration.",
		},
		func(_ context.Context, in GetSecretValueInput) (tool.Result[GetSecretValueOutput], error) {
			value, exists := secrets[in.SecretName]
			if !exists {
				return tool.Result[GetSecretValueOutput]{}, fmt.Errorf("secret %q not found", in.SecretName)
			}

			return tool.Done(GetSecretValueOutput{
				SecretName:  in.SecretName,
				SecretValue: value,
			}), nil
		},
	))
}

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
	"encoding/json"
	"fmt"
	"math/rand"
	"time"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// TemperatureSensorTool simulates reading from a temperature sensor.
// The model cannot predict sensor readings, making this a genuine external dependency.
type TemperatureSensorTool struct {
	rng *rand.Rand
}

func NewTemperatureSensorTool() *TemperatureSensorTool {
	return &TemperatureSensorTool{
		rng: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

type TemperatureSensorInput struct {
	Unit string `json:"unit,omitempty" jsonschema:"Temperature unit: 'celsius' or 'fahrenheit' (default: celsius)"`
}

type TemperatureSensorOutput struct {
	Temperature float64 `json:"temperature"`
	Unit        string  `json:"unit"`
	Timestamp   int64   `json:"timestamp"`
}

func (t *TemperatureSensorTool) Definition() llm.ToolDefinition {
	schema, err := jsonschema.For[TemperatureSensorInput](nil)
	if err != nil {
		panic(fmt.Sprintf("failed to generate schema: %v", err))
	}

	// Disable additional properties for strict validation
	falseSchema := &jsonschema.Schema{}
	schema.AdditionalProperties = falseSchema

	schemaBytes, _ := json.Marshal(schema)

	return llm.ToolDefinition{
		Name:        "read_temperature_sensor",
		Description: "Reads the current temperature from the system's temperature sensor. Returns real-time sensor data.",
		Parameters:  schemaBytes,
	}
}

// IsAsynchronous implements tool.Tool.
func (*TemperatureSensorTool) IsAsynchronous() bool { return false }

func (t *TemperatureSensorTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var input TemperatureSensorInput
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}

	unit := input.Unit
	if unit == "" {
		unit = "celsius"
	}

	// Simulate sensor reading with some randomness (18-24°C range with noise)
	baseTempC := 18.0 + t.rng.Float64()*6.0
	noise := (t.rng.Float64() - 0.5) * 0.5
	tempC := baseTempC + noise

	temp := tempC
	if unit == "fahrenheit" {
		temp = tempC*9.0/5.0 + 32.0
	}

	output := TemperatureSensorOutput{
		Temperature: temp,
		Unit:        unit,
		Timestamp:   time.Now().Unix(),
	}

	return json.Marshal(output)
}

// GetSecretValueTool retrieves a secret value from memory.
// The model cannot know runtime secrets without accessing them, making this a genuine external dependency.
type GetSecretValueTool struct {
	secrets map[string]string
}

func NewGetSecretValueTool(secrets map[string]string) *GetSecretValueTool {
	return &GetSecretValueTool{
		secrets: secrets,
	}
}

type GetSecretValueInput struct {
	SecretName string `json:"secret_name" jsonschema:"Name of the secret to retrieve"`
}

type GetSecretValueOutput struct {
	SecretName  string `json:"secret_name"`
	SecretValue string `json:"secret_value"`
}

func (t *GetSecretValueTool) Definition() llm.ToolDefinition {
	schema, err := jsonschema.For[GetSecretValueInput](nil)
	if err != nil {
		panic(fmt.Sprintf("failed to generate schema: %v", err))
	}

	// Disable additional properties for strict validation
	falseSchema := &jsonschema.Schema{}
	schema.AdditionalProperties = falseSchema

	schemaBytes, _ := json.Marshal(schema)

	return llm.ToolDefinition{
		Name:        "get_secret_value",
		Description: "Retrieves a secret value from the secure secrets store. Use this when you need to access API keys, passwords, or other sensitive configuration.",
		Parameters:  schemaBytes,
	}
}

// IsAsynchronous implements tool.Tool.
func (*GetSecretValueTool) IsAsynchronous() bool { return false }

func (t *GetSecretValueTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	var input GetSecretValueInput
	if err := json.Unmarshal(args, &input); err != nil {
		return nil, fmt.Errorf("invalid input: %w", err)
	}

	value, exists := t.secrets[input.SecretName]
	if !exists {
		return nil, fmt.Errorf("secret %q not found", input.SecretName)
	}

	output := GetSecretValueOutput{
		SecretName:  input.SecretName,
		SecretValue: value,
	}

	return json.Marshal(output)
}

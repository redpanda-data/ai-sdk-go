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

package mcp

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
)

type registryParameterBinding struct {
	path   []string
	header string
}

type registryParameterProperty struct {
	Type       json.RawMessage            `json:"type"`
	Header     json.RawMessage            `json:"x-mcp-header"` //nolint:tagliatelle // MCP protocol annotation.
	Properties map[string]json.RawMessage `json:"properties"`
}

// The SDK v1.8.0 SEP-2243 helpers and tool-context key are private. Capture only
// their parameter bindings; CallTool's live discovery cache cannot govern a snapshot.
func registryParameterBindings(schema json.RawMessage) ([]registryParameterBinding, error) {
	var root registryParameterProperty
	if err := json.Unmarshal(schema, &root); err != nil {
		return nil, fmt.Errorf("mcp: parameter header schema: %w", err)
	}

	var bindings []registryParameterBinding
	seen := make(map[string]bool)

	var collect func(map[string]json.RawMessage, []string) error

	collect = func(properties map[string]json.RawMessage, prefix []string) error {
		for name, raw := range properties {
			if string(raw) == "true" || string(raw) == "false" {
				continue
			}

			var property registryParameterProperty
			if err := json.Unmarshal(raw, &property); err != nil {
				return fmt.Errorf("mcp: parameter header property %q: %w", name, err)
			}

			path := append(append([]string(nil), prefix...), name)

			if property.Header != nil {
				var kind, header string
				if err := json.Unmarshal(property.Type, &kind); err != nil || (kind != "string" && kind != "integer" && kind != "boolean") {
					return fmt.Errorf("mcp: unsupported parameter header type at %q", strings.Join(path, "."))
				}

				if err := json.Unmarshal(property.Header, &header); err != nil || !validHeaderName(header) || seen[strings.ToLower(header)] {
					return fmt.Errorf("mcp: invalid or duplicate parameter header at %q", strings.Join(path, "."))
				}

				seen[strings.ToLower(header)] = true
				bindings = append(bindings, registryParameterBinding{path: path, header: "Mcp-Param-" + header})
			}

			if err := collect(property.Properties, path); err != nil {
				return err
			}
		}

		return nil
	}
	if err := collect(root.Properties, nil); err != nil {
		return nil, err
	}

	return bindings, nil
}

type registryParameterHeadersKey struct{}

func registryParameterHeaders(bindings []registryParameterBinding, arguments map[string]any) (http.Header, error) {
	headers := make(http.Header, len(bindings))

	for _, binding := range bindings {
		var value any = arguments
		for _, name := range binding.path {
			object, _ := value.(map[string]any)
			value = object[name]
		}

		if value == nil {
			continue
		}

		encoded, err := registryEncodeParameterHeader(value)
		if err != nil {
			return nil, fmt.Errorf("mcp: parameter header at %q: %w", strings.Join(binding.path, "."), err)
		}

		headers.Set(binding.header, encoded)
	}

	return headers, nil
}

func registryEncodeParameterHeader(value any) (string, error) {
	var text string

	switch value := value.(type) {
	case string:
		text = value
	case bool:
		text = strconv.FormatBool(value)
	case float64:
		if value != math.Trunc(value) || value < -(1<<53-1) || value > 1<<53-1 {
			return "", errors.New("expected a safe integer")
		}

		text = strconv.FormatInt(int64(value), 10)
	default:
		return "", errors.New("expected a string, boolean or safe integer")
	}

	encode := strings.Trim(text, " \t") != text || (strings.HasPrefix(text, "=?base64?") && strings.HasSuffix(text, "?="))
	for _, c := range text {
		if c < 0x20 || c > 0x7e {
			encode = true
			break
		}
	}

	if encode {
		return "=?base64?" + base64.StdEncoding.EncodeToString([]byte(text)) + "?=", nil
	}

	return text, nil
}

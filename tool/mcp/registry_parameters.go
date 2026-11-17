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
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
)

type registryParameterBinding struct {
	path   []string
	header string
}

// registryParameterBindings captures the SEP-2243 bindings of an SDK-decoded input
// schema: the SDK v1.8.0 helpers and tool-context key are private, and CallTool's
// live discovery cache cannot govern a snapshot. Like the legacy client, it leaves
// the rest of the schema to the model provider. A subtree of another shape binds
// nothing, and neither does an annotation the SDK would exclude from tools/list
// (non-primitive type, invalid or duplicate header name).
func registryParameterBindings(schema any) []registryParameterBinding {
	bindings := registryCollectParameterBindings(schema, nil, nil)

	counts := make(map[string]int, len(bindings))
	for _, binding := range bindings {
		counts[strings.ToLower(binding.header)]++
	}

	bindings = slices.DeleteFunc(bindings, func(binding registryParameterBinding) bool {
		return counts[strings.ToLower(binding.header)] > 1
	})

	for i := range bindings {
		bindings[i].header = "Mcp-Param-" + bindings[i].header
	}

	return bindings
}

func registryCollectParameterBindings(schema any, path []string, bindings []registryParameterBinding) []registryParameterBinding {
	object, _ := schema.(map[string]any)
	properties, _ := object["properties"].(map[string]any)

	for name, property := range properties {
		path := append(path, name)

		if header, ok := registryParameterHeader(property); ok {
			bindings = append(bindings, registryParameterBinding{path: slices.Clone(path), header: header})
		}

		bindings = registryCollectParameterBindings(property, path, bindings)
	}

	return bindings
}

func registryParameterHeader(property any) (string, bool) {
	object, _ := property.(map[string]any)

	header, ok := object["x-mcp-header"].(string)
	if !ok || !validHeaderName(header) {
		return "", false
	}

	switch object["type"] {
	case "string", "integer", "boolean":
		return header, true
	default:
		return "", false
	}
}

type registryParameterHeadersKey struct{}

// registryParameterHeaders mirrors the SDK: an absent or null argument, or one
// without a SEP-2243 encoding, sends no header and stays in the arguments.
func registryParameterHeaders(bindings []registryParameterBinding, arguments map[string]any) http.Header {
	headers := make(http.Header, len(bindings))

	for _, binding := range bindings {
		var value any = arguments
		for _, name := range binding.path {
			object, _ := value.(map[string]any)
			value = object[name]
		}

		if encoded, ok := registryEncodeParameterHeader(value); ok {
			headers.Set(binding.header, encoded)
		}
	}

	return headers
}

func registryEncodeParameterHeader(value any) (string, bool) {
	var text string

	switch value := value.(type) {
	case string:
		text = value
	case bool:
		text = strconv.FormatBool(value)
	case float64:
		if value != math.Trunc(value) || value < -(1<<53-1) || value > 1<<53-1 {
			return "", false
		}

		text = strconv.FormatInt(int64(value), 10)
	default:
		return "", false
	}

	encode := strings.Trim(text, " \t") != text || (strings.HasPrefix(text, "=?base64?") && strings.HasSuffix(text, "?="))
	for _, c := range text {
		if c < 0x20 || c > 0x7e {
			encode = true
			break
		}
	}

	if encode {
		return "=?base64?" + base64.StdEncoding.EncodeToString([]byte(text)) + "?=", true
	}

	return text, true
}

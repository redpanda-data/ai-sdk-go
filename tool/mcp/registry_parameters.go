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
	"strconv"
	"strings"
)

// registryParameterNode is a property on the path to at least one parameter
// header. Sharing path prefixes keeps a capture linear in the schema's size.
type registryParameterNode struct {
	header   string // with the Mcp-Param- prefix; empty if this property binds none
	children map[string]*registryParameterNode
}

// registryParameterTrie captures the SEP-2243 bindings of an SDK-decoded input
// schema: the SDK v1.8.0 helpers and tool-context key are private, and CallTool's
// live discovery cache cannot govern a snapshot. Like the legacy client, it leaves
// the rest of the schema to the model provider. A subtree of another shape binds
// nothing, and neither does an annotation the SDK would exclude from tools/list
// (non-primitive type, invalid or duplicate header name). It returns nil when
// nothing binds.
func registryParameterTrie(schema any) *registryParameterNode {
	counts := make(map[string]int)

	return registryPruneParameterNode(registryCollectParameterNode(schema, counts), counts)
}

func registryCollectParameterNode(schema any, counts map[string]int) *registryParameterNode {
	object, _ := schema.(map[string]any)
	properties, _ := object["properties"].(map[string]any)

	var node *registryParameterNode

	for name, property := range properties {
		child := registryCollectParameterNode(property, counts)

		if header, ok := registryParameterHeader(property); ok {
			if child == nil {
				child = new(registryParameterNode)
			}

			child.header = header
			counts[strings.ToLower(header)]++
		}

		if child == nil {
			continue
		}

		if node == nil {
			node = &registryParameterNode{children: make(map[string]*registryParameterNode)}
		}

		node.children[name] = child
	}

	return node
}

// registryPruneParameterNode drops every header bound more than once, ignoring
// case, and the subtrees left without a header.
func registryPruneParameterNode(node *registryParameterNode, counts map[string]int) *registryParameterNode {
	if node == nil {
		return nil
	}

	for name, child := range node.children {
		if registryPruneParameterNode(child, counts) == nil {
			delete(node.children, name)
		}
	}

	switch {
	case node.header == "":
	case counts[strings.ToLower(node.header)] > 1:
		node.header = ""
	default:
		node.header = "Mcp-Param-" + node.header
	}

	if node.header == "" && len(node.children) == 0 {
		return nil
	}

	return node
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
func registryParameterHeaders(trie *registryParameterNode, arguments map[string]any) http.Header {
	headers := make(http.Header)
	registryCollectParameterHeaders(trie, arguments, headers)

	return headers
}

// registryCollectParameterHeaders descends only where the trie and an argument
// object meet, iterating the smaller of the two at each level.
func registryCollectParameterHeaders(node *registryParameterNode, arguments map[string]any, headers http.Header) {
	if node == nil {
		return
	}

	visit := func(child *registryParameterNode, value any) {
		if child.header != "" {
			if encoded, ok := registryEncodeParameterHeader(value); ok {
				headers.Set(child.header, encoded)
			}
		}

		if object, ok := value.(map[string]any); ok {
			registryCollectParameterHeaders(child, object, headers)
		}
	}

	if len(arguments) < len(node.children) {
		for name, value := range arguments {
			if child, ok := node.children[name]; ok {
				visit(child, value)
			}
		}

		return
	}

	for name, child := range node.children {
		if value, ok := arguments[name]; ok {
			visit(child, value)
		}
	}
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

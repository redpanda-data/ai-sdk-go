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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
)

// registryMaxSchemaDepth bounds the JSON nesting of a discovered input schema.
// The schema library and the parameter-header parser decode every level's
// subtree again, so their cost grows quadratically with depth.
const registryMaxSchemaDepth = 64

var errRegistrySchemaExternalReference = errors.New("external schema references are not supported")

var registrySchemaTypeNames = []string{"array", "boolean", "integer", "null", "number", "object", "string"}

var registrySchemaPointerPath = strings.NewReplacer("~", "~0", "/", "~1")

type registrySchemaField struct {
	keyword string
	named   bool // an object of named subschemas, otherwise a subschema or an array of them
}

// registrySchemaFields lists the keywords of every subschema-bearing field of the
// library's Schema, so the type check follows the library's own notion of
// subschemas and never reads const, default, examples or extension values.
var registrySchemaFields = func() []registrySchemaField {
	var fields []registrySchemaField

	for _, field := range reflect.VisibleFields(reflect.TypeFor[jsonschema.Schema]()) {
		if !field.IsExported() {
			continue
		}

		var entry registrySchemaField

		switch field.Type {
		case reflect.TypeFor[map[string]*jsonschema.Schema]():
			entry.named = true
		case reflect.TypeFor[*jsonschema.Schema](), reflect.TypeFor[[]*jsonschema.Schema]():
		default:
			continue
		}

		switch field.Name {
		case "Items", "ItemsArray":
			entry.keyword = "items"
		case "DependencySchemas":
			entry.keyword = "dependencies"
		default:
			entry.keyword, _, _ = strings.Cut(field.Tag.Get("json"), ",")
		}

		if !slices.Contains(fields, entry) {
			fields = append(fields, entry)
		}
	}

	return fields
}()

// registryInputSchema validates a discovered input schema before it can reach a
// model and captures its parameter-header bindings.
func registryInputSchema(schema json.RawMessage) ([]registryParameterBinding, error) {
	if !bytes.HasPrefix(bytes.TrimSpace(schema), []byte("{")) {
		return nil, errors.New("mcp: input schema: must be a JSON object")
	}

	if err := registrySchemaDepth(schema); err != nil {
		return nil, fmt.Errorf("mcp: input schema: %w", err)
	}

	bindings, err := registryParameterBindings(schema)
	if err != nil {
		return nil, err
	}

	if err := registryValidateSchema(schema); err != nil {
		return nil, fmt.Errorf("mcp: input schema: %w", err)
	}

	return bindings, nil
}

// registrySchemaDepth only counts nesting; the decoders that follow reject
// invalid JSON before they recurse.
func registrySchemaDepth(schema json.RawMessage) error {
	depth, inString, escaped := 0, false, false

	for _, c := range schema {
		switch {
		case escaped:
			escaped = false
		case inString:
			escaped = c == '\\'
			inString = c != '"'
		case c == '"':
			inString = true
		case c == '{' || c == '[':
			if depth++; depth > registryMaxSchemaDepth {
				return fmt.Errorf("exceeds %d nested JSON levels", registryMaxSchemaDepth)
			}
		case c == '}' || c == ']':
			depth--
		}
	}

	return nil
}

func registryValidateSchema(raw json.RawMessage) error {
	var schema jsonschema.Schema
	if err := json.Unmarshal(raw, &schema); err != nil {
		return err
	}

	// Resolve checks references, anchors and patterns. Discovery never loads a
	// schema a server names: local references resolve, external ones fail.
	if _, err := schema.Resolve(&jsonschema.ResolveOptions{Loader: func(*url.URL) (*jsonschema.Schema, error) {
		return nil, errRegistrySchemaExternalReference
	}}); err != nil {
		return err
	}

	// The library neither checks type names nor keeps an explicit "" apart from
	// an absent type, so the check reads the raw keywords.
	var tree any
	if err := json.Unmarshal(raw, &tree); err != nil {
		return err
	}

	if err := registryCheckSchemaTypes(tree); err != nil {
		return err
	}

	return nil
}

// registrySchemaTypeError collects its JSON Pointer while the walk unwinds, so a
// valid schema builds no paths.
type registrySchemaTypeError struct {
	reason   string
	segments []string // innermost first
}

func (e *registrySchemaTypeError) Error() string {
	var pointer strings.Builder
	for _, segment := range slices.Backward(e.segments) {
		pointer.WriteString("/" + segment)
	}

	return fmt.Sprintf("%s at %q", e.reason, pointer.String())
}

// registryCheckSchemaTypes walks a schema decoded into any. The typed decode has
// already checked each keyword's shape: type is absent, a string or an array of
// strings, and subschemas are objects or booleans.
func registryCheckSchemaTypes(node any) *registrySchemaTypeError {
	schema, ok := node.(map[string]any)
	if !ok {
		return nil // a boolean schema or a dependencies property list
	}

	switch value := schema["type"].(type) {
	case string:
		if err := registryCheckSchemaTypeName(value); err != nil {
			return err
		}
	case []any:
		if len(value) == 0 {
			return &registrySchemaTypeError{reason: "empty type array"}
		}

		for i, member := range value {
			name, _ := member.(string)
			if err := registryCheckSchemaTypeName(name); err != nil {
				return err
			}

			if slices.Contains(value[:i], member) {
				return &registrySchemaTypeError{reason: fmt.Sprintf("duplicate type %q", name)}
			}
		}
	}

	for _, field := range registrySchemaFields {
		var err *registrySchemaTypeError

		switch child := schema[field.keyword].(type) {
		case map[string]any:
			if !field.named {
				err = registryCheckSchemaTypes(child)
				break
			}

			for _, key := range slices.Sorted(maps.Keys(child)) {
				if err = registryCheckSchemaTypes(child[key]); err != nil {
					err.segments = append(err.segments, registrySchemaPointerPath.Replace(key))
					break
				}
			}
		case []any:
			for i, element := range child {
				if err = registryCheckSchemaTypes(element); err != nil {
					err.segments = append(err.segments, strconv.Itoa(i))
					break
				}
			}
		}

		if err != nil {
			err.segments = append(err.segments, field.keyword)

			return err
		}
	}

	return nil
}

func registryCheckSchemaTypeName(name string) *registrySchemaTypeError {
	switch {
	case name == "":
		return &registrySchemaTypeError{reason: "empty type name"}
	case !slices.Contains(registrySchemaTypeNames, name):
		return &registrySchemaTypeError{reason: fmt.Sprintf("unknown type %q", name)}
	}

	return nil
}

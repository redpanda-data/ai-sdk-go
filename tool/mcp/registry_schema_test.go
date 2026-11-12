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
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func registrySchemaNested(levels int) string {
	return strings.Repeat(`{"items":`, levels-1) + `{}` + strings.Repeat(`}`, levels-1)
}

func TestRegistryInputSchemaAcceptsValidSchemas(t *testing.T) {
	t.Parallel()

	external := newRegistrySchemaExternalPeer(t)
	schemas := map[string]string{
		"empty object":     `{}`,
		"root union":       `{"type":["object","null"]}`,
		"boolean children": `{"type":"object","properties":{"a":true,"b":false},"additionalProperties":false,"items":true,"not":false}`,
		"depth limit":      registrySchemaNested(registryMaxSchemaDepth),
		"quoted nesting":   `{"description":"` + strings.Repeat(`{[\"\\`, 100) + `","enum":["]}\"","{"]}`,
		"escaped keys":     `{"type":"object","properties":{"a/b~c":{"type":"string"}},"$defs":{"x/y":{"type":"integer"}},"required":["a/b~c"]}`,
	}

	for name, schema := range registryValidComplexSchemas {
		schemas[name] = external.expand(schema)
	}

	for name, schema := range schemas {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			_, err := registryInputSchema(json.RawMessage(schema))
			require.NoError(t, err)
		})
	}

	bindings, err := registryInputSchema(json.RawMessage(registryValidComplexSchemas["annotations"]))
	require.NoError(t, err)
	assert.Equal(t, []registryParameterBinding{{path: []string{"route"}, header: "Mcp-Param-Tenant"}}, bindings)
}

func TestRegistryInputSchemaRejectsMalformedSchemas(t *testing.T) {
	t.Parallel()

	external := newRegistrySchemaExternalPeer(t)
	unmarshalType := func(field string) func(*testing.T, error) {
		return func(t *testing.T, err error) {
			t.Helper()

			var typeErr *json.UnmarshalTypeError
			require.ErrorAs(t, err, &typeErr)
			assert.Contains(t, typeErr.Field, field)
		}
	}
	externalReference := func(t *testing.T, err error) {
		t.Helper()
		require.ErrorIs(t, err, errRegistrySchemaExternalReference)
	}

	for _, test := range []struct {
		name   string
		schema string
		want   string
		cause  func(*testing.T, error)
	}{
		{"null", `null`, "must be a JSON object", nil},
		{"boolean", `true`, "must be a JSON object", nil},
		{"array", `[]`, "must be a JSON object", nil},
		{"numeric root type", `{"type":42}`, `invalid value for "type"`, nil},
		{"numeric nested type", `{"type":"object","properties":{"q":{"items":{"type":42}}}}`, `invalid value for "type"`, nil},
		{"nonstring type member", `{"type":["string",42]}`, "cannot unmarshal", unmarshalType("")},
		{"unknown root type", `{"type":"strin"}`, `unknown type "strin" at ""`, nil},
		{"empty root type", `{"type":""}`, `empty type name at ""`, nil},
		{"empty nested type", `{"type":"object","properties":{"q":{"type":""}}}`, `empty type name at "/properties/q"`, nil},
		{"empty union member", `{"type":["string",""]}`, `empty type name at ""`, nil},
		{"empty type of property named type", `{"properties":{"type":{"type":""}}}`, `empty type name at "/properties/type"`, nil},
		{"empty referenced definition type", `{"properties":{"q":{"$ref":"#/$defs/q"}},"$defs":{"q":{"type":["null",""]}}}`, `empty type name at "/$defs/q"`, nil},
		{"empty draft-07 tuple type", `{"$schema":"http://json-schema.org/draft-07/schema#","items":[{"type":"string"},{"type":""}]}`, `empty type name at "/items/1"`, nil},
		{"empty dependency schema type", `{"dependencies":{"a":["b"],"c":{"type":""}}}`, `empty type name at "/dependencies/c"`, nil},
		{"unknown union member", `{"properties":{"q":{"type":["string","nul"]}}}`, `unknown type "nul" at "/properties/q"`, nil},
		{"empty type array", `{"properties":{"q":{"type":[]}}}`, `empty type array at "/properties/q"`, nil},
		{"duplicate type", `{"type":["string","string"]}`, `duplicate type "string" at ""`, nil},
		{"unknown escaped definition type", `{"$defs":{"a/b~c":{"type":"bogus"}}}`, `unknown type "bogus" at "/$defs/a~1b~0c"`, nil},
		{"string required", `{"type":"object","required":"q"}`, "required", unmarshalType("required")},
		{"numeric required member", `{"type":"object","required":[1]}`, "required", unmarshalType("required")},
		{"properties array", `{"type":"object","properties":[]}`, "properties", unmarshalType("properties")},
		{"nonobject property", `{"type":"object","properties":{"q":"string"}}`, `property "q"`, unmarshalType("")},
		{"nonschema subschema", `{"type":"object","anyOf":[1]}`, "anyOf", unmarshalType("anyOf")},
		{"fractional length", `{"properties":{"q":{"minLength":1.5}}}`, "not an integer value", nil},
		{"string enum", `{"properties":{"q":{"enum":"a"}}}`, "enum", unmarshalType("enum")},
		{"invalid pattern", `{"properties":{"q":{"pattern":"("}}}`, "pattern", nil},
		{"dangling pointer", `{"properties":{"q":{"$ref":"#/$defs/missing"}}}`, `"/$defs/missing"`, nil},
		{"dangling anchor", `{"properties":{"q":{"$ref":"#missing"}}}`, `no anchor "missing"`, nil},
		{"absolute external reference", `{"properties":{"q":{"$ref":"{external}/schema.json"}}}`, "{external}/schema.json", externalReference},
		{"external pointer reference", `{"properties":{"q":{"$ref":"{external}/schema.json#/$defs/q"}}}`, "{external}/schema.json", externalReference},
		{"file reference", `{"properties":{"q":{"$ref":"file:///etc/passwd"}}}`, "file:///etc/passwd", externalReference},
		{"relative reference", `{"properties":{"q":{"$ref":"other.json"}}}`, "other.json", externalReference},
		{"external dynamic reference", `{"properties":{"q":{"$dynamicRef":"{external}/meta.json#meta"}}}`, "{external}/meta.json", externalReference},
		{"header annotation", `{"properties":{"q":{"type":"number","x-mcp-header":"Tenant"}}}`, `unsupported parameter header type at "q"`, nil},
		{"depth limit", registrySchemaNested(registryMaxSchemaDepth + 1), "exceeds 64 nested JSON levels", nil},
		{"deep nesting", strings.Repeat(`{"type":"object","properties":{"a":`, 5000) + `true` + strings.Repeat(`}}`, 5000), "exceeds 64 nested JSON levels", nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			bindings, err := registryInputSchema(json.RawMessage(external.expand(test.schema)))
			require.ErrorContains(t, err, external.expand(test.want))
			assert.Nil(t, bindings)

			if test.cause != nil {
				test.cause(t, err)
			}
		})
	}
}

// Every subschema-bearing keyword must be type checked. The coverage assertion
// fails if the library adds a subschema field this table does not exercise.
func TestRegistryInputSchemaChecksEverySubschema(t *testing.T) {
	t.Parallel()

	placements := map[string]string{
		`{"properties":{"p":%s}}`:                      "/properties/p",
		`{"patternProperties":{"^p":%s}}`:              "/patternProperties/^p",
		`{"additionalProperties":%s}`:                  "/additionalProperties",
		`{"propertyNames":%s}`:                         "/propertyNames",
		`{"unevaluatedProperties":%s}`:                 "/unevaluatedProperties",
		`{"dependentSchemas":{"p":%s}}`:                "/dependentSchemas/p",
		`{"dependencies":{"p":%s}}`:                    "/dependencies/p",
		`{"items":%s}`:                                 "/items",
		`{"items":[true,%s]}`:                          "/items/1",
		`{"prefixItems":[%s]}`:                         "/prefixItems/0",
		`{"additionalItems":%s}`:                       "/additionalItems",
		`{"contains":%s}`:                              "/contains",
		`{"unevaluatedItems":%s}`:                      "/unevaluatedItems",
		`{"allOf":[%s]}`:                               "/allOf/0",
		`{"anyOf":[%s]}`:                               "/anyOf/0",
		`{"oneOf":[%s]}`:                               "/oneOf/0",
		`{"not":%s}`:                                   "/not",
		`{"if":%s}`:                                    "/if",
		`{"then":%s}`:                                  "/then",
		`{"else":%s}`:                                  "/else",
		`{"$defs":{"d":%s}}`:                           "/$defs/d",
		`{"definitions":{"d":%s}}`:                     "/definitions/d",
		`{"contentSchema":%s}`:                         "/contentSchema",
		`{"properties":{"p":{"items":%s}}}`:            "/properties/p/items",
		`{"allOf":[{"not":{"anyOf":[%s]}}]}`:           "/allOf/0/not/anyOf/0",
		`{"type":["string","null"],"anyOf":[true,%s]}`: "/anyOf/1",
		`{"properties":{"q":{"$ref":"#/$defs/d"}},"$defs":{"d":{"oneOf":[%s]}}}`:                   "/$defs/d/oneOf/0",
		`{"$schema":"http://json-schema.org/draft-07/schema#","definitions":{"d":{"items":[%s]}}}`: "/definitions/d/items/0",
		`{"dependencies":{"a":["b"],"p":{"properties":{"q":%s}}}}`:                                 "/dependencies/p/properties/q",
	}

	covered := make(map[string]bool)

	for template, pointer := range placements {
		covered[strings.Split(pointer, "/")[1]] = true

		t.Run(pointer, func(t *testing.T) {
			t.Parallel()

			for schema, want := range map[string]string{
				`{"type":"bogus"}`:         `unknown type "bogus"`,
				`{"type":""}`:              "empty type name",
				`{"type":["string",""]}`:   "empty type name",
				`{"type":["null","null"]}`: `duplicate type "null"`,
			} {
				_, err := registryInputSchema(json.RawMessage(fmt.Sprintf(template, schema)))
				require.ErrorContains(t, err, fmt.Sprintf(`%s at %q`, want, pointer), schema)
			}

			for _, schema := range []string{`{"type":"string"}`, `{"type":["string","null"],"const":{"type":""},"x-meta":{"type":""}}`, `true`} {
				_, err := registryInputSchema(json.RawMessage(fmt.Sprintf(template, schema)))
				require.NoError(t, err, schema)
			}
		})
	}

	for _, field := range registrySchemaFields {
		assert.True(t, covered[field.keyword], "subschema keyword %q has no placement", field.keyword)
	}
}

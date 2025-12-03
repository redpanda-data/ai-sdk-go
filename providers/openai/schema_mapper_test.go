package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTransformSchemaForOpenAI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		input          map[string]any
		expectedOutput map[string]any
		description    string
	}{
		{
			name: "simple object with optional fields",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "User name",
					},
					"age": map[string]any{
						"type":        "integer",
						"description": "User age (optional)",
					},
				},
				"required":             []any{"name"},
				"additionalProperties": false,
			},
			expectedOutput: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type":        "string",
						"description": "User name",
					},
					"age": map[string]any{
						"type":        []any{"integer", "null"}, // Optional field becomes nullable union
						"description": "User age (optional)",
					},
				},
				"required":             []any{"name", "age"}, // All properties now required
				"additionalProperties": false,
			},
			description: "Optional fields should be converted to union types with null and marked as required",
		},
		{
			name: "nested object with optional fields",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"user": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"name": map[string]any{
								"type": "string",
							},
							"profile": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"bio": map[string]any{
										"type": "string",
									},
									"avatar": map[string]any{
										"type": "string",
									},
								},
								"required": []any{"bio"}, // avatar is optional
							},
						},
						"required": []any{"name"}, // profile is optional
					},
				},
				"required": []any{"user"},
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"user": map[string]any{
						"type":                 "object",
						"additionalProperties": false, // Auto-added for nested objects
						"properties": map[string]any{
							"name": map[string]any{
								"type": "string",
							},
							"profile": map[string]any{
								"type":                 []any{"object", "null"}, // Optional nested object becomes nullable
								"additionalProperties": false,
								"properties": map[string]any{
									"bio": map[string]any{
										"type": "string",
									},
									"avatar": map[string]any{
										"type": []any{"string", "null"}, // Optional field becomes nullable
									},
								},
								"required": []any{"bio", "avatar"}, // All nested properties required
							},
						},
						"required": []any{"name", "profile"}, // All properties required
					},
				},
				"required": []any{"user"},
			},
			description: "Nested objects should have transformation applied recursively",
		},
		{
			name: "array with object items",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"items": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type": "object",
							"properties": map[string]any{
								"id": map[string]any{
									"type": "integer",
								},
								"name": map[string]any{
									"type": "string",
								},
							},
							"required": []any{"id"}, // name is optional
						},
					},
				},
				"required": []any{"items"},
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"items": map[string]any{
						"type": "array",
						"items": map[string]any{
							"type":                 "object",
							"additionalProperties": false, // Auto-added for array item objects
							"properties": map[string]any{
								"id": map[string]any{
									"type": "integer",
								},
								"name": map[string]any{
									"type": []any{"string", "null"}, // Optional array item field becomes nullable
								},
							},
							"required": []any{"id", "name"}, // All array item properties required
						},
					},
				},
				"required": []any{"items"},
			},
			description: "Array item schemas should also be transformed",
		},
		{
			name: "schema with existing union types",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"value": map[string]any{
						"type": []any{"string", "integer"}, // Already a union
					},
					"optional_value": map[string]any{
						"type": []any{"string", "integer"}, // Union but optional
					},
				},
				"required": []any{"value"},
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"value": map[string]any{
						"type": []any{"string", "integer"}, // Required field keeps existing union unchanged
					},
					"optional_value": map[string]any{
						"type": []any{"string", "integer", "null"}, // Optional field gets null added to union
					},
				},
				"required": []any{"value", "optional_value"}, // All properties required
			},
			description: "Existing union types should have null added if they don't already have it",
		},
		{
			name: "schema with nullable property",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type": "string",
					},
					"description": map[string]any{
						"type":     "string",
						"nullable": true, // Should be removed and converted to union
					},
				},
				"required": []any{"name"},
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"name": map[string]any{
						"type": "string",
					},
					"description": map[string]any{
						"type": []any{"string", "null"}, // "nullable": true becomes union type, nullable removed
					},
				},
				"required": []any{"name", "description"}, // All properties required
			},
			description: "nullable properties should be converted to union types and nullable removed",
		},
		{
			name: "schema with allOf combinator",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"combined": map[string]any{
						"allOf": []any{
							map[string]any{
								"type": "object",
								"properties": map[string]any{
									"base_field": map[string]any{
										"type": "string",
									},
								},
								"required": []any{"base_field"},
							},
							map[string]any{
								"type": "object",
								"properties": map[string]any{
									"extra_field": map[string]any{
										"type": "string",
									},
								},
								// extra_field is optional in this subschema
							},
						},
					},
				},
				"required": []any{"combined"},
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"combined": map[string]any{
						"allOf": []any{
							map[string]any{
								"type":                 "object",
								"additionalProperties": false, // Auto-added for combinator subschemas
								"properties": map[string]any{
									"base_field": map[string]any{
										"type": "string", // Required field unchanged
									},
								},
								"required": []any{"base_field"},
							},
							map[string]any{
								"type":                 "object",
								"additionalProperties": false, // Auto-added for combinator subschemas
								"properties": map[string]any{
									"extra_field": map[string]any{
										"type": []any{"string", "null"}, // Optional field becomes nullable
									},
								},
								"required": []any{"extra_field"}, // All properties required in subschema
							},
						},
					},
				},
				"required": []any{"combined"},
			},
			description: "allOf combinators should have transformation applied to each subschema",
		},
		{
			name: "no transformation needed - all fields already required",
			input: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name": map[string]any{
						"type": "string",
					},
					"age": map[string]any{
						"type": "integer",
					},
				},
				"required": []any{"name", "age"}, // All fields already required
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false, // Auto-added for structured outputs
				"properties": map[string]any{
					"name": map[string]any{
						"type": "string", // Already required, no transformation needed
					},
					"age": map[string]any{
						"type": "integer", // Already required, no transformation needed
					},
				},
				"required": []any{"name", "age"}, // No change needed
			},
			description: "Schemas where all fields are already required should be unchanged",
		},
		{
			name: "non-object schema should be unchanged",
			input: map[string]any{
				"type": "string",
				"enum": []any{"red", "green", "blue"},
			},
			expectedOutput: map[string]any{
				"type": "string",
				"enum": []any{"red", "green", "blue"}, // Non-object schemas pass through unchanged
			},
			description: "Non-object schemas should pass through unchanged",
		},
		{
			name: "empty object schema should get empty properties",
			input: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
			},
			expectedOutput: map[string]any{
				"type":                 "object",
				"additionalProperties": false,
				"properties":           map[string]any{}, // Empty properties added for OpenAI compatibility
			},
			description: "Empty object schemas (no parameters) should have empty properties field added for OpenAI API compatibility",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Make a deep copy of input to ensure we don't modify the original
			input := deepCopyMapForTest(t, tt.input)

			// Apply transformation
			transformSchemaForOpenAI(input)

			// Compare result using JSON marshaling to handle type conversion issues
			expectedJSON, err := json.Marshal(tt.expectedOutput)
			require.NoError(t, err)
			actualJSON, err := json.Marshal(input)
			require.NoError(t, err)

			if !assert.JSONEq(t, string(expectedJSON), string(actualJSON), tt.description) {
				// If JSONEq fails, show the detailed diff for debugging
				t.Logf("Expected: %+v", tt.expectedOutput)
				t.Logf("Actual: %+v", input)
			}
		})
	}
}

func TestAdaptSchemaForOpenAI(t *testing.T) {
	t.Parallel()

	mapper := NewSchemaMapper()

	t.Run("deep copy behavior", func(t *testing.T) {
		t.Parallel()

		original := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name": map[string]any{"type": "string"},
				"age":  map[string]any{"type": "integer"},
			},
			"required": []any{"name"},
		}

		// Make a copy to compare against
		originalCopy := deepCopyMapForTest(t, original)

		// Adapt schema
		adapted := mapper.AdaptSchemaForOpenAI(original)

		// Original should be unchanged
		assert.Equal(t, originalCopy, original, "Original schema should not be modified")

		// Adapted should be different
		assert.NotEqual(t, original, adapted, "Adapted schema should be different from original")

		// Adapted should have all fields required (use ElementsMatch for order independence)
		assert.ElementsMatch(t, []string{"name", "age"}, adapted["required"])

		// Age should be nullable in adapted version
		properties, ok := adapted["properties"].(map[string]any)
		require.True(t, ok, "properties should be a map[string]any")
		ageProperty, ok := properties["age"].(map[string]any)
		require.True(t, ok, "age property should be a map[string]any")
		assert.EqualValues(t, []any{"integer", "null"}, ageProperty["type"])
	})
}

func TestSchemaMapperWithCaching(t *testing.T) {
	t.Parallel()

	mapper := NewSchemaMapper()

	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name":  map[string]any{"type": "string"},
			"email": map[string]any{"type": "string"},
		},
		"required": []any{"name"},
	}

	t.Run("caching works correctly", func(t *testing.T) {
		t.Parallel()
		// First call - should transform and cache
		adapted1 := mapper.AdaptSchemaForOpenAI(schema)

		// Verify transformation happened (use ElementsMatch for order independence)
		assert.ElementsMatch(t, []string{"name", "email"}, adapted1["required"])
		properties1, ok := adapted1["properties"].(map[string]any)
		require.True(t, ok, "properties should be a map[string]any")
		emailProp1, ok := properties1["email"].(map[string]any)
		require.True(t, ok, "email property should be a map[string]any")
		assert.EqualValues(t, []any{"string", "null"}, emailProp1["type"])
		assert.Equal(t, false, adapted1["additionalProperties"])

		// Second call with same schema - should use cache
		adapted2 := mapper.AdaptSchemaForOpenAI(schema)
		assert.Equal(t, adapted1, adapted2, "Second call should return identical result")
		assert.Equal(t, false, adapted2["additionalProperties"])

		// Verify that cache is being used by checking that subsequent calls return identical results
		// (we can't easily check the internal cache key since it's a SHA256 hash)
		adapted3 := mapper.AdaptSchemaForOpenAI(schema)
		assert.Equal(t, adapted1, adapted3, "Third call should return identical result")
	})

	t.Run("different schemas get different cache entries", func(t *testing.T) {
		t.Parallel()

		schema1 := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"field1": map[string]any{"type": "string"},
			},
			"required": []any{},
		}

		schema2 := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"field2": map[string]any{"type": "integer"},
			},
			"required": []any{},
		}

		adapted1 := mapper.AdaptSchemaForOpenAI(schema1)
		adapted2 := mapper.AdaptSchemaForOpenAI(schema2)

		// Should be different results
		assert.NotEqual(t, adapted1, adapted2)

		// Verify they were transformed correctly
		properties1, ok := adapted1["properties"].(map[string]any)
		require.True(t, ok, "schema1 properties should be a map[string]any")
		field1, ok := properties1["field1"].(map[string]any)
		require.True(t, ok, "field1 should be a map[string]any")
		assert.EqualValues(t, []any{"string", "null"}, field1["type"])

		properties2, ok := adapted2["properties"].(map[string]any)
		require.True(t, ok, "schema2 properties should be a map[string]any")
		field2, ok := properties2["field2"].(map[string]any)
		require.True(t, ok, "field2 should be a map[string]any")
		assert.EqualValues(t, []any{"integer", "null"}, field2["type"])

		// Both should produce consistent results when called again (indicating caching)
		adapted1Again := mapper.AdaptSchemaForOpenAI(schema1)
		adapted2Again := mapper.AdaptSchemaForOpenAI(schema2)

		assert.Equal(t, adapted1, adapted1Again, "Schema1 should produce consistent results")
		assert.Equal(t, adapted2, adapted2Again, "Schema2 should produce consistent results")
	})
}

func TestDeepCopyMap(t *testing.T) {
	t.Parallel()

	t.Run("successful deep copy", func(t *testing.T) {
		t.Parallel()

		original := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"nested": map[string]any{
					"type":  "string",
					"items": []any{"a", "b", "c"},
				},
			},
			"numbers": []any{1, 2, 3},
		}

		copied, err := deepCopyMap(original)
		require.NoError(t, err)

		// Should be equal in content (use JSONEq to handle JSON type conversion)
		originalJSON, err := json.Marshal(original)
		require.NoError(t, err)
		copiedJSON, err := json.Marshal(copied)
		require.NoError(t, err)
		assert.JSONEq(t, string(originalJSON), string(copiedJSON))

		// But different objects (not same reference)
		// Modify copied to verify independence
		copied["type"] = "modified"
		assert.NotEqual(t, original["type"], copied["type"])

		// Modify nested structure
		copiedProps, ok := copied["properties"].(map[string]any)
		require.True(t, ok, "copied properties should be a map[string]any")
		copiedNested, ok := copiedProps["nested"].(map[string]any)
		require.True(t, ok, "copied nested should be a map[string]any")

		copiedNested["type"] = "modified_nested"

		originalProps, ok := original["properties"].(map[string]any)
		require.True(t, ok, "original properties should be a map[string]any")
		originalNested, ok := originalProps["nested"].(map[string]any)
		require.True(t, ok, "original nested should be a map[string]any")
		assert.NotEqual(t, originalNested["type"], copiedNested["type"])
	})

	t.Run("handles invalid JSON gracefully", func(t *testing.T) {
		t.Parallel()

		// This shouldn't normally happen with valid map[string]any,
		// but test error handling
		invalidMap := map[string]any{
			"channel": make(chan int), // channels can't be JSON marshaled
		}

		_, err := deepCopyMap(invalidMap)
		assert.Error(t, err, "Should return error for non-JSON-serializable data")
	})
}

func TestOptionalEnumBecomesNullable(t *testing.T) {
	t.Parallel()

	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"status": map[string]any{
				"enum": []any{"open", "closed"},
			},
		},
		"required": []any{}, // status optional
	}
	transformSchemaForOpenAI(in)
	props, ok := in["properties"].(map[string]any)
	require.True(t, ok, "properties should be a map[string]any")
	status, ok := props["status"].(map[string]any)
	require.True(t, ok, "status property should be a map[string]any")
	assert.ElementsMatch(t, []any{"open", "closed", nil}, status["enum"]) // null added to enum for optionality
	assert.ElementsMatch(t, []any{"status"}, in["required"])              // property becomes required
}

func TestTupleItemsAreTransformed(t *testing.T) {
	t.Parallel()

	in := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tuple": map[string]any{
				"type": "array",
				"items": []any{
					map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}, "required": []any{}},
					map[string]any{"type": "object", "properties": map[string]any{"b": map[string]any{"type": "integer"}}, "required": []any{"b"}},
				},
			},
		},
		"required": []any{"tuple"},
	}
	transformSchemaForOpenAI(in)
	props, ok := in["properties"].(map[string]any)
	require.True(t, ok, "properties should be a map[string]any")
	tupleProps, ok := props["tuple"].(map[string]any)
	require.True(t, ok, "tuple property should be a map[string]any")
	items, ok := tupleProps["items"].([]any)
	require.True(t, ok, "items should be a []any")

	// first tuple element's 'a' should be nullable + required
	obj0, ok := items[0].(map[string]any)
	require.True(t, ok, "first tuple item should be a map[string]any")
	props0, ok := obj0["properties"].(map[string]any)
	require.True(t, ok, "first tuple item properties should be a map[string]any")
	propA, ok := props0["a"].(map[string]any)
	require.True(t, ok, "property 'a' should be a map[string]any")
	assert.Equal(t, []any{"string", "null"}, propA["type"])
	assert.ElementsMatch(t, []any{"a"}, obj0["required"]) // Property becomes required

	// second tuple element's 'b' already required -> unchanged type
	obj1, ok := items[1].(map[string]any)
	require.True(t, ok, "second tuple item should be a map[string]any")
	props1, ok := obj1["properties"].(map[string]any)
	require.True(t, ok, "second tuple item properties should be a map[string]any")
	propB, ok := props1["b"].(map[string]any)
	require.True(t, ok, "property 'b' should be a map[string]any")
	assert.Equal(t, "integer", propB["type"])
}

// Helper function for tests to create deep copies.
func deepCopyMapForTest(t *testing.T, m map[string]any) map[string]any {
	t.Helper()

	copied, err := deepCopyMap(m)
	require.NoError(t, err)

	return copied
}

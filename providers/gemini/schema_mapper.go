package gemini

// SchemaMapper transforms standard JSON Schemas to Gemini-compatible schemas.
// Gemini uses standard JSON Schema format, so no transformation is needed.
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForGemini passes through the schema unchanged.
// Gemini accepts standard JSON Schema, so no transformation is needed.
func (*SchemaMapper) AdaptSchemaForGemini(schema map[string]any) map[string]any {
	return schema
}

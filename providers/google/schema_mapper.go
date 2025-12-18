package google

// SchemaMapper transforms standard JSON Schemas to Google-compatible schemas.
// Google Gemini API uses standard JSON Schema format, so no transformation is needed.
type SchemaMapper struct{}

// NewSchemaMapper creates a new SchemaMapper for schema transformations.
func NewSchemaMapper() *SchemaMapper { return &SchemaMapper{} }

// AdaptSchemaForGoogle passes through the schema unchanged.
// Google Gemini API accepts standard JSON Schema, so no transformation is needed.
func (*SchemaMapper) AdaptSchemaForGoogle(schema map[string]any) map[string]any {
	return schema
}

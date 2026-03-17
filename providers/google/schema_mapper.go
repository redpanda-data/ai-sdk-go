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

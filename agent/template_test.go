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

package agent

import (
	"testing"
)

func TestResolveTemplate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		template string
		vars     map[string]any
		want     string
	}{
		{
			name:     "simple replacement",
			template: "Hello {name}!",
			vars:     map[string]any{"name": "Alice"},
			want:     "Hello Alice!",
		},
		{
			name:     "multiple replacements",
			template: "{greeting}, {name}!",
			vars:     map[string]any{"greeting": "Hi", "name": "Bob"},
			want:     "Hi, Bob!",
		},
		{
			name:     "missing variable",
			template: "Hello {name}, welcome to {place}!",
			vars:     map[string]any{"name": "Alice"},
			want:     "Hello Alice, welcome to {place}!",
		},
		{
			name:     "JSON safety - ignores JSON braces",
			template: `Output JSON: {"name": "{name}", "age": 30}`,
			vars:     map[string]any{"name": "Alice"},
			want:     `Output JSON: {"name": "Alice", "age": 30}`,
		},
		{
			name:     "JSON safety - ignores complex JSON",
			template: `Format: {"user": {"id": 123}}`,
			vars:     map[string]any{"id": 456},
			want:     `Format: {"user": {"id": 123}}`, // No replacement because {id} is inside quotes/braces that don't match {key}
		},
		{
			name:     "underscore and numbers",
			template: "ID: {user_id_123}",
			vars:     map[string]any{"user_id_123": "abc-789"},
			want:     "ID: abc-789",
		},
		{
			name:     "different types",
			template: "Count: {count}, Active: {active}",
			vars:     map[string]any{"count": 42, "active": true},
			want:     "Count: 42, Active: true",
		},
		{
			name:     "nil vars",
			template: "Hello {name}",
			vars:     nil,
			want:     "Hello {name}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ResolveTemplate(tt.template, tt.vars)
			if got != tt.want {
				t.Errorf("ResolveTemplate() = %q, want %q", got, tt.want)
			}
		})
	}
}

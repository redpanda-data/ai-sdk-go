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
	"fmt"
	"regexp"
)

var (
	// placeholderRegex matches {key} patterns where key is alphanumeric, underscore, hyphen, or dot.
	// This specific pattern avoids collisions with JSON objects like {"key": "value"}.
	placeholderRegex = regexp.MustCompile(`\{([a-zA-Z0-9_\-\.]+)\}`)
)

// ResolveTemplate replaces {key} placeholders in a string with values from the provided map.
//
// Replacement Rules:
//   - If {key} matches and vars[key] exists and is a primitive (string, int, bool),
//     it is replaced with its string representation.
//   - String values are sanitized by stripping any newlines to prevent prompt injection.
//   - If vars[key] does not exist or is not a supported type, the placeholder {key} is left UNCHANGED.
//   - Placeholders that don't match the alphanumeric/dot/hyphen pattern are ignored.
//
// This behavior ensures that valid JSON and Markdown structures are preserved even if
// they contain braces, while allowing for safe and predictable variable injection.
func ResolveTemplate(template string, vars map[string]any) string {
	if vars == nil {
		return template
	}

	return placeholderRegex.ReplaceAllStringFunc(template, func(match string) string {
		// match is "{key}", we need "key" which is the first submatch
		// Since placeholderRegex matches the whole {key}, we can strip the braces.
		key := match[1 : len(match)-1]

		val, ok := vars[key]
		if !ok {
			return match
		}

		// Sanitize and format based on type
		switch v := val.(type) {
		case string:
			// Strip newlines to prevent prompt injection and maintain structure
			return regexp.MustCompile(`\r?\n`).ReplaceAllString(v, " ")
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return fmt.Sprint(v)
		case float32, float64:
			return fmt.Sprint(v)
		case bool:
			return fmt.Sprint(v)
		default:
			// Unsupported type, leave as is
			return match
		}
	})
}

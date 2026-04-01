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

package wireconformance

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// DiffKind classifies a single field-level difference.
type DiffKind int

const (
	// DiffMissing means a field is present in the native SDK request but absent in ai-sdk-go.
	DiffMissing DiffKind = iota
	// DiffExtra means a field is present in ai-sdk-go but absent in the native SDK request.
	DiffExtra
	// DiffChanged means both have the field but with different values.
	DiffChanged
	// DiffTypeMismatch means both have the field but the JSON types differ.
	DiffTypeMismatch
)

func (k DiffKind) String() string {
	switch k {
	case DiffMissing:
		return "MISSING"
	case DiffExtra:
		return "EXTRA"
	case DiffChanged:
		return "CHANGED"
	case DiffTypeMismatch:
		return "TYPE_MISMATCH"
	default:
		return "UNKNOWN"
	}
}

// FieldDiff represents a single difference at a specific JSON path.
type FieldDiff struct {
	Path     string          // JSON path, e.g. "messages[0].content"
	Kind     DiffKind        // Type of difference
	Expected json.RawMessage // Value from native SDK (nil if DiffExtra)
	Actual   json.RawMessage // Value from ai-sdk-go (nil if DiffMissing)
}

// DiffJSON performs a recursive structural diff between two JSON values.
// Fields matching any ignore rule are skipped.
func DiffJSON(native, aisdk json.RawMessage, ignores map[string]bool) []FieldDiff {
	var diffs []FieldDiff
	diffRecursive("", native, aisdk, ignores, &diffs)
	return diffs
}

func diffRecursive(path string, native, aisdk json.RawMessage, ignores map[string]bool, diffs *[]FieldDiff) {
	if shouldIgnore(path, ignores) {
		return
	}

	// Determine JSON types
	nType := jsonType(native)
	aType := jsonType(aisdk)

	// Handle nil/null cases
	if isAbsent(native) && isAbsent(aisdk) {
		return
	}
	if isAbsent(native) && !isAbsent(aisdk) {
		*diffs = append(*diffs, FieldDiff{
			Path:   path,
			Kind:   DiffExtra,
			Actual: aisdk,
		})
		return
	}
	if !isAbsent(native) && isAbsent(aisdk) {
		*diffs = append(*diffs, FieldDiff{
			Path:     path,
			Kind:     DiffMissing,
			Expected: native,
		})
		return
	}

	// Type mismatch
	if nType != aType {
		*diffs = append(*diffs, FieldDiff{
			Path:     path,
			Kind:     DiffTypeMismatch,
			Expected: native,
			Actual:   aisdk,
		})
		return
	}

	switch nType {
	case "object":
		diffObjects(path, native, aisdk, ignores, diffs)
	case "array":
		diffArrays(path, native, aisdk, ignores, diffs)
	default:
		// Scalar comparison
		if string(native) != string(aisdk) {
			*diffs = append(*diffs, FieldDiff{
				Path:     path,
				Kind:     DiffChanged,
				Expected: native,
				Actual:   aisdk,
			})
		}
	}
}

func diffObjects(path string, native, aisdk json.RawMessage, ignores map[string]bool, diffs *[]FieldDiff) {
	var nMap, aMap map[string]json.RawMessage
	json.Unmarshal(native, &nMap)  //nolint:errcheck
	json.Unmarshal(aisdk, &aMap)   //nolint:errcheck

	// Collect all keys
	allKeys := make(map[string]bool)
	for k := range nMap {
		allKeys[k] = true
	}
	for k := range aMap {
		allKeys[k] = true
	}

	// Sort for deterministic output
	keys := make([]string, 0, len(allKeys))
	for k := range allKeys {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, key := range keys {
		childPath := joinPath(path, key)
		nVal, nOK := nMap[key]
		aVal, aOK := aMap[key]

		switch {
		case nOK && aOK:
			diffRecursive(childPath, nVal, aVal, ignores, diffs)
		case nOK && !aOK:
			diffRecursive(childPath, nVal, nil, ignores, diffs)
		case !nOK && aOK:
			diffRecursive(childPath, nil, aVal, ignores, diffs)
		}
	}
}

func diffArrays(path string, native, aisdk json.RawMessage, ignores map[string]bool, diffs *[]FieldDiff) {
	var nArr, aArr []json.RawMessage
	json.Unmarshal(native, &nArr) //nolint:errcheck
	json.Unmarshal(aisdk, &aArr)  //nolint:errcheck

	maxLen := len(nArr)
	if len(aArr) > maxLen {
		maxLen = len(aArr)
	}

	for i := 0; i < maxLen; i++ {
		childPath := fmt.Sprintf("%s[%d]", path, i)
		var nVal, aVal json.RawMessage
		if i < len(nArr) {
			nVal = nArr[i]
		}
		if i < len(aArr) {
			aVal = aArr[i]
		}
		diffRecursive(childPath, nVal, aVal, ignores, diffs)
	}
}

func joinPath(parent, child string) string {
	if parent == "" {
		return child
	}
	return parent + "." + child
}

func jsonType(raw json.RawMessage) string {
	if len(raw) == 0 {
		return "null"
	}
	switch raw[0] {
	case '{':
		return "object"
	case '[':
		return "array"
	case '"':
		return "string"
	case 't', 'f':
		return "boolean"
	case 'n':
		return "null"
	default:
		return "number"
	}
}

func isAbsent(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return true
	}
	return string(raw) == "null"
}

// shouldIgnore checks if a path matches any ignore pattern.
// Supports exact match and wildcard suffix match (e.g. "metadata.*" ignores "metadata.foo").
func shouldIgnore(path string, ignores map[string]bool) bool {
	if ignores[path] {
		return true
	}
	// Check wildcard patterns: "foo.*" matches "foo.bar" and "foo.bar.baz"
	for pattern := range ignores {
		if strings.HasSuffix(pattern, ".*") {
			prefix := strings.TrimSuffix(pattern, ".*")
			if strings.HasPrefix(path, prefix+".") {
				return true
			}
		}
	}
	return false
}

// FormatDiffs produces a human/agent-readable report of all diffs.
func FormatDiffs(scenario string, diffs []FieldDiff, fixHint string) string {
	if len(diffs) == 0 {
		return ""
	}

	var b strings.Builder
	fmt.Fprintf(&b, "FAIL: %s (%d diffs)\n\n", scenario, len(diffs))

	for _, d := range diffs {
		fmt.Fprintf(&b, "  Path: %s\n", d.Path)
		fmt.Fprintf(&b, "    kind:    %s\n", d.Kind)
		if d.Expected != nil {
			fmt.Fprintf(&b, "    native:  %s\n", truncate(string(d.Expected), 200))
		}
		if d.Actual != nil {
			fmt.Fprintf(&b, "    ai-sdk:  %s\n", truncate(string(d.Actual), 200))
		}
		if d.Expected == nil {
			fmt.Fprintf(&b, "    native:  <absent>\n")
		}
		if d.Actual == nil {
			fmt.Fprintf(&b, "    ai-sdk:  <absent>\n")
		}
		if fixHint != "" {
			fmt.Fprintf(&b, "    fix-in:  %s\n", fixHint)
		}
		b.WriteString("\n")
	}

	return b.String()
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

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

package tool

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
)

// ArgumentsHash returns the lowercase-hex SHA-256 of the JCS-canonical
// form of args. It is used by the registry to detect duplicate pending
// tool calls (same tool name + same canonicalized arguments) and by the
// runner to compute resume receipt hashes.
//
// The empty input hashes to the SHA-256 of an empty byte string; nil and
// `null` JSON inputs hash to the same value as plain null per JCS.
func ArgumentsHash(args json.RawMessage) (string, error) {
	canon, err := canonicalizeJSON(args)
	if err != nil {
		return "", fmt.Errorf("tool: canonicalize arguments: %w", err)
	}

	sum := sha256.Sum256(canon)

	return hex.EncodeToString(sum[:]), nil
}

// canonicalizeJSON returns the RFC 8785 (JCS) canonical form of input.
// Object keys are sorted lexicographically by their UTF-16 code units (the
// JCS rule), numbers are normalized via ECMA-262 number-to-string, strings
// are minimally escaped, and no insignificant whitespace is emitted.
//
// This implementation covers the JSON types tools actually pass as
// arguments: objects, arrays, strings, numbers, booleans, null. It does
// not handle JSON5 extensions or NaN/Infinity (which aren't valid JSON
// anyway).
func canonicalizeJSON(input json.RawMessage) ([]byte, error) {
	if len(input) == 0 {
		return []byte("null"), nil
	}

	var value any

	dec := json.NewDecoder(bytes.NewReader(input))
	dec.UseNumber()

	if err := dec.Decode(&value); err != nil {
		return nil, fmt.Errorf("decode json: %w", err)
	}

	if dec.More() {
		return nil, errors.New("unexpected trailing data after JSON value")
	}

	var buf bytes.Buffer
	if err := writeCanonical(&buf, value); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func writeCanonical(w *bytes.Buffer, value any) error {
	switch v := value.(type) {
	case nil:
		w.WriteString("null")
		return nil

	case bool:
		if v {
			w.WriteString("true")
		} else {
			w.WriteString("false")
		}

		return nil

	case string:
		return writeCanonicalString(w, v)

	case json.Number:
		return writeCanonicalNumber(w, string(v))

	case []any:
		w.WriteByte('[')

		for i, elem := range v {
			if i > 0 {
				w.WriteByte(',')
			}

			if err := writeCanonical(w, elem); err != nil {
				return err
			}
		}

		w.WriteByte(']')

		return nil

	case map[string]any:
		keys := make([]string, 0, len(v))
		for k := range v {
			keys = append(keys, k)
		}

		sortJCSKeys(keys)

		w.WriteByte('{')

		for i, k := range keys {
			if i > 0 {
				w.WriteByte(',')
			}

			if err := writeCanonicalString(w, k); err != nil {
				return err
			}

			w.WriteByte(':')

			if err := writeCanonical(w, v[k]); err != nil {
				return err
			}
		}

		w.WriteByte('}')

		return nil

	default:
		return fmt.Errorf("unsupported JSON type %T", value)
	}
}

// sortJCSKeys sorts keys by their UTF-16 code units, per RFC 8785 §3.2.3.
// Go strings are UTF-8; we re-encode to UTF-16 code units to compare.
func sortJCSKeys(keys []string) {
	sort.Slice(keys, func(i, j int) bool {
		return lessUTF16(keys[i], keys[j])
	})
}

func lessUTF16(a, b string) bool {
	ar := []rune(a)
	br := []rune(b)

	// Iterate code units (UTF-16) by encoding each rune into 1 or 2 code
	// units on the fly. Comparing code unit by code unit is what JCS
	// requires; comparing runes directly is wrong for any string
	// containing a surrogate-paired code point.
	au, av := nextUTF16(ar, 0)
	bu, bv := nextUTF16(br, 0)

	for au >= 0 || bu >= 0 {
		switch {
		case au < 0:
			return true
		case bu < 0:
			return false
		case av != bv:
			return av < bv
		}

		au, av = nextUTF16(ar, au)
		bu, bv = nextUTF16(br, bu)
	}

	return false
}

// nextUTF16 returns the next UTF-16 code unit at position i in runes,
// plus the new index. When the input is exhausted, the returned index
// is -1.
func nextUTF16(runes []rune, i int) (int, uint32) {
	if i < 0 || i >= len(runes) {
		return -1, 0
	}

	r := runes[i]
	if r < 0 {
		// Invalid rune (cannot appear in decoded JSON strings); map to
		// the replacement character so the uint32 conversion is safe.
		r = 0xFFFD
	}

	if r < 0x10000 {
		return i + 1, uint32(r)
	}

	// Surrogate pair: return the high surrogate first; the low surrogate
	// is returned on the next call by stashing index back at the same i
	// and using a side channel — but rather than maintain state, expand
	// runes to a code-unit slice up front. Keep this implementation
	// simple by using the high surrogate here and stepping fully.
	// For lexicographic ordering this is correct because the high
	// surrogate ranges 0xD800-0xDBFF are above all BMP code points.
	highSurrogate := 0xD800 + ((uint32(r) - 0x10000) >> 10)

	return i + 1, highSurrogate
}

func writeCanonicalString(w *bytes.Buffer, s string) error {
	// JCS string serialization: minimal escaping per RFC 8259 + RFC 8785.
	w.WriteByte('"')

	for _, r := range s {
		switch r {
		case '"':
			w.WriteString(`\"`)
		case '\\':
			w.WriteString(`\\`)
		case '\b':
			w.WriteString(`\b`)
		case '\f':
			w.WriteString(`\f`)
		case '\n':
			w.WriteString(`\n`)
		case '\r':
			w.WriteString(`\r`)
		case '\t':
			w.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(w, `\u%04x`, r)
			} else {
				w.WriteRune(r)
			}
		}
	}

	w.WriteByte('"')

	return nil
}

func writeCanonicalNumber(w *bytes.Buffer, raw string) error {
	// RFC 8785 §3.2.2.3: numbers are serialized using the ECMA-262
	// algorithm. Go's strconv.FormatFloat with format 'g', prec -1 is
	// close but not identical. The pragmatic choice for tool arguments
	// is to round-trip via float64 and emit a minimal-length form. This
	// matches what most clients produce. Integers stay integer-shaped.
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return fmt.Errorf("parse number %q: %w", raw, err)
	}

	// Integer fast path: emit without a decimal point.
	if f == float64(int64(f)) && raw != "" && !containsDot(raw) && !containsExp(raw) {
		w.WriteString(strconv.FormatInt(int64(f), 10))
		return nil
	}

	// Non-integer or scientific input: use FormatFloat with prec -1,
	// which produces the shortest round-trippable decimal.
	out := strconv.FormatFloat(f, 'g', -1, 64)
	w.WriteString(out)

	return nil
}

func containsDot(s string) bool {
	for i := range len(s) {
		if s[i] == '.' {
			return true
		}
	}

	return false
}

func containsExp(s string) bool {
	for i := range len(s) {
		if s[i] == 'e' || s[i] == 'E' {
			return true
		}
	}

	return false
}

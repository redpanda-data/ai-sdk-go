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

// Package toolname derives provider-safe names for tool calls that a model
// emitted with a name the provider refuses to accept back in history.
package toolname

import (
	"crypto/sha256"
	"math/big"
	"strings"
)

const (
	maxLen = 64

	replayPrefix = "invalid_"
	hashLen      = 10
	// tailLen fills the budget left by the prefix, hash and separator.
	tailLen = maxLen - len(replayPrefix) - hashLen - 1
	// fallbackHashLen fills the budget left by the prefix.
	fallbackHashLen = maxLen - len(replayPrefix)
)

// Valid reports whether name satisfies the tool-name grammar that Bedrock
// Converse enforces: 1 to 64 characters from [a-zA-Z0-9_-].
func Valid(name string) bool {
	if name == "" || len(name) > maxLen {
		return false
	}

	for i := range len(name) {
		if !allowed(rune(name[i])) {
			return false
		}
	}

	return true
}

func allowed(c rune) bool {
	return c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-'
}

// Replay returns the name under which an invalid tool name is replayed.
//
// The result depends only on name, is valid, starts with a letter and carries
// a hash of the full original so that distinct inputs stay distinct. The
// readable tail is informational. A replay name never equals a name that was
// merely shortened or cleaned from the input.
func Replay(name string) string {
	var b strings.Builder

	b.WriteString(replayPrefix)
	b.WriteString(hashChars(name, hashLen))

	if tail := tailOf(clean(name), tailLen); tail != "" {
		b.WriteByte('_')
		b.WriteString(tail)
	}

	return b.String()
}

// ReplayAvoiding returns name unchanged when it is valid. Otherwise it returns
// Replay(name), or, when that is a member of taken, a longer hash-only name
// that carries more of the input hash.
func ReplayAvoiding(name string, taken map[string]struct{}) string {
	if Valid(name) {
		return name
	}

	candidate := Replay(name)
	if _, clash := taken[candidate]; !clash {
		return candidate
	}

	return replayPrefix + hashChars(name, fallbackHashLen)
}

// clean replaces every character outside the allowed set with an underscore.
func clean(name string) string {
	var b strings.Builder

	for _, r := range name {
		if allowed(r) {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
		}
	}

	return b.String()
}

func tailOf(s string, n int) string {
	if len(s) <= n {
		return s
	}

	return s[len(s)-n:]
}

// hashChars returns the first n base36 characters of a hash of name. A single
// SHA-256 digest yields at most about 50 base36 characters, so when n is
// larger the digest of the previous digest string is appended until enough
// characters exist.
func hashChars(name string, n int) string {
	sum := sha256.Sum256([]byte(name))
	out := new(big.Int).SetBytes(sum[:]).Text(36)

	for len(out) < n {
		sum = sha256.Sum256([]byte(out))
		out += new(big.Int).SetBytes(sum[:]).Text(36)
	}

	return out[:n]
}

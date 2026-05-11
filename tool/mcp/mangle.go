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
	"crypto/sha256"
	"math/big"
)

// maxToolNameLen is the maximum allowed length for tool names sent to LLM
// providers. Bedrock enforces 64 characters; this appears to be a common
// limit across providers.
const maxToolNameLen = 64

// mangleHeadIfTooLong truncates a tool name that exceeds maxLen by replacing
// the head with a deterministic hash prefix and keeping the tail (the most
// specific / human-readable part).
//
// The algorithm matches protoc-gen-go-mcp's MangleHeadIfTooLong so that names
// mangled at either layer use the same scheme.
//
// Output format for names that exceed maxLen:
//
//	{10-char-base36-hash}_{tail}
//
// The hash is derived from the full original name, so the mapping is stable
// and collision-resistant (~31 bits of entropy, birthday bound ~46k).
func mangleHeadIfTooLong(name string, maxLen int) string {
	if maxLen <= 0 {
		return ""
	}

	if len(name) <= maxLen {
		return name
	}

	hash := sha256.Sum256([]byte(name))
	fullHash := base36String(hash[:])

	hashPrefix := fullHash
	if len(hashPrefix) > 10 {
		hashPrefix = hashPrefix[:10]
	}

	if maxLen <= len(hashPrefix) {
		return hashPrefix[:maxLen]
	}

	available := maxLen - len(hashPrefix) - 1 // -1 for separator
	if available <= 0 {
		return hashPrefix
	}

	tail := name[len(name)-available:]

	return hashPrefix + "_" + tail
}

func base36String(b []byte) string {
	n := new(big.Int).SetBytes(b)
	return n.Text(36)
}

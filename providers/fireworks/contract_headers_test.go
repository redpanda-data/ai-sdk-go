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

package fireworks_test

import (
	"net/http"
	"slices"
	"strconv"
	"strings"
)

func contractHeaderKind(name string) string {
	switch strings.ToLower(name) {
	case "content-type":
		return "mime"
	case "fireworks-prompt-tokens", "fireworks-cached-prompt-tokens", "fireworks-output-tokens", "fireworks-prefill-tokens", "fireworks-server-time-to-first-token", "fireworks-server-processing-time", "x-fireworks-cached-prompt-tokens", "x-prompt-cache-hit-tokens", "x-prompt-cache-miss-tokens":
		return "number"
	case "x-request-id", "request-id", "fireworks-request-id", "x-fireworks-request-id", "fireworks-server-id", "x-fireworks-server-id", "x-served-by", "x-session-affinity", "x-prompt-cache-isolation-key", "x-claude-code-session-id":
		return "id"
	default:
		return ""
	}
}

func contractResponseMetadata(headers http.Header, key string) (map[string]string, []string) {
	kept := map[string]string{}

	names := make([]string, 0, len(headers))
	for name := range headers {
		if key != "" && strings.Contains(strings.ToLower(name), strings.ToLower(key)) {
			continue
		}

		if contractHeaderKind(name) == "" {
			continue
		}

		names = append(names, name)

		value := headers.Get(name)
		switch contractHeaderKind(name) {
		case "mime":
			if key != "" {
				value = strings.ReplaceAll(value, key, "[REDACTED]")
			}

			kept[name] = value
		case "number":
			if _, err := strconv.ParseFloat(value, 64); err == nil {
				kept[name] = value
			}
		case "id":
			if strings.HasPrefix(value, "capture_") && len(value) == 32 {
				kept[name] = value
			} else {
				kept[name] = "capture_" + contractHash([]byte(value))[:24]
			}
		}
	}

	slices.Sort(names)

	return kept, names
}

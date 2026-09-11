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

// Package tokens estimates request sizes for the context and tool-loading
// budgets. The heuristic counts high on purpose: an estimate that is low costs
// a dead session, one that is high costs a slightly early compaction. The
// reactive overflow path is the backstop for when the estimate is still wrong.
package tokens

import (
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

const (
	// CharsPerToken counts high: real tokenizers average ~4 chars/token on
	// English text, so dividing by 3 fires thresholds before providers reject.
	CharsPerToken = 3

	// PerMessageOverhead covers per-message framing (role, delimiters).
	PerMessageOverhead = 5
)

// Text estimates tokens for a string, rounding up.
func Text(s string) int {
	if len(s) == 0 {
		return 0
	}

	return (len(s) + CharsPerToken - 1) / CharsPerToken
}

// Part estimates tokens for one part. Unknown part types fall back to their
// marshalled length so future part kinds are never counted as free.
func Part(part llm.Part) int {
	switch p := part.(type) {
	case *llm.TextPart:
		if p == nil {
			return 0
		}

		return Text(p.Text)

	case *llm.ToolSearchPart:
		if p == nil {
			return 0
		}

		if len(p.Tools) > 0 {
			// Loaded schemas are counted in the fixed tool budget, regardless
			// of whether the provider stores references or schemas in history.
			total := 100
			for _, name := range p.Tools {
				total += Text(name)
			}

			return total
		}

		return Text(string(p.Data))

	case *llm.ReasoningPart:
		if p == nil {
			return 0
		}

		return Text(p.Text) + Text(p.Signature)

	case *llm.ToolRequestPart:
		if p == nil {
			return 0
		}

		return Text(p.Name) + Text(string(p.Arguments))

	case *llm.ToolResponsePart:
		if p == nil {
			return 0
		}

		return Text(p.Name) + Text(string(p.Result))

	default:
		raw, err := json.Marshal(part)
		if err != nil {
			return 0
		}

		return Text(string(raw))
	}
}

// Message estimates tokens for one message including framing overhead.
func Message(msg llm.Message) int {
	total := PerMessageOverhead
	for _, part := range msg.Content {
		total += Part(part)
	}

	return total
}

// History estimates tokens for a message slice.
func History(msgs []llm.Message) int {
	total := 0
	for _, msg := range msgs {
		total += Message(msg)
	}

	return total
}

// Tools estimates tokens for the tool schemas sent with every request.
func Tools(defs []llm.ToolDefinition) int {
	total := 0
	for _, def := range defs {
		total += Text(def.Name) + Text(def.Description) + Text(string(def.Parameters))
	}

	return total
}

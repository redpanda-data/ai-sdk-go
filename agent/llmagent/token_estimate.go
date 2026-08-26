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

package llmagent

import (
	"encoding/json"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Token counting for the compaction budget. Heuristic and counted high on
// purpose: an estimate that is low costs a dead session, one that is high
// costs a slightly early compaction. The reactive path is the backstop for
// when the estimate is still wrong.
const (
	// charsPerToken counts high: real tokenizers average ~4 chars/token on
	// English text, so dividing by 3 fires thresholds before providers reject.
	charsPerToken = 3

	// perMessageOverheadTokens covers per-message framing (role, delimiters).
	perMessageOverheadTokens = 5
)

// estimateTextTokens estimates tokens for a string, rounding up.
func estimateTextTokens(s string) int {
	if len(s) == 0 {
		return 0
	}

	return (len(s) + charsPerToken - 1) / charsPerToken
}

// estimatePartTokens estimates tokens for one part. Unknown part types fall back to
// their marshalled length so future part kinds are never counted as free.
func estimatePartTokens(part llm.Part) int {
	switch p := part.(type) {
	case *llm.TextPart:
		if p == nil {
			return 0
		}

		return estimateTextTokens(p.Text)

	case *llm.ReasoningPart:
		if p == nil {
			return 0
		}

		return estimateTextTokens(p.Text) + estimateTextTokens(p.Signature)

	case *llm.ToolRequestPart:
		if p == nil {
			return 0
		}

		return estimateTextTokens(p.Name) + estimateTextTokens(string(p.Arguments))

	case *llm.ToolResponsePart:
		if p == nil {
			return 0
		}

		return estimateTextTokens(p.Name) + estimateTextTokens(string(p.Result))

	default:
		raw, err := json.Marshal(part)
		if err != nil {
			return 0
		}

		return estimateTextTokens(string(raw))
	}
}

// estimateMessageTokens estimates tokens for one message including framing overhead.
func estimateMessageTokens(msg llm.Message) int {
	total := perMessageOverheadTokens
	for _, part := range msg.Content {
		total += estimatePartTokens(part)
	}

	return total
}

// estimateHistoryTokens estimates tokens for a message slice.
func estimateHistoryTokens(msgs []llm.Message) int {
	total := 0
	for _, msg := range msgs {
		total += estimateMessageTokens(msg)
	}

	return total
}

// measureContext breaks the estimated request footprint down by category for
// observability. Unknown part kinds count as text until they earn a category.
func measureContext(systemTokens, toolDefTokens int, msgs []llm.Message) agent.ContextUsage {
	var text, reasoning, toolCalls, toolResults, framing int

	for _, msg := range msgs {
		framing += perMessageOverheadTokens

		for _, part := range msg.Content {
			size := estimatePartTokens(part)

			switch part.(type) {
			case *llm.ReasoningPart:
				reasoning += size
			case *llm.ToolRequestPart:
				toolCalls += size
			case *llm.ToolResponsePart:
				toolResults += size
			default:
				text += size
			}
		}
	}

	u := agent.ContextUsage{
		SystemPrompt:    systemTokens,
		ToolDefinitions: toolDefTokens,
		Text:            text,
		Reasoning:       reasoning,
		ToolCalls:       toolCalls,
		ToolResults:     toolResults,
		Framing:         framing,
	}
	u.Total = u.SystemPrompt + u.ToolDefinitions + u.Text + u.Reasoning + u.ToolCalls + u.ToolResults + u.Framing

	return u
}

// estimateToolTokens estimates tokens for the tool schemas sent with every request.
func estimateToolTokens(defs []llm.ToolDefinition) int {
	total := 0
	for _, def := range defs {
		total += estimateTextTokens(def.Name) + estimateTextTokens(def.Description) + estimateTextTokens(string(def.Parameters))
	}

	return total
}

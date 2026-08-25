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
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Burst division: capping fresh tool results at collection time, before they
// enter the history. Compaction (compaction.go) never touches the unread
// frontier, so the only defense against a parallel burst of oversized results
// is to cap each one as it is collected.

// markerFloorTokens guarantees a capped result always has room for at least
// its marker object.
const markerFloorTokens = 128

// effectiveResultCap divides the remaining budget across a turn's tool calls
// before any tool runs, so a parallel burst cannot assemble an unfittable
// frontier. The same cap applies to every result independently of completion
// order. Without compaction the configured limit applies as-is; zero means
// uncapped. scale is the invocation's observed billed/estimated ratio.
//
// The marker floor wins over the remaining budget: when headroom is already
// gone, a burst may still exceed it by up to numCalls x markerFloorTokens.
// That is deliberate - a result smaller than its marker breaks pairing - and
// bounded small enough that the next turn's ensureFits reclaims it from
// older history.
func (a *LLMAgent) effectiveResultCap(countedRequest, numCalls int, scale float64) int {
	capTokens := a.config.toolResultLimit

	if a.config.compaction == nil || numCalls == 0 {
		return capTokens
	}

	headroom := a.deriveContextBudget(scale).hardLimit - countedRequest - perMessageOverheadTokens

	perCall := headroom / numCalls
	if capTokens == 0 || perCall < capTokens {
		capTokens = perCall
	}

	return max(capTokens, markerFloorTokens)
}

// capToolResult replaces a result over the cap with a truncation marker.
// The part's identity and error flag survive; zero cap means uncapped.
func capToolResult(part *llm.ToolResponsePart, capTokens int) *llm.ToolResponsePart {
	if capTokens <= 0 || estimatePartTokens(part) <= capTokens {
		return part
	}

	return &llm.ToolResponsePart{
		ID:      part.ID,
		Name:    part.Name,
		Result:  marshalMarker(part, markerTruncated),
		IsError: part.IsError,
	}
}

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

package catalog

import (
	"slices"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// WireAPISupport is the verified feature subset available through one API
// binding. A record with zero capabilities explicitly withholds all features;
// an absent record means feature support is unknown.
type WireAPISupport struct {
	API          WireAPI
	Capabilities llm.ModelCapabilities
	Reasoning    ReasoningSupport
}

func validateWireAPISupport(e Entry, fail func(string, ...any)) {
	seen := make(map[WireAPI]bool, len(e.WireAPISupport))

	for i, support := range e.WireAPISupport {
		api := support.API
		if !slices.Contains(e.WireAPIs, api) {
			fail("WireAPISupport[%d].API is not declared in WireAPIs", i)
		}

		if seen[api] {
			fail("WireAPISupport[%d] duplicates protocol %q and transport %q", i, api.Protocol, api.Transport)
		}

		seen[api] = true

		if protocol := support.Capabilities.ToolSearchProtocol; protocol != "" && protocol != e.Capabilities.ToolSearchProtocol {
			fail("WireAPISupport[%d].Capabilities.ToolSearchProtocol exceeds offering capabilities", i)
		}

		for _, feature := range []struct {
			name      string
			supported bool
			available bool
		}{
			{"Streaming", support.Capabilities.Streaming, e.Capabilities.Streaming},
			{"Tools", support.Capabilities.Tools, e.Capabilities.Tools},
			{"JSONMode", support.Capabilities.JSONMode, e.Capabilities.JSONMode},
			{"StructuredOutput", support.Capabilities.StructuredOutput, e.Capabilities.StructuredOutput},
			{"Vision", support.Capabilities.Vision, e.Capabilities.Vision},
			{"Audio", support.Capabilities.Audio, e.Capabilities.Audio},
			{"MultiTurn", support.Capabilities.MultiTurn, e.Capabilities.MultiTurn},
			{"SystemPrompts", support.Capabilities.SystemPrompts, e.Capabilities.SystemPrompts},
			{"Reasoning", support.Capabilities.Reasoning, e.Capabilities.Reasoning},
		} {
			if feature.supported && !feature.available {
				fail("WireAPISupport[%d].Capabilities.%s exceeds offering capabilities", i, feature.name)
			}
		}

		if (len(support.Reasoning.Efforts) > 0 || support.Reasoning.Adaptive || support.Reasoning.Budget || support.Reasoning.ThinkingPrefixCheck) && !support.Capabilities.Reasoning {
			fail("WireAPISupport[%d].Reasoning requires API Capabilities.Reasoning", i)
		}

		if support.Reasoning.Adaptive && !e.Reasoning.Adaptive {
			fail("WireAPISupport[%d].Reasoning.Adaptive exceeds offering reasoning support", i)
		}

		if support.Reasoning.Budget && !e.Reasoning.Budget {
			fail("WireAPISupport[%d].Reasoning.Budget exceeds offering reasoning support", i)
		}

		if support.Reasoning.ThinkingPrefixCheck && !e.Reasoning.ThinkingPrefixCheck {
			fail("WireAPISupport[%d].Reasoning.ThinkingPrefixCheck exceeds offering reasoning support", i)
		}

		if support.Reasoning.ThinkingPrefixCheck && !support.Reasoning.Adaptive {
			fail("WireAPISupport[%d].Reasoning.ThinkingPrefixCheck requires API Reasoning.Adaptive", i)
		}

		for _, effort := range support.Reasoning.Efforts {
			if !slices.Contains(e.Reasoning.Efforts, effort) {
				fail("WireAPISupport[%d].Reasoning.Efforts contains %q outside offering reasoning support", i, effort)
			}
		}
	}
}

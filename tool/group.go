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
	"errors"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Group collects tools written in code under one llm.ToolGroup and registers
// them in one call, the way an MCP client does for a server. Tools added with
// Add stay in the model's context; tools added with Defer are loaded on demand.
//
//	servicenow := tool.NewGroup(llm.ToolGroup{
//	    Name:         "servicenow",
//	    Description:  "ServiceNow incidents and CMDB lookups",
//	    Instructions: "Resolve the caller's sys_id before opening an incident.",
//	})
//	servicenow.Add(searchIncidents).Defer(createIncident, closeIncident)
//
//	if err := servicenow.Register(registry); err != nil { ... }
type Group struct {
	info     llm.ToolGroup
	always   []Tool
	deferred []Tool
}

// NewGroup starts a group with the given metadata.
func NewGroup(info llm.ToolGroup) *Group {
	return &Group{info: info}
}

// Add adds tools whose schemas are always in the model's context.
func (g *Group) Add(tools ...Tool) *Group {
	g.always = append(g.always, tools...)

	return g
}

// Defer adds tools whose schemas the model loads on demand.
func (g *Group) Defer(tools ...Tool) *Group {
	g.deferred = append(g.deferred, tools...)

	return g
}

// Register registers every tool of the group. opts apply to all of them, after
// the group's own options. On failure the tools registered by this call are
// removed again and the first error is returned, naming the tool.
func (g *Group) Register(registry Registry, opts ...Option) error {
	if registry == nil {
		return errors.New("tool: registry is nil")
	}

	var registered []string

	rollback := func() {
		for _, name := range registered {
			_ = registry.Unregister(name)
		}
	}

	register := func(t Tool, deferred bool) error {
		toolOpts := []Option{WithGroup(g.info)}
		if deferred {
			toolOpts = append(toolOpts, WithDeferred())
		}

		toolOpts = append(toolOpts, opts...)

		if err := registry.Register(t, toolOpts...); err != nil {
			return err
		}

		registered = append(registered, t.Definition().Name)

		return nil
	}

	for _, t := range g.always {
		if err := register(t, false); err != nil {
			rollback()

			return fmt.Errorf("tool: register group %q: %w", g.info.Name, err)
		}
	}

	for _, t := range g.deferred {
		if err := register(t, true); err != nil {
			rollback()

			return fmt.Errorf("tool: register group %q: %w", g.info.Name, err)
		}
	}

	return nil
}

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

package tool_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// deferralTool is a minimal tool that reports whatever definition it is built
// with, so a test can assert the registry's overlay rather than the tool's own
// self-description.
type deferralTool struct {
	def llm.ToolDefinition
}

func (t *deferralTool) Definition() llm.ToolDefinition { return t.def }

func (*deferralTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{}`), nil
}

// TestRegistryListOverlaysRegistrationPolicy pins the layering: a tool
// describes itself, and the deployment decides whether its schema is withheld
// and which group it belongs to. A tool that hard-codes Deferred/Group in its
// own Definition must not be able to override the registration, because
// deferral is a per-agent policy - the same tool is always-on in one agent and
// deferred in another.
func TestRegistryListOverlaysRegistrationPolicy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		selfDeferred bool
		selfGroup    string
		opts         []tool.Option
		wantDeferred bool
		wantGroup    string
	}{
		{
			name:         "no options leaves the tool always-on and ungrouped",
			wantDeferred: false,
			wantGroup:    "",
		},
		{
			name:         "WithDeferred defers",
			opts:         []tool.Option{tool.WithDeferred()},
			wantDeferred: true,
		},
		{
			name:      "WithGroup groups",
			opts:      []tool.Option{tool.WithGroup(llm.ToolGroup{Name: "servicenow"})},
			wantGroup: "servicenow",
		},
		{
			name:         "both compose",
			opts:         []tool.Option{tool.WithDeferred(), tool.WithGroup(llm.ToolGroup{Name: "jira"})},
			wantDeferred: true,
			wantGroup:    "jira",
		},
		{
			name:         "registration overrides a self-declared policy",
			selfDeferred: true,
			selfGroup:    "self",
			opts:         []tool.Option{tool.WithGroup(llm.ToolGroup{Name: "operator"})},
			wantDeferred: false,
			wantGroup:    "operator",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, registry.Register(&deferralTool{def: llm.ToolDefinition{
				Name:        "probe",
				Description: "a probe",
				Parameters:  json.RawMessage(`{"type":"object"}`),
				Deferred:    tt.selfDeferred,
				Group:       llm.ToolGroup{Name: tt.selfGroup},
			}}, tt.opts...))

			defs := registry.List()
			require.Len(t, defs, 1)
			assert.Equal(t, tt.wantDeferred, defs[0].Deferred)
			assert.Equal(t, tt.wantGroup, defs[0].Group.Name)
		})
	}
}

// TestRegistryListStableWithDeferral: the overlay must not disturb the
// name-sorted order that keeps the tools array byte-identical across calls.
func TestRegistryListStableWithDeferral(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	names := []string{"zebra", "alpha", "mike", "bravo", "yankee", "charlie"}
	for i, name := range names {
		opts := []tool.Option{tool.WithGroup(llm.ToolGroup{Name: "g"})}
		if i%2 == 0 {
			opts = append(opts, tool.WithDeferred())
		}

		require.NoError(t, registry.Register(&deferralTool{def: llm.ToolDefinition{
			Name:       name,
			Parameters: json.RawMessage(`{"type":"object"}`),
		}}, opts...))
	}

	first, err := json.Marshal(registry.List())
	require.NoError(t, err)

	for range 20 {
		again, err := json.Marshal(registry.List())
		require.NoError(t, err)
		require.JSONEq(t, string(first), string(again))
	}

	defs := registry.List()
	got := make([]string, len(defs))

	for i, def := range defs {
		got[i] = def.Name
	}

	assert.Equal(t, []string{"alpha", "bravo", "charlie", "mike", "yankee", "zebra"}, got)
}

// TestRegistryRejectsConflictingGroupMetadata: tools of one group may repeat or
// omit the group's description and instructions, but not contradict them.
func TestRegistryRejectsConflictingGroupMetadata(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	newTool := func(name string) *deferralTool {
		return &deferralTool{def: llm.ToolDefinition{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}}
	}

	require.NoError(t, registry.Register(newTool("a"), tool.WithGroup(llm.ToolGroup{
		Name: "svc", Description: "Service tools", Instructions: "Confirm first.",
	})))
	require.NoError(t, registry.Register(newTool("b"), tool.WithGroup(llm.ToolGroup{Name: "svc"})),
		"omitting the metadata is fine")
	require.NoError(t, registry.Register(newTool("c"), tool.WithGroup(llm.ToolGroup{
		Name: "svc", Description: "Service tools", Instructions: "Confirm first.",
	})), "repeating it is fine")

	err := registry.Register(newTool("d"), tool.WithGroup(llm.ToolGroup{Name: "svc", Instructions: "Never confirm."}))
	require.ErrorIs(t, err, tool.ErrInvalidToolConfig)
	assert.Contains(t, err.Error(), "different instructions")

	err = registry.Register(newTool("e"), tool.WithGroup(llm.ToolGroup{Description: "nameless"}))
	require.ErrorIs(t, err, tool.ErrInvalidToolConfig)
}

func TestRegistryResolvesGroupMetadataOnEveryMember(t *testing.T) {
	t.Parallel()

	for _, reverse := range []bool{false, true} {
		registry := tool.NewRegistry(tool.RegistryConfig{})

		members := []struct {
			name  string
			group llm.ToolGroup
		}{
			{name: "lookup", group: llm.ToolGroup{Name: "svc", Description: "Service records"}},
			{name: "create", group: llm.ToolGroup{Name: "svc", Instructions: "Confirm the tenant first."}},
			{name: "update", group: llm.ToolGroup{Name: "svc"}},
		}
		if reverse {
			members[0], members[2] = members[2], members[0]
		}

		for _, member := range members {
			require.NoError(t, registry.Register(&deferralTool{def: llm.ToolDefinition{
				Name: member.name, Parameters: json.RawMessage(`{"type":"object"}`),
			}}, tool.WithGroup(member.group)))
		}

		want := llm.ToolGroup{Name: "svc", Description: "Service records", Instructions: "Confirm the tenant first."}
		for _, def := range registry.List() {
			assert.Equal(t, want, def.Group, def.Name)
		}

		require.NoError(t, registry.Unregister("create"))

		for _, def := range registry.List() {
			assert.Empty(t, def.Group.Instructions, "metadata follows the current registered members")
			assert.Equal(t, "Service records", def.Group.Description)
		}
	}
}

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
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func groupProbe(name string) *deferralTool {
	return &deferralTool{def: llm.ToolDefinition{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}}
}

// TestGroupRegistersWithPolicy: Add keeps tools always loaded, Defer defers
// them, every tool carries the group metadata, and extra options apply to all.
func TestGroupRegistersWithPolicy(t *testing.T) {
	t.Parallel()

	info := llm.ToolGroup{Name: "svc", Description: "Service tools", Instructions: "Confirm first."}
	registry := tool.NewRegistry(tool.RegistryConfig{})

	err := tool.NewGroup(info).
		Add(groupProbe("search")).
		AddDeferred(groupProbe("create"), groupProbe("close")).
		Register(registry, tool.WithTimeout(time.Second))
	require.NoError(t, err)

	defs := registry.List()
	require.Len(t, defs, 3)

	deferred := map[string]bool{}

	for _, def := range defs {
		assert.Equal(t, info, def.Group, def.Name)
		deferred[def.Name] = def.Deferred
	}

	assert.Equal(t, map[string]bool{"search": false, "create": true, "close": true}, deferred)
}

// TestGroupRegisterRollsBackOnError: a failure part-way leaves the registry as
// it was before the call.
func TestGroupRegisterRollsBackOnError(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(groupProbe("taken")))

	err := tool.NewGroup(llm.ToolGroup{Name: "svc"}).
		Add(groupProbe("new")).
		AddDeferred(groupProbe("taken")).
		Register(registry)

	require.ErrorIs(t, err, tool.ErrToolAlreadyRegistered)
	assert.Contains(t, err.Error(), `group "svc"`)

	defs := registry.List()
	require.Len(t, defs, 1, "the tool registered before the failure is removed again")
	assert.Equal(t, "taken", defs[0].Name)

	require.Error(t, tool.NewGroup(llm.ToolGroup{Name: "svc"}).Register(nil))
	require.ErrorIs(t, tool.NewGroup(llm.ToolGroup{}).Add(groupProbe("a")).Register(tool.NewRegistry(tool.RegistryConfig{})),
		tool.ErrInvalidToolConfig, "a group needs a name")
}

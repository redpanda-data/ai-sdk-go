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
	"log/slog"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// TestRegistrationPolicy covers the deferral and grouping decisions the client
// makes for each tool it registers. They are deployment policy - nothing the
// MCP server tells us - so they are decided from the client's options alone.
func TestRegistrationPolicy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name           string
		opts           []ClientOption
		serverToolName string
		wantDeferred   bool
		wantGroup      string
	}{
		{
			name:           "default: always-on, grouped by server ID",
			serverToolName: "create_incident",
			wantDeferred:   false,
			wantGroup:      "servicenow",
		},
		{
			name:           "WithDeferredTools defers",
			opts:           []ClientOption{WithDeferredTools()},
			serverToolName: "create_incident",
			wantDeferred:   true,
			wantGroup:      "servicenow",
		},
		{
			name:           "WithAlwaysLoad exempts the entry point",
			opts:           []ClientOption{WithDeferredTools(), WithAlwaysLoad("search_incidents", "list_users")},
			serverToolName: "search_incidents",
			wantDeferred:   false,
			wantGroup:      "servicenow",
		},
		{
			name:           "WithAlwaysLoad does not exempt anything else",
			opts:           []ClientOption{WithDeferredTools(), WithAlwaysLoad("search_incidents")},
			serverToolName: "create_incident",
			wantDeferred:   true,
			wantGroup:      "servicenow",
		},
		{
			name:           "WithAlwaysLoad alone defers nothing",
			opts:           []ClientOption{WithAlwaysLoad("search_incidents")},
			serverToolName: "create_incident",
			wantDeferred:   false,
			wantGroup:      "servicenow",
		},
		{
			name:           "WithToolGroup overrides the server ID",
			opts:           []ClientOption{WithDeferredTools(), WithToolGroup(llm.ToolGroup{Name: "itsm"})},
			serverToolName: "create_incident",
			wantDeferred:   true,
			wantGroup:      "itsm",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			registry := tool.NewRegistry(tool.RegistryConfig{})

			client := &clientImpl{
				serverID: "servicenow",
				registry: registry,
				logger:   slog.New(slog.DiscardHandler),
			}
			for _, opt := range tt.opts {
				opt(client)
			}

			// Go through executeRegistryOps rather than asserting on the option
			// slice, so the test pins what the registry actually ends up with.
			client.executeRegistryOps([]registryOp{{
				register: &toolWrapper{
					client:         client,
					definition:     llm.ToolDefinition{Name: "servicenow__" + tt.serverToolName},
					serverToolName: tt.serverToolName,
				},
				serverName: tt.serverToolName,
			}})

			defs := registry.List()
			require.Len(t, defs, 1)
			assert.Equal(t, tt.wantDeferred, defs[0].Deferred)
			assert.Equal(t, tt.wantGroup, defs[0].Group.Name)
		})
	}
}

// TestRegistrationPolicyAppliesToEveryToolOfAServer checks the whole-server
// case an operator actually configures: defer the lot, keep the search entry
// point, and let the group default to the server ID.
func TestRegistrationPolicyAppliesToEveryToolOfAServer(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})

	client := &clientImpl{
		serverID: "jira",
		registry: registry,
		logger:   slog.New(slog.DiscardHandler),
	}
	WithDeferredTools()(client)
	WithAlwaysLoad("search_issues")(client)

	serverNames := []string{"search_issues", "create_issue", "transition_issue"}
	ops := make([]registryOp, 0, len(serverNames))

	for _, serverName := range serverNames {
		ops = append(ops, registryOp{
			register: &toolWrapper{
				client:         client,
				definition:     llm.ToolDefinition{Name: "jira__" + serverName},
				serverToolName: serverName,
			},
			serverName: serverName,
		})
	}

	client.executeRegistryOps(ops)

	deferred := map[string]bool{}

	for _, def := range registry.List() {
		assert.Equal(t, "jira", def.Group.Name)

		deferred[def.Name] = def.Deferred
	}

	assert.Equal(t, map[string]bool{
		"jira__search_issues":    false,
		"jira__create_issue":     true,
		"jira__transition_issue": true,
	}, deferred)
}

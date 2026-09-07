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
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic"
	"github.com/redpanda-data/ai-sdk-go/providers/anthropic/anthropictest"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock"
	"github.com/redpanda-data/ai-sdk-go/providers/bedrock/bedrocktest"
	"github.com/redpanda-data/ai-sdk-go/providers/google"
	"github.com/redpanda-data/ai-sdk-go/providers/google/googletest"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Live provider tests. Each provider skips without its credentials; see the
// provider *test packages for the environment variables.

// liveModels returns one model constructor per provider; each skips the test
// without its credentials.
func liveModels() map[string]func(t *testing.T) llm.Model {
	return map[string]func(t *testing.T) llm.Model{
		"anthropic": func(t *testing.T) llm.Model {
			t.Helper()

			p, err := anthropic.NewProvider(anthropictest.GetAPIKeyOrSkipTest(t), anthropic.WithTimeout(2*time.Minute))
			require.NoError(t, err)

			m, err := p.NewModel(anthropictest.TestModelName)
			require.NoError(t, err)

			return m
		},
		"openai": func(t *testing.T) llm.Model {
			t.Helper()

			p, err := openai.NewProvider(openaitest.GetAPIKeyOrSkipTest(t), openai.WithTimeout(2*time.Minute))
			require.NoError(t, err)

			m, err := p.NewModel(openaitest.TestModelName)
			require.NoError(t, err)

			return m
		},
		"google": func(t *testing.T) llm.Model {
			t.Helper()

			p, err := google.NewProvider(t.Context(), googletest.GetAPIKeyOrSkipTest(t), google.WithTimeout(2*time.Minute))
			require.NoError(t, err)

			m, err := p.NewModel(googletest.TestModelName)
			require.NoError(t, err)

			return m
		},
		"bedrock": func(t *testing.T) llm.Model {
			t.Helper()
			bedrocktest.SkipUnlessAWSCredentials(t)

			p, err := bedrock.NewProvider(t.Context())
			require.NoError(t, err)

			m, err := p.NewModel(bedrocktest.TestModelName)
			require.NoError(t, err)

			return m
		},
	}
}

// toolsSeen records the tools array of every model call.
type toolsSeen struct {
	mu    sync.Mutex
	calls [][]string
}

func (r *toolsSeen) InterceptModel(_ context.Context, info *agent.ModelCallInfo, next agent.ModelCallHandler) agent.ModelCallHandler {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.calls = append(r.calls, requestToolNames(info.Req))

	return next
}

func (r *toolsSeen) snapshot() [][]string {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([][]string(nil), r.calls...)
}

const adaSysID = "6816f79cc0a8016401c5a33be04be441"

// liveRegistry is a service desk with one always-on lookup, deferred actions in
// two groups, and instructions the model must follow to produce valid arguments.
func liveRegistry(t *testing.T) (tool.Registry, *stubTool, *stubTool) {
	t.Helper()

	create := &stubTool{
		def: llm.ToolDefinition{
			Name: incidentTool, Description: "Open a ServiceNow incident on behalf of a caller.",
			Parameters: json.RawMessage(`{"type":"object","properties":{
				"caller_sys_id":{"type":"string","description":"sys_id of the affected user"},
				"short_description":{"type":"string"},
				"urgency":{"type":"string","enum":["low","medium","high"]}},
				"required":["caller_sys_id","short_description"]}`),
		},
		result: json.RawMessage(`{"number":"INC0012345","state":"new"}`),
	}
	closeIncident := &stubTool{
		def: llm.ToolDefinition{
			Name: "servicenow__close_incident", Description: "Close a ServiceNow incident with a resolution code.",
			Parameters: json.RawMessage(`{"type":"object","properties":{
				"number":{"type":"string"},
				"resolution_code":{"type":"string","enum":["solved","workaround","not_reproducible"]}},
				"required":["number","resolution_code"]}`),
		},
		result: json.RawMessage(`{"number":"INC0012345","state":"closed"}`),
	}
	searchUsers := &stubTool{
		def: llm.ToolDefinition{
			Name: "servicenow__search_users", Description: "Find a ServiceNow user by name and return their sys_id.",
			Parameters: json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}`),
		},
		result: json.RawMessage(`{"users":[{"sys_id":"` + adaSysID + `","name":"Ada Lovelace"}]}`),
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, tool.NewGroup(llm.ToolGroup{
		Name:         "servicenow",
		Description:  "ServiceNow incidents",
		Instructions: "Resolve the caller with servicenow__search_users before opening an incident and pass the returned sys_id as caller_sys_id.",
	}).Add(searchUsers).Defer(create, closeIncident).Register(registry))
	require.NoError(t, tool.NewGroup(llm.ToolGroup{Name: "jira", Description: "Jira issues"}).Defer(&stubTool{def: llm.ToolDefinition{
		Name: "jira__create_issue", Description: "Create a Jira issue.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"project":{"type":"string"},"summary":{"type":"string"}},"required":["project","summary"]}`),
	}}).Register(registry))

	return registry, create, closeIncident
}

// TestToolLoadingLiveProviders_Integration runs the deferred-tool flow on every
// provider with credentials: discover a deferred tool, load it, call it with
// arguments that satisfy its schema and the group's instructions, and keep it
// loaded into the next turn.
func TestToolLoadingLiveProviders_Integration(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	for name, build := range liveModels() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			model := build(t)
			registry, create, closeIncident := liveRegistry(t)
			seen := &toolsSeen{}

			ag, err := New("service-desk",
				"You are an internal service desk agent. Use the tools to act on the user's request without asking for confirmation.",
				model, WithTools(registry), WithMaxTurns(8), WithInterceptors(seen))
			require.NoError(t, err)

			sess := &session.State{ID: "live-" + name, Messages: userMessage("Open an incident for Ada Lovelace: her laptop will not boot.")}
			require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))
			logTranscript(t, sess)

			require.Equal(t, int32(1), create.calls.Load(), "create_incident must execute exactly once")

			var args struct {
				CallerSysID      string `json:"caller_sys_id"`
				ShortDescription string `json:"short_description"`
			}
			require.NoError(t, json.Unmarshal(create.arguments()[0], &args))
			assert.Equal(t, adaSysID, args.CallerSysID, "the sys_id must come from search_users, per the group instructions")
			assert.NotEmpty(t, args.ShortDescription)
			assert.Contains(t, sess.Metadata[loadedToolsMetadataKey], incidentTool)

			turn1 := len(seen.snapshot())
			assert.LessOrEqual(t, turn1, 6, "discovery should cost at most a couple of extra round trips")

			// Turn 2: the loaded tool is still present; a second one is discovered.
			sess.Messages = append(sess.Messages, userMessage("Now close that incident as solved.")...)
			require.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))

			calls := seen.snapshot()
			assert.Contains(t, calls[turn1], incidentTool, "turn 2 starts with the loaded tool present")
			assert.NotContains(t, calls[turn1], "servicenow__close_incident")
			require.Equal(t, int32(1), closeIncident.calls.Load())

			var closeArgs struct {
				Number         string `json:"number"`
				ResolutionCode string `json:"resolution_code"`
			}
			require.NoError(t, json.Unmarshal(closeIncident.arguments()[0], &closeArgs))
			assert.Equal(t, "INC0012345", closeArgs.Number)
			assert.Equal(t, "solved", closeArgs.ResolutionCode)

			t.Logf("%s: turn 1 %d model calls, turn 2 %d, loaded %v", name, turn1, len(calls)-turn1, sess.Metadata[loadedToolsMetadataKey])
		})
	}
}

// TestToolLoadingStaleHistory_Integration resumes a transcript that references
// tool_search and a deferred tool that no longer exist, so the request declares
// neither. Every provider must accept the history as-is.
func TestToolLoadingStaleHistory_Integration(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test in short mode")
	}

	for name, build := range liveModels() {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			model := build(t)

			registry := tool.NewRegistry(tool.RegistryConfig{})
			require.NoError(t, registry.Register(&stubTool{def: llm.ToolDefinition{
				Name: "todo_write", Description: "Record the plan for a multi-step task.",
				Parameters: json.RawMessage(`{"type":"object","properties":{"plan":{"type":"string"}}}`),
			}}))

			seen := &toolsSeen{}
			ag, err := New("service-desk", "You are a service desk agent.", model, WithTools(registry), WithMaxTurns(3), WithInterceptors(seen))
			require.NoError(t, err)

			sess := &session.State{
				ID:       "stale-" + name,
				Metadata: map[string]any{loadedToolsMetadataKey: []any{incidentTool}},
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Open an incident for Ada: her laptop will not boot.")),
					llm.NewMessage(llm.RoleAssistant, toolCall("s1", "tool_search", `{"query":"select:`+incidentTool+`"}`)),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("s1", "tool_search", json.RawMessage(`{"loaded":["`+incidentTool+`"],"note":"1 tool(s) loaded."}`), false)),
					llm.NewMessage(llm.RoleAssistant, toolCall("c1", incidentTool, `{"caller_sys_id":"`+adaSysID+`","short_description":"Laptop will not boot"}`)),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("c1", incidentTool, json.RawMessage(`{"number":"INC0012345"}`), false)),
					llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Opened INC0012345 for Ada.")),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Which incident number did you open? Answer with the number only.")),
				},
			}

			assert.Equal(t, agent.FinishReasonStop, finishReason(runAgent(t, ag, sess)))
			assert.Equal(t, []string{"todo_write"}, seen.snapshot()[0], "nothing deferred: no tool_search, no stale tool declared")

			last := sess.Messages[len(sess.Messages)-1]
			assert.Equal(t, llm.RoleAssistant, last.Role)
			assert.Contains(t, last.TextContent(), "INC0012345")
		})
	}
}

// logTranscript prints every tool call and result, so a run that took more
// round trips than expected explains itself.
func logTranscript(t *testing.T, sess *session.State) {
	t.Helper()

	for _, msg := range sess.Messages {
		for _, req := range msg.ToolRequests() {
			t.Logf("  -> %s %s", req.Name, req.Arguments)
		}

		for _, resp := range msg.ToolResponses() {
			t.Logf("  <- %s %s", resp.Name, resp.Result)
		}
	}
}

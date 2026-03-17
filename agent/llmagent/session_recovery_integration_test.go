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

package llmagent_test

import (
	"context"
	_ "embed"
	"encoding/json"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

//go:embed testdata/session_recovery_single.json
var sessionRecoverySingleJSON []byte

//go:embed testdata/session_recovery_multiple.json
var sessionRecoveryMultipleJSON []byte

// TestSessionRecovery_Single verifies recovery from a single incomplete tool call.
//
// Loads a session with pattern: [user, assistant(tool_request), user] where the
// tool request has no response. The agent should detect this, execute the
// incomplete tool, insert the response, and proceed normally.
func TestSessionRecovery_Single(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), integrationTestTimeout)
	defer cancel()

	// Parse session state from testdata
	sess, incompleteReqs := loadRecoverySession(t, sessionRecoverySingleJSON)
	require.Len(t, incompleteReqs, 1, "expected 1 incomplete tool request")

	t.Logf("Incomplete tool: id=%s name=%s", incompleteReqs[0].ID, incompleteReqs[0].Name)

	// Create mock tool matching the incomplete request
	registry := tool.NewRegistry(tool.RegistryConfig{})
	err := registry.Register(&mockTool{
		definition: llm.ToolDefinition{
			Name:        incompleteReqs[0].Name,
			Description: "Mock tool for session recovery test",
			Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
		},
		executeFn: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return json.RawMessage(`{"status":"recovered"}`), nil
		},
	})
	require.NoError(t, err)

	// Create agent
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	ag, err := llmagent.New("test-agent", "You are a helpful assistant.", model,
		llmagent.WithTools(registry),
	)
	require.NoError(t, err)

	// Run agent directly with incomplete session
	inv := agent.NewInvocationMetadata(sess, agent.Info{})
	events := collectEvents(t, ag.Run(ctx, inv))

	// Verify no errors, completed successfully
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Verify incomplete tool was recovered
	toolResponses := filterEvents[agent.ToolResponseEvent](events)
	var recovered bool

	for _, evt := range toolResponses {
		if evt.Response.ID == incompleteReqs[0].ID {
			recovered = true
			break
		}
	}

	assert.True(t, recovered, "incomplete tool call should have been recovered")
}

// TestSessionRecovery_Multiple verifies recovery when the assistant made
// multiple parallel tool requests before interruption.
func TestSessionRecovery_Multiple(t *testing.T) {
	t.Parallel()

	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), integrationTestTimeout)
	defer cancel()

	// Parse session state from testdata
	sess, incompleteReqs := loadRecoverySession(t, sessionRecoveryMultipleJSON)
	require.Len(t, incompleteReqs, 3, "expected 3 incomplete tool requests")

	for i, req := range incompleteReqs {
		t.Logf("Incomplete tool %d: id=%s name=%s", i+1, req.ID, req.Name)
	}

	// Track tool executions (guarded by mutex since executeTools runs concurrently)
	var (
		mu             sync.Mutex
		executedCities []string
	)

	// Create mock tool
	registry := tool.NewRegistry(tool.RegistryConfig{})
	err := registry.Register(&mockTool{
		definition: llm.ToolDefinition{
			Name:        "get_weather",
			Description: "Get weather for a city",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		},
		executeFn: func(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
			var params struct {
				City string `json:"city"`
			}

			if err := json.Unmarshal(args, &params); err != nil {
				return nil, err
			}

			mu.Lock()
			executedCities = append(executedCities, params.City)

			mu.Unlock()

			weather := map[string]string{
				"Paris":  "Sunny, 22C",
				"London": "Cloudy, 15C",
				"Tokyo":  "Rainy, 18C",
			}

			return json.Marshal(map[string]string{
				"city":    params.City,
				"weather": weather[params.City],
			})
		},
	})
	require.NoError(t, err)

	// Create agent
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	ag, err := llmagent.New("weather-agent",
		"You are a helpful weather assistant.",
		model,
		llmagent.WithTools(registry),
	)
	require.NoError(t, err)

	// Run agent
	inv := agent.NewInvocationMetadata(sess, agent.Info{})
	events := collectEvents(t, ag.Run(ctx, inv))

	// Verify completion
	endEvent := findInvocationEndEvent(events)
	require.NotNil(t, endEvent)
	assert.Equal(t, agent.FinishReasonStop, endEvent.FinishReason)

	// Verify all 3 incomplete tools were executed
	assert.Len(t, executedCities, 3, "all 3 incomplete tools should have been executed")
	t.Logf("Executed cities: %v", executedCities)

	// Verify all incomplete IDs have responses
	toolResponses := filterEvents[agent.ToolResponseEvent](events)

	incompleteIDs := make(map[string]bool)
	for _, req := range incompleteReqs {
		incompleteIDs[req.ID] = true
	}

	recoveredCount := 0

	for _, evt := range toolResponses {
		if incompleteIDs[evt.Response.ID] {
			recoveredCount++
		}
	}

	assert.Equal(t, 3, recoveredCount, "all 3 incomplete tool calls should have responses")
}

// loadRecoverySession parses a protojson session and extracts incomplete tool requests.
func loadRecoverySession(t *testing.T, data []byte) (*session.State, []*llm.ToolRequest) {
	t.Helper()

	var protoState llmpb.SessionState
	err := protojson.Unmarshal(data, &protoState)
	require.NoError(t, err, "failed to parse session JSON")

	sess, err := kvstore.FromProtoSessionState(&protoState)
	require.NoError(t, err, "failed to convert proto to session")

	// Find incomplete tool requests (in second-to-last message if pattern matches)
	if len(sess.Messages) < 2 {
		return sess, nil
	}

	prevMsg := sess.Messages[len(sess.Messages)-2]
	if prevMsg.Role == llm.RoleAssistant {
		return sess, prevMsg.ToolRequests()
	}

	return sess, nil
}

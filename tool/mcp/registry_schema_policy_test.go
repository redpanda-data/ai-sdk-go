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
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func TestRegistryMalformedSchemaFailsWholeAgentStep(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{false, true} {
		for _, test := range []struct {
			name   string
			schema string
			field  string
		}{
			{"properties array", `{"type":"object","properties":[]}`, "properties"},
			{"nonobject property", `{"type":"object","properties":{"q":"string"}}`, `property "q"`},
		} {
			t.Run(fmt.Sprintf("legacy=%t/%s", legacy, test.name), func(t *testing.T) {
				t.Parallel()

				peer := newRegistryPeer(t, legacy)
				var malformed atomic.Bool
				malformed.Store(true)
				peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
					return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
						result, err := next(ctx, method, req)
						if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil && malformed.Load() {
							listed.Tools = append(listed.Tools, &sdkmcp.Tool{
								Name: "broken", InputSchema: json.RawMessage(test.schema),
							})
						}

						return result, err
					}
				})
				remote := newContextRegistry(t, peer)
				healthyPeer := newRegistryPeer(t, legacy)
				healthy, err := NewRegistry("healthy", healthyPeer.url, WithCredentials(registryContextCredentials))
				require.NoError(t, err)
				t.Cleanup(func() { assert.NoError(t, healthy.Close()) })
				var nativeCalls atomic.Int64
				local := tool.NewRegistry(tool.RegistryConfig{})
				require.NoError(t, local.Register(registrySchemaPolicyNativeTool{calls: &nativeCalls}))

				model := fakellm.NewFakeModel()
				model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
					definitions := registryNames(req.Tools)
					assert.Contains(t, definitions, "local_lookup")

					owner := "alice"

					for _, name := range definitions {
						if name == "remote__bob_only" {
							owner = "bob"
						}
					}

					assert.Contains(t, definitions, "remote__"+owner+"_only")
					assert.Contains(t, definitions, "healthy__"+owner+"_only")

					return &llm.Response{
						Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ready")), FinishReason: llm.FinishReasonStop,
					}, nil
				})
				ag, err := llmagent.New("schema-policy", "Use the available tools.", model, llmagent.WithTools(local, healthy, remote))
				require.NoError(t, err)

				ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
				defer cancel()

				valid, err := healthy.List(ctx)
				require.NoError(t, err)
				require.Contains(t, registryNames(valid), "healthy__alice_only")

				for range 2 {
					errors := registrySchemaPolicyRun(ctx, ag)
					require.NotEmpty(t, errors)

					for _, err := range errors {
						require.ErrorContains(t, err, `server "remote" tool "broken"`)
						require.ErrorContains(t, err, test.field)
						var schemaErr *json.UnmarshalTypeError
						require.ErrorAs(t, err, &schemaErr, "the diagnostic must preserve the underlying error")
					}
				}

				assert.Zero(t, model.CallCount(), "valid remote/local tools must not turn a failed snapshot into a partial agent step")
				assert.Zero(t, nativeCalls.Load())
				assert.Len(t, peer.requests(registryParameterListMethod), 2, "both failures must come from real MCP discovery")
				assert.Empty(t, peer.requests(registryAgentCallMethod))
				assert.Empty(t, healthyPeer.requests(registryAgentCallMethod))

				bob, cancelBob := context.WithTimeout(registryCaller(t.Context(), "bob", "bob:one"), 10*time.Second)
				defer cancelBob()

				require.Empty(t, registrySchemaPolicyRun(bob, ag), "another caller's valid catalog remains usable")
				assert.Equal(t, 1, model.CallCount())

				malformed.Store(false)
				require.Empty(t, registrySchemaPolicyRun(ctx, ag), "fixing the server restores the same reusable agent")
				assert.Equal(t, 2, model.CallCount())
				assert.Zero(t, nativeCalls.Load())
			})
		}
	}
}

func registrySchemaPolicyRun(ctx context.Context, ag *llmagent.LLMAgent) []error {
	state := &session.State{ID: "schema-policy", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))}}
	var failures []error

	for _, err := range ag.Run(ctx, agent.NewInvocationMetadata(state, ag.Info())) {
		if err != nil {
			failures = append(failures, err)
		}
	}

	return failures
}

type registrySchemaPolicyNativeTool struct{ calls *atomic.Int64 }

func (registrySchemaPolicyNativeTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{Name: "local_lookup", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}
}

func (t registrySchemaPolicyNativeTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	t.calls.Add(1)

	return json.RawMessage(`{}`), nil
}

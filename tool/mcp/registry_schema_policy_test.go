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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

	external := newRegistrySchemaExternalPeer(t)
	unmarshalType := func(t *testing.T, err error) {
		t.Helper()

		var schemaErr *json.UnmarshalTypeError
		require.ErrorAs(t, err, &schemaErr, "the diagnostic must preserve the underlying decode error")
	}

	for _, legacy := range []bool{false, true} {
		for _, test := range []struct {
			name   string
			schema string
			want   string
			cause  func(*testing.T, error)
		}{
			{"properties array", `{"type":"object","properties":[]}`, "properties", unmarshalType},
			{"nonobject property", `{"type":"object","properties":{"q":"string"}}`, `property "q"`, unmarshalType},
			{"absent schema", `null`, "input schema: must be a JSON object", nil},
			{"numeric root type", `{"type":42}`, `invalid value for "type"`, nil},
			{"numeric nested type", `{"type":"object","properties":{"q":{"type":42}}}`, `invalid value for "type"`, nil},
			{"empty root type", `{"type":""}`, `empty type name at ""`, nil},
			{"empty nested type", `{"type":"object","properties":{"q":{"type":""}}}`, `empty type name at "/properties/q"`, nil},
			{"string required", `{"type":"object","required":"q"}`, "required", unmarshalType},
			{"unknown nested type", `{"type":"object","properties":{"q":{"anyOf":[{"type":"string"},{"type":"strin"}]}}}`, `unknown type "strin" at "/properties/q/anyOf/1"`, nil},
			{"dangling local reference", `{"type":"object","properties":{"q":{"$ref":"#/$defs/missing"}}}`, `"/$defs/missing"`, nil},
			{"external reference", `{"type":"object","properties":{"q":{"$ref":"{external}/schema.json"}}}`, "external schema references are not supported", nil},
		} {
			t.Run(fmt.Sprintf("legacy=%t/%s", legacy, test.name), func(t *testing.T) {
				t.Parallel()

				f := newRegistrySchemaPolicyFixture(t, legacy, external.expand(test.schema))

				ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
				defer cancel()

				for range 2 {
					f.requireStepFailure(ctx, t, test.want, test.cause)
				}

				f.requireNoModelOrToolCalls(t)
				assert.Len(t, f.peer.requests(registryParameterListMethod), 2, "without a server TTL both failures must come from real MCP discovery")

				bob, cancelBob := context.WithTimeout(registryCaller(t.Context(), "bob", "bob:one"), 10*time.Second)
				defer cancelBob()

				require.Empty(t, registrySchemaPolicyRun(bob, f.agent), "another caller's valid catalog remains usable")
				assert.Equal(t, 1, f.model.CallCount())

				lists := f.peer.requests(registryParameterListMethod)
				require.Len(t, lists, 3)
				assert.Equal(t, "Bearer bob:one", lists[2].auth, "bob's discovery must not reuse alice's failed partition")

				f.malformed.Store(false)
				require.Empty(t, registrySchemaPolicyRun(ctx, f.agent), "fixing the server restores the same reusable agent")
				assert.Equal(t, 2, f.model.CallCount())
				assert.Zero(t, f.nativeCalls.Load())
			})
		}
	}
}

// A positive server TTL serves the second step from the cached catalog; it must
// be validated again, not trusted because an earlier discovery fetched it.
func TestRegistryCachedMalformedSchemaFailsEveryAgentStep(t *testing.T) {
	t.Parallel()

	const ttl = time.Minute

	for _, legacy := range []bool{false, true} {
		for _, recovery := range []string{"notification", "ttl expiry"} {
			t.Run(fmt.Sprintf("legacy=%t/%s", legacy, recovery), func(t *testing.T) {
				t.Parallel()

				f := newRegistrySchemaPolicyFixture(t, legacy, `{"type":"object","properties":{"q":{"type":"strin"}}}`, WithRegistryMaxToolCacheTTL(ttl))
				clock := newRegistryTestClock()
				f.remote.config.now = clock.Now
				sent := registryAdvertiseToolTTL(f.peer, time.Hour, false)
				const want = `unknown type "strin" at "/properties/q"`

				ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
				defer cancel()

				for range 2 {
					f.requireStepFailure(ctx, t, want, nil)
				}

				f.requireNoModelOrToolCalls(t)
				require.Len(t, f.peer.requests(registryParameterListMethod), 1, "the second step must use the cached catalog")

				bob, cancelBob := context.WithTimeout(registryCaller(t.Context(), "bob", "bob:one"), 10*time.Second)
				defer cancelBob()

				require.Empty(t, registrySchemaPolicyRun(bob, f.agent), "another caller's valid catalog remains usable")
				assert.Equal(t, 1, f.model.CallCount())

				lists := f.peer.requests(registryParameterListMethod)
				require.Len(t, lists, 2)
				assert.Equal(t, "Bearer bob:one", lists[1].auth, "bob's discovery must use his own credentials")

				f.requireStepFailure(ctx, t, want, nil)
				require.Len(t, f.peer.requests(registryParameterListMethod), 2, "alice's cached catalog is still rejected without refetching")

				f.malformed.Store(false)

				switch recovery {
				case "notification":
					waitForChange := registryToolChangeBarrier(t, f.remote, "alice", sent)
					f.peer.addTool("alice", "alice_only", "original", "old")
					waitForChange()
				default:
					clock.Advance(ttl)
				}

				require.Empty(t, registrySchemaPolicyRun(ctx, f.agent), "fixing the server restores the same reusable agent")
				assert.Equal(t, 2, f.model.CallCount())
				assert.Len(t, f.peer.requests(registryParameterListMethod), 3, "recovery must refetch the fixed catalog")
				assert.Zero(t, f.nativeCalls.Load())
			})
		}
	}
}

type registrySchemaPolicyFixture struct {
	peer        *registryPeer
	healthyPeer *registryPeer
	remote      *Registry
	malformed   atomic.Bool
	nativeCalls atomic.Int64
	model       *fakellm.FakeModel
	agent       *llmagent.LLMAgent
}

// newRegistrySchemaPolicyFixture adds a "broken" tool with schema to alice's
// catalog until malformed is cleared. The agent also has a local tool and a
// healthy MCP source; its model asserts the broken tool never reaches it.
func newRegistrySchemaPolicyFixture(t *testing.T, legacy bool, schema string, opts ...RegistryOption) *registrySchemaPolicyFixture {
	t.Helper()

	f := &registrySchemaPolicyFixture{peer: newRegistryPeer(t, legacy), healthyPeer: newRegistryPeer(t, legacy)}
	f.malformed.Store(true)
	f.peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			result, err := next(ctx, method, req)
			if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil && f.malformed.Load() {
				listed.Tools = append(listed.Tools, &sdkmcp.Tool{
					Name: "broken", InputSchema: json.RawMessage(schema),
				})
			}

			return result, err
		}
	})
	f.remote = newContextRegistry(t, f.peer, opts...)
	healthy, err := NewRegistry("healthy", f.healthyPeer.url, WithCredentials(registryContextCredentials))
	require.NoError(t, err)
	t.Cleanup(func() { assert.NoError(t, healthy.Close()) })

	local := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, local.Register(registrySchemaPolicyNativeTool{calls: &f.nativeCalls}))

	f.model = fakellm.NewFakeModel()
	f.model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
		definitions := registryNames(req.Tools)
		assert.NotContains(t, definitions, "remote__broken", "a malformed schema must never reach the model")
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
	f.agent, err = llmagent.New("schema-policy", "Use the available tools.", f.model, llmagent.WithTools(local, healthy, f.remote))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
	defer cancel()

	valid, err := healthy.List(ctx)
	require.NoError(t, err)
	require.Contains(t, registryNames(valid), "healthy__alice_only")

	return f
}

func (f *registrySchemaPolicyFixture) requireStepFailure(ctx context.Context, t *testing.T, want string, cause func(*testing.T, error)) {
	t.Helper()

	errors := registrySchemaPolicyRun(ctx, f.agent)
	require.NotEmpty(t, errors, "the malformed schema must fail the agent step")

	for _, err := range errors {
		require.ErrorContains(t, err, `server "remote" tool "broken"`)
		require.ErrorContains(t, err, want)

		if cause != nil {
			cause(t, err)
		}
	}
}

func (f *registrySchemaPolicyFixture) requireNoModelOrToolCalls(t *testing.T) {
	t.Helper()

	assert.Zero(t, f.model.CallCount(), "valid remote/local tools must not turn a failed snapshot into a partial agent step")
	assert.Zero(t, f.nativeCalls.Load())
	assert.Empty(t, f.peer.requests(registryAgentCallMethod))
	assert.Empty(t, f.healthyPeer.requests(registryAgentCallMethod))
}

// Valid schemas that exercise references, boolean subschemas, unions, both
// supported dialects and non-schema keyword values must reach the model unchanged.
var registryValidComplexSchemas = map[string]string{
	"local_refs": `{"type":"object","$defs":{"node":{"type":"object","properties":{"children":{"type":"array","items":{"$ref":"#/$defs/node"}},` +
		`"label":{"$ref":"#label"}}},"label":{"$anchor":"label","type":["string","null"]}},` +
		`"properties":{"root":{"$ref":"#/$defs/node"},"self":{"$ref":"#"}},"required":["root"]}`,
	"annotations": `{"type":"object","x-extension":{"type":42,"required":"q"},"$comment":"remote","title":"Lookup",` +
		`"properties":{"any":true,"never":false,"route":{"type":"string","x-mcp-header":"Tenant","x-vendor":{"type":42}},` +
		`"choice":{"oneOf":[{"type":"integer","minimum":0},{"type":"null"}]},"type":{"type":"string","enum":["a","b"]},` +
		`"literal":{"const":{"type":"bogus"},"default":{"type":42},"examples":[{"required":"q"}],"deprecated":true},` +
		`"blob":{"type":"string","format":"byte","contentEncoding":"base64","readOnly":true}},` +
		`"patternProperties":{"^x-":{"type":"string","pattern":"^[a-z]+$"}},"additionalProperties":false}`,
	"draft07": `{"$schema":"http://json-schema.org/draft-07/schema#","type":"object",` +
		`"definitions":{"pair":{"type":"array","items":[{"type":"string"},{"type":"number"}],"additionalItems":false}},` +
		`"properties":{"pair":{"$ref":"#/definitions/pair"},"other":{"type":"integer"}},"dependencies":{"pair":["other"]}}`,
	"applicators": `{"type":"object","properties":{"kind":{"type":"string"},"tuple":{"type":"array","prefixItems":[{"type":"string"}],` +
		`"items":{"type":"number"},"contains":{"type":"number"},"minContains":1,"unevaluatedItems":false}},` +
		`"if":{"properties":{"kind":{"const":"a"}}},"then":{"required":["tuple"]},"else":{"not":{"required":["tuple"]}},` +
		`"allOf":[{"propertyNames":{"pattern":"^[a-z]+$"}}],"anyOf":[true,{"minProperties":1}],` +
		`"dependentSchemas":{"kind":{"properties":{"kind":{"type":"string"}}}},"dependentRequired":{"tuple":["kind"]},"unevaluatedProperties":false}`,
	"identified": `{"$id":"{external}/tool.json","type":"object","properties":{"a":{"$ref":"{external}/tool.json#/$defs/a"}},` +
		`"$defs":{"a":{"type":"string"}}}`,
	// Empty type names in data, extensions and case-variant (non-keyword) members are not schema types.
	"empty_type_data": `{"type":"object","x-vendor":{"type":"","properties":{"r":{"type":""}}},"description":"","Type":"",` +
		`"properties":{"type":{"type":"string","const":{"type":""},"default":{"type":""},"examples":[{"type":""}],"enum":[{"type":""},""]},` +
		`"items":{"type":"array","items":{"type":"string","x-meta":{"type":""}}},"Properties":{"Type":""}},` +
		`"$defs":{"type":{"type":"string"}},"required":["type"],"dependentRequired":{"type":["items"]}}`,
}

func TestRegistryValidComplexSchemasReachModel(t *testing.T) {
	t.Parallel()

	external := newRegistrySchemaExternalPeer(t)

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, legacy)
			peer.servers["alice"].AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
				return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
					result, err := next(ctx, method, req)
					if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil {
						for name, schema := range registryValidComplexSchemas {
							listed.Tools = append(listed.Tools, &sdkmcp.Tool{Name: "valid_" + name, InputSchema: json.RawMessage(external.expand(schema))})
						}
					}

					return result, err
				}
			})
			remote := newContextRegistry(t, peer)

			var mu sync.Mutex
			seen := make(map[string]json.RawMessage)
			model := fakellm.NewFakeModel()
			model.When(fakellm.Any()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				mu.Lock()
				defer mu.Unlock()

				for _, definition := range req.Tools {
					seen[definition.Name] = definition.Parameters
				}

				return &llm.Response{
					Message: llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("ready")), FinishReason: llm.FinishReasonStop,
				}, nil
			})
			ag, err := llmagent.New("schema-policy", "Use the available tools.", model, llmagent.WithTools(remote))
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
			defer cancel()

			require.Empty(t, registrySchemaPolicyRun(ctx, ag))
			require.Equal(t, 1, model.CallCount())

			mu.Lock()
			defer mu.Unlock()

			assert.Contains(t, seen, "remote__alice_only")

			for name, schema := range registryValidComplexSchemas {
				parameters, ok := seen["remote__valid_"+name]
				if assert.True(t, ok, "valid schema %q must reach the model", name) {
					assert.JSONEq(t, external.expand(schema), string(parameters))
				}
			}
		})
	}
}

// registrySchemaExternalPeer is the target of external schema references. Discovery
// must never fetch it.
type registrySchemaExternalPeer struct{ url string }

func newRegistrySchemaExternalPeer(t *testing.T) registrySchemaExternalPeer {
	t.Helper()

	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)

		_, _ = io.WriteString(w, `{"type":"string"}`)
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { assert.Zero(t, hits.Load(), "schema validation must not fetch external references") })

	return registrySchemaExternalPeer{url: server.URL}
}

func (p registrySchemaExternalPeer) expand(schema string) string {
	return strings.ReplaceAll(schema, "{external}", p.url)
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

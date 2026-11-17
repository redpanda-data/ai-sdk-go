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
)

// Schemas that are malformed, use another dialect or need an ECMA-262 regular
// expression engine. The legacy client forwards each unchanged; so must the
// registry, leaving validation to the model provider.
var registryLooseSchemas = []struct{ name, schema string }{
	{"properties array", `{"type":"object","properties":[]}`},
	{"string property", `{"type":"object","properties":{"q":"string"}}`},
	{"null schema", `null`},
	{"numeric root type", `{"type":42}`},
	{"numeric nested type", `{"type":"object","properties":{"q":{"type":42}}}`},
	{"empty root type", `{"type":""}`},
	{"empty nested type", `{"type":"object","properties":{"q":{"type":""}}}`},
	{"string required", `{"type":"object","required":"q"}`},
	{"unknown type name", `{"type":"object","properties":{"q":{"anyOf":[{"type":"string"},{"type":"strin"}]}}}`},
	{"dangling local reference", `{"type":"object","properties":{"q":{"$ref":"#/$defs/missing"}}}`},
	{"external reference", `{"type":"object","properties":{"q":{"$ref":"{external}/schema.json"}}}`},
	// zod 4.6.5: z.toJSONSchema(z.object({q: z.hostname()})).
	{"zod hostname lookahead", `{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"q":{"type":"string","format":"hostname",` +
		`"pattern":"^(?=.{1,253}\\.?$)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[-0-9a-zA-Z]{0,61}[0-9a-zA-Z])?)*\\.?$"}},` +
		`"required":["q"],"additionalProperties":false}`},
	{"relative id", `{"$id":"input.json","type":"object","properties":{"q":{"type":"string"}}}`},
	{"draft-04 boolean exclusiveMinimum", `{"$schema":"http://json-schema.org/draft-04/schema#","type":"object",` +
		`"properties":{"q":{"type":"string"},"n":{"type":"number","minimum":0,"exclusiveMinimum":true}}}`},
}

func TestRegistryLooseSchemasReachModelAndExecute(t *testing.T) {
	t.Parallel()

	external := newRegistrySchemaExternalPeer(t)

	for _, legacy := range []bool{false, true} {
		for _, test := range registryLooseSchemas {
			t.Run(fmt.Sprintf("legacy=%t/%s", legacy, test.name), func(t *testing.T) {
				t.Parallel()

				schema := external.expand(test.schema)
				peer := newRegistryPeer(t, legacy)
				registrySchemaServeLoose(peer, schema)
				fx := legacyCompatStart(t, peer)
				remote := newContextRegistry(t, peer)

				var mu sync.Mutex
				definitions := make(map[string]json.RawMessage)
				model := fakellm.NewFakeModel()
				model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
					mu.Lock()
					defer mu.Unlock()

					for _, definition := range req.Tools {
						definitions[definition.Name] = definition.Parameters
					}

					return registryAgentRespond(
						llm.NewToolRequestPart("remote", "remote__loose", json.RawMessage(`{"q":"a.example"}`)),
						llm.NewToolRequestPart("legacy", "old__loose", json.RawMessage(`{"q":"a.example"}`)),
					)
				})
				model.When(fakellm.Any()).ThenRespondText("done")

				ag, err := llmagent.New("schema-policy", "Use the available tools.", model, llmagent.WithTools(fx.registry, remote))
				require.NoError(t, err)

				ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
				defer cancel()

				state, failures := registrySchemaPolicyRun(ctx, ag)
				assert.Empty(t, failures, "a schema the legacy client forwards must not fail the agent step")
				require.Equal(t, 2, model.CallCount(), "the model must be called with the tool and again with its result")

				mu.Lock()
				parameters, legacyParameters := definitions["remote__loose"], definitions["old__loose"]
				mu.Unlock()

				require.NotNil(t, legacyParameters, "the legacy client must offer the tool: %v", definitions)
				require.NotNil(t, parameters, "the registry must offer the tool: %v", definitions)
				assert.Equal(t, string(legacyParameters), string(parameters), "the registry must forward the bytes the legacy client forwards")
				assert.JSONEq(t, schema, string(parameters))

				calls := registryAgentReadCalls(t, fx.wire, 2)
				for _, call := range calls {
					assert.Equal(t, "loose", call.name)
					assert.JSONEq(t, `{"q":"a.example"}`, string(call.arguments))
				}

				for _, name := range []string{"remote__loose", "old__loose"} {
					results := registryAgentResults(state, name)
					if assert.Len(t, results, 1, name) {
						assert.False(t, results[0].IsError, string(results[0].Result))
						assert.Contains(t, string(results[0].Result), `loose {\"q\":\"a.example\"}`)
					}
				}
			})
		}
	}
}

// registrySchemaServeLoose serves a callable "loose" tool and lists it with
// schema, which the SDK server would refuse to register.
func registrySchemaServeLoose(peer *registryPeer, schema string) {
	alice := peer.servers["alice"]
	alice.AddTool(&sdkmcp.Tool{Name: "loose", InputSchema: json.RawMessage(`{"type":"object"}`)},
		func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
			return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "loose " + string(req.Params.Arguments)}}}, nil
		})
	alice.AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
		return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
			result, err := next(ctx, method, req)
			if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil {
				for i, listedTool := range listed.Tools {
					if listedTool.Name == "loose" {
						served := *listedTool
						served.InputSchema = json.RawMessage(schema)
						listed.Tools[i] = &served
					}
				}
			}

			return result, err
		}
	})
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

			_, failures := registrySchemaPolicyRun(ctx, ag)
			require.Empty(t, failures)
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

// registrySchemaExternalPeer is the target of external schema references. The
// registry must never fetch it.
type registrySchemaExternalPeer struct{ url string }

func newRegistrySchemaExternalPeer(t *testing.T) registrySchemaExternalPeer {
	t.Helper()

	var hits atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits.Add(1)

		_, _ = io.WriteString(w, `{"type":"string"}`)
	}))
	t.Cleanup(server.Close)
	t.Cleanup(func() { assert.Zero(t, hits.Load(), "the registry must not fetch external schema references") })

	return registrySchemaExternalPeer{url: server.URL}
}

func (p registrySchemaExternalPeer) expand(schema string) string {
	return strings.ReplaceAll(schema, "{external}", p.url)
}

func registrySchemaPolicyRun(ctx context.Context, ag *llmagent.LLMAgent) (*session.State, []error) {
	state := &session.State{ID: "schema-policy", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("hello"))}}
	var failures []error

	for _, err := range ag.Run(ctx, agent.NewInvocationMetadata(state, ag.Info())) {
		if err != nil {
			failures = append(failures, err)
		}
	}

	return state, failures
}

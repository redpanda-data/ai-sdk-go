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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

const registryParameterListMethod = "tools/list"

func registryParameterSchema(header string) json.RawMessage {
	return json.RawMessage(`{"type":"object","required":["route"],"properties":{"route":{"type":"string","x-mcp-header":"` + header + `"},"optional":{"type":"boolean","x-mcp-header":"Optional"}}}`)
}

func TestContextRegistryCapturedParameterHeaders(t *testing.T) {
	t.Parallel()

	for _, refresh := range []bool{false, true} {
		name := "invalidated SDK cache"
		if refresh {
			name = "changed SDK cache"
		}

		t.Run(name, func(t *testing.T) {
			t.Parallel()

			server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "parameter-headers", Version: "1"}, &sdkmcp.ServerOptions{HasTools: true})
			addTool := func(schema json.RawMessage) {
				server.AddTool(&sdkmcp.Tool{Name: "routed", InputSchema: schema}, func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
					return &sdkmcp.CallToolResult{}, nil
				})
			}
			addTool(registryParameterSchema("Tenant"))

			sent := registryAdvertiseToolTTL(&registryPeer{servers: map[string]*sdkmcp.Server{"alice": server}}, time.Hour, false)
			handler := sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: true, JSONResponse: true})
			type wireCall struct {
				header http.Header
				name   string
				args   json.RawMessage
			}
			calls := make(chan wireCall, 1)
			var lists atomic.Int64
			const nativeResult = `{"content":[{"type":"text","text":"routed"}],"structuredContent":{"ok":true},"_meta":{"opaque":"retained"}}`
			peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				if req.Body != nil {
					body, err := io.ReadAll(req.Body)
					if !assert.NoError(t, err) {
						http.Error(w, "read body", http.StatusBadRequest)
						return
					}

					req.Body = io.NopCloser(bytes.NewReader(body))

					var message struct {
						ID     json.RawMessage `json:"id"`
						Method string          `json:"method"`
						Params struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						} `json:"params"`
					}
					if err := json.Unmarshal(body, &message); err == nil && message.Method == registryParameterListMethod {
						lists.Add(1)
					}

					if message.Method == registryAgentCallMethod {
						calls <- wireCall{header: req.Header.Clone(), name: message.Params.Name, args: message.Params.Arguments}

						w.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(w, `{"jsonrpc":"2.0","id":`+string(message.ID)+`,"result":`+nativeResult+`}`)

						return
					}
				}

				handler.ServeHTTP(w, req)
			}))
			t.Cleanup(peer.Close)
			registry, err := NewRegistry("remote", peer.URL, WithCredentials(registryContextCredentials), WithRegistryMaxToolCacheTTL(time.Hour))
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, registry.Close()) })
			ctx := registryCaller(t.Context(), "alice", "alice:one")
			old, err := registry.Snapshot(ctx)
			require.NoError(t, err)
			waitForChange := registryToolChangeBarrier(t, registry, "alice", sent)
			changedSchema := json.RawMessage(`{"type":"object","required":["route","new_required"],"properties":{"route":{"type":"string","x-mcp-header":"New-Tenant"},"other":{"type":"string","x-mcp-header":"Tenant"},"optional":{"type":"string","x-mcp-header":"Optional"},"new_required":{"type":"string"}}}`)
			addTool(changedSchema)
			waitForChange()

			if refresh {
				current, err := registry.Snapshot(ctx)
				require.NoError(t, err)
				definitions, err := current.List(ctx)
				require.NoError(t, err)
				require.Len(t, definitions, 1)
				assert.JSONEq(t, string(changedSchema), string(definitions[0].Parameters))
			}

			definitions, err := old.List(ctx)
			require.NoError(t, err)
			require.Len(t, definitions, 1)
			assert.Contains(t, string(definitions[0].Parameters), `"Tenant"`)

			listsBeforeCall := lists.Load()
			response, err := old.Execute(ctx, &llm.ToolRequestPart{ID: "call", Name: "remote__routed", Arguments: json.RawMessage(`{"route":" tenant/\u03b1 ","other":"wrong-tenant","optional":false}`)})
			require.NoError(t, err)

			select {
			case call := <-calls:
				assert.Equal(t, "=?base64?IHRlbmFudC/OsSA=?=", call.header.Get("Mcp-Param-Tenant"))
				assert.Equal(t, "false", call.header.Get("Mcp-Param-Optional"))

				for header := range call.header {
					if strings.HasPrefix(strings.ToLower(header), "mcp-param-") {
						assert.Contains(t, []string{"mcp-param-tenant", "mcp-param-optional"}, strings.ToLower(header), "new discovery must not add headers to an old snapshot")
					}
				}

				assert.Equal(t, registryAgentCallMethod, call.header.Get("Mcp-Method"))
				assert.Equal(t, "routed", call.header.Get("Mcp-Name"))
				assert.Equal(t, "Bearer alice:one", call.header.Get("Authorization"))
				assert.Equal(t, "routed", call.name)
				assert.JSONEq(t, `{"route":" tenant/\u03b1 ","other":"wrong-tenant","optional":false}`, string(call.args))
			case <-time.After(5 * time.Second):
				t.Fatal("no tools/call reached the HTTP peer")
			}

			require.NotNil(t, response)
			assert.False(t, response.IsError)
			assert.JSONEq(t, nativeResult, string(response.Result))
			assert.Equal(t, listsBeforeCall, lists.Load(), "execution must not rediscover tools")

			const unencodable = `{"route":{"unsupported":"object"},"optional":1.5}`
			response, err = old.Execute(ctx, &llm.ToolRequestPart{ID: "unencodable", Name: "remote__routed", Arguments: json.RawMessage(unencodable)})
			require.NoError(t, err)
			require.NotNil(t, response)
			assert.False(t, response.IsError, string(response.Result))

			select {
			case call := <-calls:
				assert.Empty(t, registryParameterHeaderSubset(call.header), "like the SDK, values without a header encoding send no header")
				assert.JSONEq(t, unencodable, string(call.args), "the values stay in the arguments")
			case <-time.After(5 * time.Second):
				t.Fatal("a call with unencodable parameter values must still reach the HTTP peer")
			}
		})
	}
}

// Through an agent step over the wire: the SDK excludes a tool whose annotation it
// rejects while the rest of the catalog stays usable, and the registry mirrors valid
// annotations, including one beside a boolean subschema the SDK's parser cannot read.
func TestRegistryAgentParameterAnnotations(t *testing.T) {
	t.Parallel()

	excluded := map[string]string{
		"number_header":    `{"type":"object","properties":{"value":{"type":"number","x-mcp-header":"Value"}}}`,
		"invalid_header":   `{"type":"object","properties":{"value":{"type":"string","x-mcp-header":"Tenant\r\nInjected"}}}`,
		"duplicate_header": `{"type":"object","properties":{"one":{"type":"string","x-mcp-header":"Tenant"},"parent":{"type":"object","properties":{"two":{"type":"string","x-mcp-header":"TENANT"}}}}}`,
	}

	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, legacy)
			alice := peer.servers["alice"]
			echo := func(_ context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
				return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: string(req.Params.Arguments)}}}, nil
			}
			alice.AddTool(&sdkmcp.Tool{Name: "counted", InputSchema: json.RawMessage(
				`{"type":"object","properties":{"parent":{"type":"object","properties":{"count":{"type":"integer","x-mcp-header":"Count"}}}}}`)}, echo)
			alice.AddTool(&sdkmcp.Tool{Name: "flagged", InputSchema: json.RawMessage(
				`{"type":"object","properties":{"any":true,"route":{"type":"string","x-mcp-header":"Tenant"}}}`)}, echo)
			alice.AddReceivingMiddleware(func(next sdkmcp.MethodHandler) sdkmcp.MethodHandler {
				return func(ctx context.Context, method string, req sdkmcp.Request) (sdkmcp.Result, error) {
					result, err := next(ctx, method, req)
					if listed, ok := result.(*sdkmcp.ListToolsResult); ok && err == nil {
						for name, schema := range excluded {
							listed.Tools = append(listed.Tools, &sdkmcp.Tool{Name: name, InputSchema: json.RawMessage(schema)})
						}
					}

					return result, err
				}
			})

			calls := registryAgentCaptureCalls(peer)
			remote := newContextRegistry(t, peer)

			model := fakellm.NewFakeModel()
			model.When(fakellm.FirstCall()).ThenRespondWith(func(req *llm.Request, _ *fakellm.CallContext) (*llm.Response, error) {
				names := registryNames(req.Tools)
				assert.Subset(t, names, []string{"remote__counted", "remote__flagged", "remote__alice_only"})

				for name := range excluded {
					assert.NotContains(t, names, "remote__"+name, "the SDK excludes a tool with an invalid annotation")
				}

				return registryAgentRespond(
					llm.NewToolRequestPart("counted", "remote__counted", json.RawMessage(`{"parent":{"count":42}}`)),
					llm.NewToolRequestPart("flagged", "remote__flagged", json.RawMessage(`{"route":" tenant/\u03b1 "}`)),
					llm.NewToolRequestPart("only", "remote__alice_only", json.RawMessage(`{"old":"x"}`)),
				)
			})
			model.When(fakellm.Any()).ThenRespondText("done")

			ag, err := llmagent.New("parameter-headers", "Use the available tools.", model, llmagent.WithTools(remote))
			require.NoError(t, err)

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 10*time.Second)
			defer cancel()

			state := &session.State{ID: "parameter-headers", Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("call all"))}}
			registryAgentRun(ctx, t, ag, state)
			require.Equal(t, 2, model.CallCount())

			want := map[string]http.Header{
				"counted":    {"Mcp-Param-Count": {"42"}},
				"flagged":    {"Mcp-Param-Tenant": {"=?base64?IHRlbmFudC/OsSA=?="}},
				"alice_only": {},
			}
			if legacy {
				want = map[string]http.Header{"counted": {}, "flagged": {}, "alice_only": {}}
			}

			for _, call := range registryAgentReadCalls(t, calls, 3) {
				assert.Equal(t, want[call.name], registryParameterHeaderSubset(call.headers), call.name)
			}

			for _, name := range []string{"remote__counted", "remote__flagged", "remote__alice_only"} {
				results := registryAgentResults(state, name)
				if assert.Len(t, results, 1, name) {
					assert.False(t, results[0].IsError, string(results[0].Result))
				}
			}
		})
	}
}

// A union sibling makes the SDK's string-typed annotation parser fail, so the SDK
// neither excludes the tool nor sends any parameter header for it. The registry
// still binds the valid annotation and ignores those it cannot honor.
func TestRegistrySnapshotIgnoresUnhonorableParameterAnnotations(t *testing.T) {
	t.Parallel()

	route := http.Header{"Mcp-Param-Route": {"r"}}

	for _, test := range []struct {
		name     string
		property string
		value    string
		want     http.Header
	}{
		{"unsupported type", `{"type":"number","x-mcp-header":"Value"}`, `7`, route},
		{"nonstring annotation", `{"type":"string","x-mcp-header":true}`, `"v"`, route},
		{"empty annotation", `{"type":"string","x-mcp-header":""}`, `"v"`, route},
		{"invalid header name", `{"type":"string","x-mcp-header":"Tenant\r\nInjected"}`, `"v"`, route},
		{"duplicate header", `{"type":"string","x-mcp-header":"ROUTE"}`, `"v"`, http.Header{}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			peer := newRegistryPeer(t, false)
			schema := `{"type":"object","properties":{"nullable":{"type":["string","null"]},"route":{"type":"string","x-mcp-header":"Route"},"value":` + test.property + `}}`
			peer.servers["alice"].AddTool(&sdkmcp.Tool{Name: "annotated", InputSchema: json.RawMessage(schema)},
				func(context.Context, *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
					return &sdkmcp.CallToolResult{}, nil
				})
			calls := registryAgentCaptureCalls(peer)
			registry := newContextRegistry(t, peer)

			ctx, cancel := context.WithTimeout(registryCaller(t.Context(), "alice", "alice:one"), 5*time.Second)
			defer cancel()

			snapshot, err := registry.Snapshot(ctx)
			require.NoError(t, err, "an annotation the registry cannot honor must not fail the snapshot")
			annotated, err := snapshot.Get(ctx, "remote__annotated")
			require.NoError(t, err)
			assert.JSONEq(t, schema, string(annotated.Definition().Parameters))

			arguments := `{"route":"r","value":` + test.value + `}`
			response, err := snapshot.Execute(ctx, &llm.ToolRequestPart{ID: "call", Name: "remote__annotated", Arguments: json.RawMessage(arguments)})
			require.NoError(t, err)
			assert.False(t, response.IsError, string(response.Result))

			call := registryAgentReadCalls(t, calls, 1)[0]
			assert.Equal(t, test.want, registryParameterHeaderSubset(call.headers))
			assert.JSONEq(t, arguments, string(call.arguments))
		})
	}
}

func TestRegistryParameterBindings(t *testing.T) {
	t.Parallel()

	tenant := []registryParameterBinding{{path: []string{"route"}, header: "Mcp-Param-Tenant"}}
	nested := []registryParameterBinding{{path: []string{"parent", "route"}, header: "Mcp-Param-Tenant"}}
	primitives := []registryParameterBinding{{path: []string{"count"}, header: "Mcp-Param-Count"}, {path: []string{"flag"}, header: "Mcp-Param-Flag"}, tenant[0]}
	annotated := `{"type":"string","x-mcp-header":"Tenant"}`

	for _, test := range []struct {
		name   string
		schema string
		want   []registryParameterBinding
	}{
		{"nested", `{"properties":{"parent":{"type":"object","properties":{"route":` + annotated + `}}}}`, nested},
		{"primitive types", `{"properties":{"count":{"type":"integer","x-mcp-header":"Count"},"flag":{"type":"boolean","x-mcp-header":"Flag"},"route":` + annotated + `}}`, primitives},
		{"boolean siblings", `{"properties":{"any":true,"never":false,"route":` + annotated + `}}`, tenant},
		{"union sibling", `{"properties":{"nullable":{"type":["string","null"]},"route":` + annotated + `}}`, tenant},
		{"malformed siblings", `{"type":42,"required":"q","properties":{"q":"string","n":42,"l":[],"x":null,"p":{"properties":[]},"o":{"properties":"route"},"route":` + annotated + `}}`, tenant},
		{"annotated object still walked", `{"properties":{"parent":{"type":"object","x-mcp-header":"Parent","properties":{"route":` + annotated + `}}}}`, nested},
		{"duplicate nested", `{"properties":{"one":{"type":"string","x-mcp-header":"TENANT"},"parent":{"properties":{"two":{"type":"boolean","x-mcp-header":"tenant"}}},"route":` + annotated + `}}`, nil},
		{"duplicate beside valid", `{"properties":{"one":{"type":"string","x-mcp-header":"ONE"},"two":{"type":"string","x-mcp-header":"one"},"route":` + annotated + `}}`, tenant},
		{"properties array", `{"properties":[` + annotated + `]}`, nil},
		{"null schema", `null`, nil},
		{"boolean schema", `true`, nil},
		{"array schema", `[{"properties":{"route":` + annotated + `}}]`, nil},
		{"string schema", `"object"`, nil},
		{"number", `{"properties":{"value":{"type":"number","x-mcp-header":"Tenant"}}}`, nil},
		{"object", `{"properties":{"value":{"type":"object","x-mcp-header":"Tenant"}}}`, nil},
		{"array", `{"properties":{"value":{"type":"array","x-mcp-header":"Tenant"}}}`, nil},
		{"union", `{"properties":{"value":{"type":["string","null"],"x-mcp-header":"Tenant"}}}`, nil},
		{"numeric type", `{"properties":{"value":{"type":42,"x-mcp-header":"Tenant"}}}`, nil},
		{"missing type", `{"properties":{"value":{"x-mcp-header":"Tenant"}}}`, nil},
		{"null annotation", `{"properties":{"value":{"type":"string","x-mcp-header":null}}}`, nil},
		{"empty annotation", `{"properties":{"value":{"type":"string","x-mcp-header":""}}}`, nil},
		{"nonstring annotation", `{"properties":{"value":{"type":"string","x-mcp-header":true}}}`, nil},
		{"invalid name", `{"properties":{"value":{"type":"string","x-mcp-header":"Tenant\r\nInjected"}}}`, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			bindings := registryFlattenParameterTrie(registryParameterTrie(registryDecodedSchema(t, test.schema)))

			if test.want == nil {
				assert.Empty(t, bindings)
			} else {
				assert.Equal(t, test.want, bindings)
			}
		})
	}
}

// The SDK caps tools/list nesting at 1000 levels; extraction must not depend on it.
func TestRegistryParameterBindingsDeepSchema(t *testing.T) {
	t.Parallel()

	const depth = 9000

	var (
		schema    any = map[string]any{"type": "string", "x-mcp-header": "Deep"}
		arguments any = "deep"
	)

	path := make([]string, depth)
	for i := depth - 1; i >= 0; i-- {
		path[i] = "p" + strconv.Itoa(i)
		schema = map[string]any{"type": "object", "properties": map[string]any{path[i]: schema, "any": true, "loose": "string"}}
		arguments = map[string]any{path[i]: arguments}
	}

	trie := registryParameterTrie(schema)
	require.Equal(t, []registryParameterBinding{{path: path, header: "Mcp-Param-Deep"}}, registryFlattenParameterTrie(trie))
	assert.Equal(t, depth, registryRetainedParameterElements(trie))

	object, ok := arguments.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, http.Header{"Mcp-Param-Deep": {"deep"}}, registryParameterHeaders(trie, object))
}

// Under go-sdk's 1000-level JSON cap a server can nest hundreds of properties
// levels and end them in many annotated leaves. Capture must retain one element
// per property on an annotated path, not one copy of the path per annotation.
func TestRegistryParameterTrieIsLinearInSchema(t *testing.T) {
	t.Parallel()

	const depth, leaves = 400, 2000

	schema, arguments := registryParameterAmplificationSchema(depth, leaves)
	trie := registryParameterTrie(schema)
	require.Equal(t, depth+leaves, registryRetainedParameterElements(trie), "retained elements must equal the properties on annotated paths")

	headers := registryParameterHeaders(trie, arguments)
	assert.Len(t, headers, leaves)
	assert.Equal(t, "v0", headers.Get("Mcp-Param-H0"))
	assert.Equal(t, "v1999", headers.Get("Mcp-Param-H1999"))

	bottom := arguments
	for i := range depth {
		bottom, _ = bottom["p"+strconv.Itoa(i)].(map[string]any)
	}

	require.NotNil(t, bottom)
	clear(bottom)
	bottom["l7"] = "v7"
	bottom["l8"] = map[string]any{"type": "string"}

	assert.Equal(t, http.Header{"Mcp-Param-H7": {"v7"}}, registryParameterHeaders(trie, arguments), "only present, encodable leaves send headers")

	arguments["p0"] = "not an object"
	assert.Empty(t, registryParameterHeaders(trie, arguments))
}

// registryParameterAmplificationSchema nests depth properties objects, each with
// unannotated siblings, above leaves uniquely annotated properties, and returns
// arguments that set every leaf.
func registryParameterAmplificationSchema(depth, leaves int) (any, map[string]any) {
	properties := map[string]any{"plain": map[string]any{"type": "string"}}
	arguments := map[string]any{"plain": "x"}

	for i := range leaves {
		name := "l" + strconv.Itoa(i)
		properties[name] = map[string]any{"type": "string", "x-mcp-header": "H" + strconv.Itoa(i)}
		arguments[name] = "v" + strconv.Itoa(i)
	}

	for i := depth - 1; i >= 0; i-- {
		name := "p" + strconv.Itoa(i)
		properties = map[string]any{
			name:    map[string]any{"type": "object", "properties": properties},
			"noise": map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "string"}}},
			"flag":  true,
		}
		arguments = map[string]any{name: arguments, "noise": map[string]any{"a": "x"}}
	}

	return map[string]any{"type": "object", "properties": properties}, arguments
}

func TestRegistryParameterHeaderEncoding(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name  string
		value string
		want  string
		ok    bool
	}{
		{"ASCII", `"tenant/path"`, "tenant/path", true},
		{"empty", `""`, "", true},
		{"Unicode", `"α"`, "=?base64?zrE=?=", true},
		{"leading space", `" a"`, "=?base64?IGE=?=", true},
		{"trailing space", `"a "`, "=?base64?YSA=?=", true},
		{"control", `"a\nb"`, "=?base64?YQpi?=", true},
		{"sentinel", `"=?base64?YQ==?="`, "=?base64?PT9iYXNlNjQ/WVE9PT89?=", true},
		{"true", `true`, "true", true},
		{"false", `false`, "false", true},
		{"zero", `0`, "0", true},
		{"negative zero", `-0`, "0", true},
		{"negative integer", `-42`, "-42", true},
		{"integer exponent", `1e3`, "1000", true},
		{"largest safe integer", `9007199254740991`, "9007199254740991", true},
		{"smallest safe integer", `-9007199254740991`, "-9007199254740991", true},
		{"unsafe integer", `9007199254740992`, "", false},
		{"unsafe negative integer", `-9007199254740992`, "", false},
		{"fraction", `1.5`, "", false},
		{"null", `null`, "", false},
		{"object", `{}`, "", false},
		{"array", `[]`, "", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var value any
			require.NoError(t, json.Unmarshal(json.RawMessage(test.value), &value))

			encoded, ok := registryEncodeParameterHeader(value)
			assert.Equal(t, test.ok, ok)
			assert.Equal(t, test.want, encoded)
		})
	}
}

// Like the SDK, a value with no SEP-2243 encoding sends no header; the call is
// not rejected locally and the value stays in the arguments.
func TestRegistryParameterHeadersSkipUnencodableValues(t *testing.T) {
	t.Parallel()

	trie := registryParameterTrie(registryDecodedSchema(t,
		`{"properties":{"route":{"type":"string","x-mcp-header":"Tenant"},"optional":{"type":"boolean","x-mcp-header":"Optional"},`+
			`"parent":{"properties":{"count":{"type":"integer","x-mcp-header":"Count"}}}}}`))
	require.Len(t, registryFlattenParameterTrie(trie), 3)

	for _, test := range []struct {
		name      string
		arguments string
		want      http.Header
	}{
		{"absent", `{}`, http.Header{}},
		{"null", `{"route":null,"optional":null,"parent":{"count":null}}`, http.Header{}},
		{"null parent", `{"parent":null}`, http.Header{}},
		{"nonobject parent", `{"parent":"count"}`, http.Header{}},
		{"objects and arrays", `{"route":{"a":"b"},"optional":[true],"parent":{"count":{}}}`, http.Header{}},
		{"unsafe numbers", `{"route":1.5,"optional":9007199254740992,"parent":{"count":-9007199254740992}}`, http.Header{}},
		{"mixed", `{"route":{"a":"b"},"optional":false,"parent":{"count":7}}`, http.Header{"Mcp-Param-Optional": {"false"}, "Mcp-Param-Count": {"7"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var arguments, original map[string]any
			require.NoError(t, json.Unmarshal([]byte(test.arguments), &arguments))
			require.NoError(t, json.Unmarshal([]byte(test.arguments), &original))

			assert.Equal(t, test.want, registryParameterHeaders(trie, arguments))
			assert.Equal(t, original, arguments, "header encoding must not change the arguments")
		})
	}
}

// registryParameterBinding is one flattened trie entry, for assertions.
type registryParameterBinding struct {
	path   []string
	header string
}

// registryFlattenParameterTrie lists the trie's bindings sorted by header.
func registryFlattenParameterTrie(trie *registryParameterNode) []registryParameterBinding {
	var (
		bindings []registryParameterBinding
		walk     func(*registryParameterNode, []string)
	)

	walk = func(node *registryParameterNode, path []string) {
		for name, child := range node.children {
			path := append(path, name)
			if child.header != "" {
				bindings = append(bindings, registryParameterBinding{path: slices.Clone(path), header: child.header})
			}

			walk(child, path)
		}
	}

	if trie != nil {
		walk(trie, nil)
	}

	slices.SortFunc(bindings, func(a, b registryParameterBinding) int { return strings.Compare(a.header, b.header) })

	return bindings
}

// registryRetainedParameterElements counts the trie's property nodes.
func registryRetainedParameterElements(trie *registryParameterNode) int {
	if trie == nil {
		return 0
	}

	retained := len(trie.children)
	for _, child := range trie.children {
		retained += registryRetainedParameterElements(child)
	}

	return retained
}

func registryDecodedSchema(t *testing.T, schema string) any {
	t.Helper()

	var decoded any
	require.NoError(t, json.Unmarshal([]byte(schema), &decoded))

	return decoded
}

func registryParameterHeaderSubset(header http.Header) http.Header {
	parameters := make(http.Header)

	for name, values := range header {
		if strings.HasPrefix(strings.ToLower(name), "mcp-param-") {
			parameters[name] = values
		}
	}

	return parameters
}

func TestRegistryParameterTransportOverridesOnlyModernCalls(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name     string
		protocol string
		method   string
		captured http.Header
		want     http.Header
	}{
		{"modern", "2026-07-28", registryAgentCallMethod, http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"captured"}}},
		{"modern unannotated snapshot", "2026-07-28", registryAgentCallMethod, http.Header{}, http.Header{}},
		{"legacy", "2025-06-18", registryAgentCallMethod, http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
		{"discovery", "2026-07-28", registryParameterListMethod, http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
		{"no capture", "2026-07-28", registryAgentCallMethod, nil, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				parameters := make(http.Header)

				for name, values := range req.Header {
					if strings.HasPrefix(strings.ToLower(name), "mcp-param-") {
						parameters[name] = values
					}
				}

				assert.Equal(t, test.want, parameters)
				w.WriteHeader(http.StatusNoContent)
			}))
			t.Cleanup(peer.Close)
			registry, err := NewRegistry("remote", peer.URL, WithCredentials(registryContextCredentials))
			require.NoError(t, err)
			t.Cleanup(func() { assert.NoError(t, registry.Close()) })

			ctx := registryCaller(t.Context(), "alice", "alice:one")
			if test.captured != nil {
				ctx = context.WithValue(ctx, registryParameterHeadersKey{}, test.captured)
			}

			req, err := http.NewRequestWithContext(ctx, http.MethodPost, peer.URL, nil)
			require.NoError(t, err)
			req.Header.Set("Mcp-Protocol-Version", test.protocol)
			req.Header.Set("Mcp-Method", test.method)
			req.Header.Set("Mcp-Param-Tenant", "SDK")
			req.Header.Set("Mcp-Param-New", "SDK")
			original := req.Header.Clone()
			response, err := registry.httpClient("alice").Do(req)
			require.NoError(t, err)
			require.NoError(t, response.Body.Close())
			assert.Equal(t, original, req.Header, "transport must not mutate SDK requests")
		})
	}
}

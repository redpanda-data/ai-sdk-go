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
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
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

					if message.Method == "tools/call" {
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

				assert.Equal(t, "tools/call", call.header.Get("Mcp-Method"))
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

			invalid, err := old.Execute(ctx, &llm.ToolRequestPart{ID: "invalid", Name: "remote__routed", Arguments: json.RawMessage(`{"route":{"unsupported":"object"}}`)})
			require.NoError(t, err)
			require.NotNil(t, invalid)
			assert.True(t, invalid.IsError)
			assert.Contains(t, string(invalid.Result), "expected a string, boolean or safe integer")
			assert.Empty(t, calls, "unsupported parameter values must fail before HTTP execution")
		})
	}
}

func TestRegistryParameterBindings(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name   string
		schema string
		valid  bool
	}{
		{"nested", `{"properties":{"parent":{"type":"object","properties":{"route":{"type":"string","x-mcp-header":"Tenant"}}}}}`, true},
		{"unannotated union", `{"properties":{"value":{"type":["string","null"]}}}`, true},
		{"boolean property schemas", `{"properties":{"allowed":true,"forbidden":false}}`, true},
		{"number", `{"properties":{"value":{"type":"number","x-mcp-header":"Tenant"}}}`, false},
		{"object", `{"properties":{"value":{"type":"object","x-mcp-header":"Tenant"}}}`, false},
		{"array", `{"properties":{"value":{"type":"array","x-mcp-header":"Tenant"}}}`, false},
		{"union", `{"properties":{"value":{"type":["string","null"],"x-mcp-header":"Tenant"}}}`, false},
		{"missing type", `{"properties":{"value":{"x-mcp-header":"Tenant"}}}`, false},
		{"null annotation", `{"properties":{"value":{"type":"string","x-mcp-header":null}}}`, false},
		{"empty annotation", `{"properties":{"value":{"type":"string","x-mcp-header":""}}}`, false},
		{"nonstring annotation", `{"properties":{"value":{"type":"string","x-mcp-header":true}}}`, false},
		{"invalid name", `{"properties":{"value":{"type":"string","x-mcp-header":"Tenant\r\nInjected"}}}`, false},
		{"duplicate nested", `{"properties":{"one":{"type":"string","x-mcp-header":"TENANT"},"parent":{"properties":{"two":{"type":"boolean","x-mcp-header":"tenant"}}}}}`, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			_, err := registryParameterBindings(json.RawMessage(test.schema))
			if test.valid {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
			}
		})
	}

	bindings, err := registryParameterBindings(json.RawMessage(`{"properties":{"parent":{"properties":{"route":{"type":"string","x-mcp-header":"Tenant"}}}}}`))
	require.NoError(t, err)
	headers, err := registryParameterHeaders(bindings, map[string]any{"parent": map[string]any{"route": "nested"}})
	require.NoError(t, err)
	assert.Equal(t, http.Header{"Mcp-Param-Tenant": {"nested"}}, headers)
}

func TestRegistryParameterHeaderEncoding(t *testing.T) {
	t.Parallel()

	for _, test := range []struct {
		name  string
		value string
		want  string
		valid bool
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
		{"object", `{}`, "", false},
		{"array", `[]`, "", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			var value any
			require.NoError(t, json.Unmarshal(json.RawMessage(test.value), &value))

			encoded, err := registryEncodeParameterHeader(value)
			if test.valid {
				require.NoError(t, err)
				assert.Equal(t, test.want, encoded)
			} else {
				require.Error(t, err)
			}
		})
	}

	bindings, err := registryParameterBindings(registryParameterSchema("Tenant"))
	require.NoError(t, err)

	for _, args := range []map[string]any{{}, {"route": nil}, {"optional": nil}} {
		headers, err := registryParameterHeaders(bindings, args)
		require.NoError(t, err)
		assert.Empty(t, headers, "absent and null parameters must not emit headers")
	}
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
		{"modern", "2026-07-28", "tools/call", http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"captured"}}},
		{"modern unannotated snapshot", "2026-07-28", "tools/call", http.Header{}, http.Header{}},
		{"legacy", "2025-06-18", "tools/call", http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
		{"discovery", "2026-07-28", registryParameterListMethod, http.Header{"Mcp-Param-Tenant": {"captured"}}, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
		{"no capture", "2026-07-28", "tools/call", nil, http.Header{"Mcp-Param-Tenant": {"SDK"}, "Mcp-Param-New": {"SDK"}}},
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

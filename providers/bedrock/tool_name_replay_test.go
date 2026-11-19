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

package bedrock

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/internal/toolname"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

//go:embed testdata/session_invalid_tool_name.json
var sessionInvalidToolNameJSON []byte

var wireToolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

const validToolName = "inventory__get_item"

func longToolName() string { return strings.Repeat("a", 30) + "__" + strings.Repeat("b", 33) }

func newMapper(t *testing.T) *RequestMapper {
	t.Helper()

	m, ok := newTestModel(t, ModelClaudeSonnet55Global, WithAWSConfig(aws.Config{Region: "us-east-1"})).(*Model)
	require.True(t, ok)

	return m.requestMapper
}

func TestMapAssistantMessage_InvalidToolNamesAreReplayable(t *testing.T) {
	t.Parallel()

	long := longToolName()

	tests := []struct {
		name      string
		toolName  string
		tools     []llm.ToolDefinition
		wantExact bool
	}{
		{name: "empty", toolName: ""},
		{name: "inner space", toolName: "a b"},
		{name: "trailing space", toolName: "name "},
		{name: "dot", toolName: "functions.foo"},
		{name: "65 chars all valid", toolName: long},
		{
			name:     "65 chars with a 64-char tool registered",
			toolName: long,
			tools: []llm.ToolDefinition{{
				Name: strings.Repeat("c", 64), Description: "d", Parameters: json.RawMessage(`{"type":"object"}`),
			}},
		},
		{name: "valid name is unchanged", toolName: validToolName, wantExact: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rm := newMapper(t)

			req := &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("go")),
					llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call-1", tt.toolName, json.RawMessage(`{}`))),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart("call-1", tt.toolName,
						json.RawMessage(`{"error":"tool not found"}`), true)),
				},
				Tools: tt.tools,
			}

			conv, err := rm.ToConverseInput(req)
			require.NoError(t, err)

			stream, err := rm.ToConverseStreamInput(req)
			require.NoError(t, err)

			again, err := rm.ToConverseInput(req)
			require.NoError(t, err)

			for label, msgs := range map[string][]types.Message{"converse": conv.Messages, "stream": stream.Messages} {
				use := onlyToolUse(t, msgs[1])

				assert.Regexp(t, wireToolName, aws.ToString(use.Name), label)

				if !tt.wantExact {
					assert.True(t, strings.HasPrefix(aws.ToString(use.Name), "invalid_"), "%s: replay prefix", label)
				}

				assert.Equal(t, "call-1", aws.ToString(use.ToolUseId), label)

				result := onlyToolResult(t, msgs[2])
				assert.Equal(t, "call-1", aws.ToString(result.ToolUseId), label)
				assert.Equal(t, types.ToolResultStatusError, result.Status, label)
				require.Len(t, result.Content, 1)
				text, ok := result.Content[0].(*types.ToolResultContentBlockMemberText)
				require.True(t, ok)
				assert.JSONEq(t, `{"error":"tool not found"}`, text.Value, label)

				if tt.wantExact {
					assert.Equal(t, tt.toolName, aws.ToString(use.Name), label)
				}

				for _, def := range tt.tools {
					assert.NotEqual(t, def.Name, aws.ToString(use.Name), "%s: replay name must not claim a registered tool", label)
				}
			}

			assert.Equal(t, aws.ToString(onlyToolUse(t, conv.Messages[1]).Name), aws.ToString(onlyToolUse(t, again.Messages[1]).Name),
				"mapping twice yields the same name")
			assert.Equal(t, aws.ToString(onlyToolUse(t, conv.Messages[1]).Name), aws.ToString(onlyToolUse(t, stream.Messages[1]).Name),
				"converse and stream agree")

			if len(tt.tools) > 0 {
				require.NotNil(t, conv.ToolConfig)
				spec, ok := conv.ToolConfig.Tools[0].(*types.ToolMemberToolSpec)
				require.True(t, ok)
				assert.Equal(t, tt.tools[0].Name, aws.ToString(spec.Value.Name), "tool definitions stay verbatim")
			}
		})
	}
}

func onlyToolUse(t *testing.T, msg types.Message) types.ToolUseBlock {
	t.Helper()

	for _, c := range msg.Content {
		if tu, ok := c.(*types.ContentBlockMemberToolUse); ok {
			return tu.Value
		}
	}

	require.FailNow(t, "no toolUse block")

	return types.ToolUseBlock{}
}

func onlyToolResult(t *testing.T, msg types.Message) types.ToolResultBlock {
	t.Helper()

	for _, c := range msg.Content {
		if tr, ok := c.(*types.ContentBlockMemberToolResult); ok {
			return tr.Value
		}
	}

	require.FailNow(t, "no toolResult block")

	return types.ToolResultBlock{}
}

// A registered tool whose name equals the replay name of an invalid call must
// not be impersonated by the replayed call.
func TestMapAssistantMessage_ReplayAvoidsRegisteredTool(t *testing.T) {
	t.Parallel()

	rm := newMapper(t)
	bad := "name "

	req := &llm.Request{
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("go")),
			llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart("call-1", bad, json.RawMessage(`{}`))),
		},
		Tools: []llm.ToolDefinition{{Name: toolname.Replay(bad), Description: "d", Parameters: json.RawMessage(`{"type":"object"}`)}},
	}

	in, err := rm.ToConverseInput(req)
	require.NoError(t, err)

	got := aws.ToString(onlyToolUse(t, in.Messages[1]).Name)
	assert.NotEqual(t, toolname.Replay(bad), got)
	assert.Regexp(t, wireToolName, got)
}

// bedrockPeer is a ConverseStream endpoint that rejects, like Bedrock, any toolUse
// whose name is outside the allowed grammar.
type bedrockPeer struct {
	srv *httptest.Server

	mu       sync.Mutex
	replies  [][]byte
	requests []converseRequest
	rejected int
}

// The field names are the Converse wire names.
//
//nolint:tagliatelle // Converse wire names
type converseRequest struct {
	Messages []struct {
		Role    string `json:"role"`
		Content []struct {
			ToolUse *struct {
				Name      string `json:"name"`
				ToolUseID string `json:"toolUseId"`
			} `json:"toolUse"`
			ToolResult *struct {
				ToolUseID string `json:"toolUseId"`
				Status    string `json:"status"`
			} `json:"toolResult"`
		} `json:"content"`
	} `json:"messages"`
}

func newBedrockPeer(t *testing.T, replies ...[]byte) *bedrockPeer {
	t.Helper()

	p := &bedrockPeer{replies: replies}
	p.srv = httptest.NewServer(http.HandlerFunc(p.serve))
	t.Cleanup(p.srv.Close)

	return p
}

func (p *bedrockPeer) serve(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)

	var req converseRequest
	_ = json.Unmarshal(body, &req)

	p.mu.Lock()
	defer p.mu.Unlock()

	p.requests = append(p.requests, req)

	w.Header().Set("Content-Type", "application/json")

	for mi, m := range req.Messages {
		for ci, c := range m.Content {
			if c.ToolUse != nil && !wireToolName.MatchString(c.ToolUse.Name) {
				p.rejected++

				w.Header().Set("X-Amzn-Errortype", "ValidationException")
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"message":"The model returned the following errors: messages.` +
					strconv.Itoa(mi) + `.content.` + strconv.Itoa(ci) + `.tool_use.name: String should match pattern '^[a-zA-Z0-9_-]{1,64}$'"}`))

				return
			}
		}
	}

	reply := p.replies[min(len(p.requests)-1, len(p.replies)-1)]

	if strings.HasSuffix(r.URL.Path, "/converse-stream") {
		w.Header().Set("Content-Type", "application/vnd.amazon.eventstream")
		_, _ = w.Write(reply)

		return
	}

	http.Error(w, "unexpected non-streaming request", http.StatusInternalServerError)
}

type streamEvent struct{ name, payload string }

func encodeStream(t *testing.T, events ...streamEvent) []byte {
	t.Helper()

	var buf bytes.Buffer

	enc := eventstream.NewEncoder()

	for _, e := range events {
		headers := eventstream.Headers{}
		headers.Set(":message-type", eventstream.StringValue("event"))
		headers.Set(":event-type", eventstream.StringValue(e.name))
		headers.Set(":content-type", eventstream.StringValue("application/json"))
		require.NoError(t, enc.Encode(&buf, eventstream.Message{Headers: headers, Payload: []byte(e.payload)}))
	}

	return buf.Bytes()
}

const usageEvent = `{"usage":{"inputTokens":5,"outputTokens":5,"totalTokens":10},"metrics":{"latencyMs":1}}`

func toolUseReply(t *testing.T, name string) []byte {
	t.Helper()

	return encodeStream(t,
		streamEvent{"messageStart", `{"role":"assistant"}`},
		streamEvent{"contentBlockStart", `{"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call-0001","name":` + jsonString(name) + `}}}`},
		streamEvent{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"toolUse":{"input":"{\"item\":\"x-1\"}"}}}`},
		streamEvent{"contentBlockStop", `{"contentBlockIndex":0}`},
		streamEvent{"messageStop", `{"stopReason":"tool_use"}`},
		streamEvent{"metadata", usageEvent},
	)
}

func textReply(t *testing.T) []byte {
	t.Helper()

	return encodeStream(t,
		streamEvent{"messageStart", `{"role":"assistant"}`},
		streamEvent{"contentBlockDelta", `{"contentBlockIndex":0,"delta":{"text":"Done."}}`},
		streamEvent{"contentBlockStop", `{"contentBlockIndex":0}`},
		streamEvent{"messageStop", `{"stopReason":"end_turn"}`},
		streamEvent{"metadata", usageEvent},
	)
}

// jsonString quotes an ASCII test name as a JSON string.
func jsonString(s string) string { return strconv.Quote(s) }

func newAgentOnPeer(t *testing.T, peer *bedrockPeer, registered string) agent.Agent {
	t.Helper()

	model := newTestModel(t, ModelClaudeSonnet55Global, WithAWSConfig(testAWSConfig(peer.srv.URL)))

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(&echoTool{name: registered}))

	ag, err := llmagent.New("test-agent", "You are a helpful assistant.", model, llmagent.WithTools(registry))
	require.NoError(t, err)

	return ag
}

type echoTool struct{ name string }

func (e *echoTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{Name: e.name, Description: "Looks up an item.", Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}
}

func (*echoTool) Execute(context.Context, json.RawMessage) (json.RawMessage, error) {
	return json.RawMessage(`{"ok":true}`), nil
}

func runAgent(t *testing.T, ag agent.Agent, sess *session.State) agent.InvocationEndEvent {
	t.Helper()

	var end *agent.InvocationEndEvent

	for evt, err := range ag.Run(context.Background(), agent.NewInvocationMetadata(sess, agent.Info{})) {
		require.NoError(t, err)

		if e, ok := evt.(agent.InvocationEndEvent); ok {
			end = &e
		}
	}

	require.NotNil(t, end)

	return *end
}

func TestAgent_InvalidToolNameDoesNotBlockNextTurn(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		toolName string
	}{
		{name: "trailing space", toolName: validToolName + " "},
		{name: "65 chars all valid", toolName: longToolName()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			peer := newBedrockPeer(t, toolUseReply(t, tt.toolName), textReply(t))
			ag := newAgentOnPeer(t, peer, validToolName)

			sess := &session.State{
				ID:       "sess-1",
				Messages: []llm.Message{llm.NewMessage(llm.RoleUser, llm.NewTextPart("Look up item x-1."))},
			}

			end := runAgent(t, ag, sess)
			assert.Equal(t, agent.FinishReasonStop, end.FinishReason)

			peer.mu.Lock()
			defer peer.mu.Unlock()

			assert.Zero(t, peer.rejected, "the peer must accept every request")
			require.Len(t, peer.requests, 2)

			// The session keeps the original name and the tool-not-found result.
			var calls []*llm.ToolRequestPart

			var responses []*llm.ToolResponsePart

			for _, m := range sess.Messages {
				calls = append(calls, m.ToolRequests()...)
				responses = append(responses, m.ToolResponses()...)
			}

			require.Len(t, calls, 1)
			require.Len(t, responses, 1)
			assert.Equal(t, tt.toolName, calls[0].Name)
			assert.Equal(t, "call-0001", calls[0].ID)
			assert.Equal(t, calls[0].ID, responses[0].ID)
			assert.True(t, responses[0].IsError)
			assert.Contains(t, string(responses[0].Result), "tool not found")
			assert.Contains(t, string(responses[0].Result), jsonStringBody(tt.toolName))

			// The second request carries the safe form, paired by id.
			var use, result bool

			for _, m := range peer.requests[1].Messages {
				for _, c := range m.Content {
					if c.ToolUse != nil {
						use = true

						assert.Equal(t, "call-0001", c.ToolUse.ToolUseID)
						assert.Regexp(t, wireToolName, c.ToolUse.Name)
					}

					if c.ToolResult != nil {
						result = true

						assert.Equal(t, "call-0001", c.ToolResult.ToolUseID)
						assert.Equal(t, "error", c.ToolResult.Status)
					}
				}
			}

			assert.True(t, use && result)
		})
	}
}

// jsonStringBody quotes name the way the registry error does, then escapes it
// for embedding in a JSON string.
func jsonStringBody(name string) string {
	q := jsonString(`"` + name + `"`)

	return strings.Trim(q, `"`)
}

func TestReplay_PoisonedStoredSession(t *testing.T) {
	t.Parallel()

	var pb llmpb.SessionState
	require.NoError(t, protojson.Unmarshal(sessionInvalidToolNameJSON, &pb))

	sess, err := kvstore.FromProtoSessionState(&pb)
	require.NoError(t, err)

	stored := sess.Messages[1].ToolRequests()
	require.Len(t, stored, 1)
	require.Equal(t, validToolName+" ", stored[0].Name, "fixture holds the unsafe name")

	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("Try again.")))

	peer := newBedrockPeer(t, textReply(t))
	ag := newAgentOnPeer(t, peer, validToolName)

	end := runAgent(t, ag, sess)
	assert.Equal(t, agent.FinishReasonStop, end.FinishReason)

	peer.mu.Lock()
	defer peer.mu.Unlock()

	assert.Zero(t, peer.rejected)
	require.Len(t, peer.requests, 1)

	var sawUse bool

	for _, m := range peer.requests[0].Messages {
		for _, c := range m.Content {
			if c.ToolUse != nil {
				sawUse = true

				assert.Regexp(t, wireToolName, c.ToolUse.Name)
				assert.NotEqual(t, validToolName+" ", c.ToolUse.Name)
			}
		}
	}

	assert.True(t, sawUse)
	assert.Equal(t, validToolName+" ", sess.Messages[1].ToolRequests()[0].Name, "the session keeps the original name")
}

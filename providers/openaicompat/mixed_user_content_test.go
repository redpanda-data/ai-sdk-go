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

package openaicompat_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
)

func TestToProviderPreservesMixedUserContent(t *testing.T) {
	t.Parallel()

	for _, textFirst := range []bool{false, true} {
		name := "tool_first"
		if textFirst {
			name = "text_first"
		}

		t.Run(name, func(t *testing.T) {
			t.Parallel()

			text := llm.NewTextPart("Summarize both results concisely.")

			parts := []llm.Part{
				llm.NewToolResponsePart("call_1", "lookup", json.RawMessage(`"first"`), false),
				llm.NewToolResponsePart("call_2", "lookup", json.RawMessage(`"second"`), false),
			}
			if textFirst {
				parts = append([]llm.Part{text}, parts...)
			} else {
				parts = append(parts, text)
			}

			req := &llm.Request{Messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant,
					llm.NewToolRequestPart("call_1", "lookup", json.RawMessage(`{}`)),
					llm.NewToolRequestPart("call_2", "lookup", json.RawMessage(`{}`))),
				llm.NewMessage(llm.RoleUser, parts...),
			}}
			mapped, err := openaicompat.NewRequestMapper(&openaicompat.Config{ModelName: "test"}).ToProvider(req)
			require.NoError(t, err)
			wire, err := json.Marshal(mapped.Messages)
			require.NoError(t, err)
			require.JSONEq(t, `[
    {"role":"assistant","tool_calls":[
     {"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}},
     {"id":"call_2","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
    {"role":"tool","tool_call_id":"call_1","content":"\"first\""},
    {"role":"tool","tool_call_id":"call_2","content":"\"second\""},
    {"role":"user","content":"Summarize both results concisely."}
   ]`, string(wire))
		})
	}
}

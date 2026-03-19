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

package openai_test

import (
	"context"
	"net/http"
	"testing"
	"time"

	nativeopenai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/responses"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/wireconformance"
)

// cannedOpenAIResponse is a minimal valid OpenAI Responses API response.
// It must be valid enough that neither SDK chokes during response parsing,
// but the exact content doesn't matter -- we only care about the request.
var cannedOpenAIResponse = []byte(`{
	"id": "resp_test_001",
	"object": "response",
	"created_at": 1700000000,
	"status": "completed",
	"model": "gpt-5-mini",
	"output": [
		{
			"type": "message",
			"id": "msg_test_001",
			"status": "completed",
			"role": "assistant",
			"content": [
				{
					"type": "output_text",
					"text": "Hello, World!",
					"annotations": []
				}
			]
		}
	],
	"usage": {
		"input_tokens": 10,
		"output_tokens": 5,
		"total_tokens": 15,
		"input_tokens_details": {"cached_tokens": 0},
		"output_tokens_details": {"reasoning_tokens": 0}
	},
	"text": {
		"format": {"type": "text"}
	}
}`)

func TestOpenAIWireConformance(t *testing.T) {
	t.Parallel()

	scenarios := []wireconformance.WireScenario{
		simpleTextGeneration(),
	}

	for _, scenario := range scenarios {
		wireconformance.RunScenario(t, scenario, cannedOpenAIResponse)
	}
}

func simpleTextGeneration() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "simple_text_generation",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()

			client := nativeopenai.NewClient(
				option.WithAPIKey("test-key"),
				option.WithHTTPClient(&http.Client{Transport: transport}),
			)

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			// Use the native SDK exactly how a user would.
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Say 'Hello, World!' and nothing else.", responses.EasyInputMessageRoleUser),
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()

			provider, err := openai.NewProvider("test-key",
				openai.WithHTTPClient(&http.Client{Transport: transport}),
			)
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}

			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Say 'Hello, World!' and nothing else.")),
				},
			})
		},
		IgnorePaths: []string{
			"stream",           // ai-sdk may or may not set this
			"stream_options",   // streaming config
			"truncation",      // SDK default that native may not set
		},
		IgnoreHeaders: []string{
			"X-Stainless-Lang",
			"X-Stainless-Package-Version",
			"X-Stainless-Os",
			"X-Stainless-Arch",
			"X-Stainless-Runtime",
			"X-Stainless-Runtime-Version",
			"X-Stainless-Retry-Count",
			"X-Stainless-Read-Timeout",
			"X-Stainless-Poll-Helper",
			"Idempotency-Key",
			"Openai-Organization",
		},
		FixHint: "providers/openai/request_mapper.go (ToProvider method)",
	}
}

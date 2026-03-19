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
	"encoding/json"
	"net/http"
	"testing"
	"time"

	nativeopenai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
	"github.com/openai/openai-go/v3/shared/constant"

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

// defaultIgnorePaths are fields that legitimately differ between native and ai-sdk calls.
var defaultIgnorePaths = []string{
	"stream",
	"stream_options",
	"truncation",
}

// defaultIgnoreHeaders are SDK-internal headers that don't affect behavior.
var defaultIgnoreHeaders = []string{
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
}

func TestOpenAIWireConformance(t *testing.T) {
	t.Parallel()

	scenarios := []wireconformance.WireScenario{
		simpleTextGeneration(),
		systemMessage(),
		multiTurnConversation(),
		temperatureSetting(),
		topPSetting(),
		multipleToolDefinitions(),
		toolDefinitionWithAutoChoice(),
		toolChoiceRequired(),
		toolChoiceSpecificFunction(),
		toolCallRoundTrip(),
		structuredOutputJSONSchema(),
		reasoningEffortAndSummary(),
		schemaWithOptionalFields(),
	}

	for _, scenario := range scenarios {
		wireconformance.RunScenario(t, scenario, cannedOpenAIResponse)
	}
}

func newTestCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

func simpleTextGeneration() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "simple_text_generation",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
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
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Say 'Hello, World!' and nothing else.")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method)",
	}
}

func systemMessage() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "system_message",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("You are a helpful assistant that speaks like a pirate.", responses.EasyInputMessageRoleSystem),
						responses.ResponseInputItemParamOfMessage("Tell me about Go.", responses.EasyInputMessageRoleUser),
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleSystem, llm.NewTextPart("You are a helpful assistant that speaks like a pirate.")),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Tell me about Go.")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method)",
	}
}

func multiTurnConversation() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "multi_turn_conversation",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("What is 2+2?", responses.EasyInputMessageRoleUser),
						responses.ResponseInputItemParamOfMessage("2+2 equals 4.", responses.EasyInputMessageRoleAssistant),
						responses.ResponseInputItemParamOfMessage("And what is 4+4?", responses.EasyInputMessageRoleUser),
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("What is 2+2?")),
					llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("2+2 equals 4.")),
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("And what is 4+4?")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method)",
	}
}

func temperatureSetting() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "temperature_setting",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Hello", responses.EasyInputMessageRoleUser),
					},
				},
				Temperature:     nativeopenai.Float(0.7),
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithTemperature(0.7), openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method)",
	}
}

func toolDefinitionWithAutoChoice() wireconformance.WireScenario {
	weatherSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "City name",
			},
		},
		"required":             []any{"location"},
		"additionalProperties": false,
	}

	schemaBytes, _ := json.Marshal(weatherSchema)

	return wireconformance.WireScenario{
		Name: "tool_definition_with_auto_choice",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("What's the weather in Berlin?", responses.EasyInputMessageRoleUser),
					},
				},
				Tools: []responses.ToolUnionParam{
					{
						OfFunction: &responses.FunctionToolParam{
							Type:        constant.Function(""),
							Name:        "get_weather",
							Description: param.NewOpt("Get current weather for a city"),
							Parameters:  weatherSchema,
							Strict:      param.NewOpt(true),
						},
					},
				},
				ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
					OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsAuto),
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather in Berlin?")),
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get current weather for a city",
						Parameters:  schemaBytes,
					},
				},
				ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceAuto},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapToolDefinitions / mapToolChoice)",
	}
}

func toolChoiceRequired() wireconformance.WireScenario {
	weatherSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "City name",
			},
		},
		"required":             []any{"location"},
		"additionalProperties": false,
	}
	schemaBytes, _ := json.Marshal(weatherSchema)

	return wireconformance.WireScenario{
		Name: "tool_choice_required",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("What's the weather?", responses.EasyInputMessageRoleUser),
					},
				},
				Tools: []responses.ToolUnionParam{
					{
						OfFunction: &responses.FunctionToolParam{
							Type:        constant.Function(""),
							Name:        "get_weather",
							Description: param.NewOpt("Get current weather for a city"),
							Parameters:  weatherSchema,
							Strict:      param.NewOpt(true),
						},
					},
				},
				ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
					OfToolChoiceMode: param.NewOpt(responses.ToolChoiceOptionsRequired),
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather?")),
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get current weather for a city",
						Parameters:  schemaBytes,
					},
				},
				ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceRequired},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapToolChoice)",
	}
}

func toolChoiceSpecificFunction() wireconformance.WireScenario {
	weatherSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{"type": "string"},
		},
		"required":             []any{"location"},
		"additionalProperties": false,
	}
	schemaBytes, _ := json.Marshal(weatherSchema)

	return wireconformance.WireScenario{
		Name: "tool_choice_specific_function",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Weather in Berlin", responses.EasyInputMessageRoleUser),
					},
				},
				Tools: []responses.ToolUnionParam{
					{
						OfFunction: &responses.FunctionToolParam{
							Type:        constant.Function(""),
							Name:        "get_weather",
							Description: param.NewOpt("Get weather"),
							Parameters:  weatherSchema,
							Strict:      param.NewOpt(true),
						},
					},
				},
				ToolChoice: responses.ResponseNewParamsToolChoiceUnion{
					OfFunctionTool: &responses.ToolChoiceFunctionParam{
						Type: "function",
						Name: "get_weather",
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			funcName := "get_weather"
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Weather in Berlin")),
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get weather",
						Parameters:  schemaBytes,
					},
				},
				ToolChoice: &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &funcName},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapToolChoice)",
	}
}

func toolCallRoundTrip() wireconformance.WireScenario {
	// Simulates: user asks -> assistant calls tool -> user provides tool result -> next turn
	return wireconformance.WireScenario{
		Name: "tool_call_round_trip",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						// User message
						responses.ResponseInputItemParamOfMessage("What's the weather in Berlin?", responses.EasyInputMessageRoleUser),
						// Assistant's tool call
						{
							OfFunctionCall: &responses.ResponseFunctionToolCallParam{
								CallID:    "call_123",
								Name:      "get_weather",
								Arguments: `{"location":"Berlin"}`,
								Type:      constant.FunctionCall(""),
							},
						},
						// Tool result
						{
							OfFunctionCallOutput: &responses.ResponseInputItemFunctionCallOutputParam{
								CallID: "call_123",
								Output: responses.ResponseInputItemFunctionCallOutputOutputUnionParam{
									OfString: param.NewOpt(`{"temp":20,"unit":"celsius"}`),
								},
								Type: constant.FunctionCallOutput(""),
							},
						},
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the weather in Berlin?")),
					llm.NewMessage(llm.RoleAssistant, llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        "call_123",
						Name:      "get_weather",
						Arguments: json.RawMessage(`{"location":"Berlin"}`),
					})),
					llm.NewMessage(llm.RoleUser, llm.NewToolResponsePart(&llm.ToolResponse{
						ID:     "call_123",
						Result: json.RawMessage(`{"temp":20,"unit":"celsius"}`),
					})),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapToolRequestMessage / mapToolResponseMessage)",
	}
}

func structuredOutputJSONSchema() wireconformance.WireScenario {
	outputSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": "string",
			},
			"age": map[string]any{
				"type": "integer",
			},
		},
		"required":             []any{"name", "age"},
		"additionalProperties": false,
	}
	schemaBytes, _ := json.Marshal(outputSchema)

	return wireconformance.WireScenario{
		Name: "structured_output_json_schema",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Extract: John is 30 years old", responses.EasyInputMessageRoleUser),
					},
				},
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
							Type:        "json_schema",
							Name:        "person",
							Schema:      outputSchema,
							Description: param.NewOpt("A person's details"),
							Strict:      param.NewOpt(true),
						},
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Extract: John is 30 years old")),
				},
				ResponseFormat: &llm.ResponseFormat{
					Type: llm.ResponseFormatJSONSchema,
					JSONSchema: &llm.JSONSchema{
						Name:        "person",
						Description: "A person's details",
						Schema:      schemaBytes,
					},
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapResponseFormat)",
	}
}

// multipleToolDefinitions tests that multiple tools with different schemas are correctly mapped.
func multipleToolDefinitions() wireconformance.WireScenario {
	weatherSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{"type": "string"},
		},
		"required":             []any{"location"},
		"additionalProperties": false,
	}
	calcSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"expression": map[string]any{"type": "string", "description": "Math expression to evaluate"},
		},
		"required":             []any{"expression"},
		"additionalProperties": false,
	}
	weatherBytes, _ := json.Marshal(weatherSchema)
	calcBytes, _ := json.Marshal(calcSchema)

	return wireconformance.WireScenario{
		Name: "multiple_tool_definitions",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("What's 2+2 and what's the weather in Berlin?", responses.EasyInputMessageRoleUser),
					},
				},
				Tools: []responses.ToolUnionParam{
					{
						OfFunction: &responses.FunctionToolParam{
							Type:        constant.Function(""),
							Name:        "get_weather",
							Description: param.NewOpt("Get weather for a location"),
							Parameters:  weatherSchema,
							Strict:      param.NewOpt(true),
						},
					},
					{
						OfFunction: &responses.FunctionToolParam{
							Type:        constant.Function(""),
							Name:        "calculate",
							Description: param.NewOpt("Evaluate a math expression"),
							Parameters:  calcSchema,
							Strict:      param.NewOpt(true),
						},
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's 2+2 and what's the weather in Berlin?")),
				},
				Tools: []llm.ToolDefinition{
					{
						Name:        "get_weather",
						Description: "Get weather for a location",
						Parameters:  weatherBytes,
					},
					{
						Name:        "calculate",
						Description: "Evaluate a math expression",
						Parameters:  calcBytes,
					},
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (mapToolDefinitions)",
	}
}

// topPSetting tests that top_p is actually sent on the wire.
// BUG: The request mapper has a TopP field in Config but never maps it to the API request.
func topPSetting() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "top_p_setting",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Hello", responses.EasyInputMessageRoleUser),
					},
				},
				TopP:            nativeopenai.Float(0.9),
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithTopP(0.9), openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method - TopP not mapped)",
	}
}

// reasoningEffortAndSummary tests reasoning parameters with a reasoning-capable model.
func reasoningEffortAndSummary() wireconformance.WireScenario {
	return wireconformance.WireScenario{
		Name: "reasoning_effort_and_summary",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Solve: what is 123 * 456?", responses.EasyInputMessageRoleUser),
					},
				},
				Reasoning: shared.ReasoningParam{
					Effort:  shared.ReasoningEffortHigh,
					Summary: shared.ReasoningSummaryDetailed,
				},
				MaxOutputTokens: nativeopenai.Int(1024),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5",
				openai.WithReasoningEffort(openai.ReasoningEffortHigh),
				openai.WithReasoningSummary(openai.ReasoningSummaryDetailed),
				openai.WithMaxTokens(1024),
			)
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Solve: what is 123 * 456?")),
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/request_mapper.go (ToProvider method - reasoning config)",
	}
}

// schemaWithOptionalFields tests that our schema adaptation (making all fields required,
// using ["type", "null"] for optional fields) matches what a user would pass natively.
// This catches divergences in the SchemaMapper.
func schemaWithOptionalFields() wireconformance.WireScenario {
	// Schema where "nickname" is optional (not in required).
	// Our SDK's SchemaMapper should transform this to match OpenAI's strict mode requirements.
	// The native side passes the schema already in OpenAI format (all required, nullable union).
	openaiStyleSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": "string",
			},
			"nickname": map[string]any{
				"type": []any{"string", "null"},
			},
		},
		"required":             []any{"name", "nickname"},
		"additionalProperties": false,
	}

	// The schema as a user would naturally write it (standard JSON Schema).
	userStyleSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{
				"type": "string",
			},
			"nickname": map[string]any{
				"type": "string",
			},
		},
		"required":             []any{"name"},
		"additionalProperties": false,
	}
	userSchemaBytes, _ := json.Marshal(userStyleSchema)

	return wireconformance.WireScenario{
		Name: "schema_with_optional_fields",
		NativeCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			client := nativeopenai.NewClient(option.WithAPIKey("test-key"), option.WithHTTPClient(&http.Client{Transport: transport}))
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = client.Responses.New(ctx, responses.ResponseNewParams{
				Model: "gpt-5-mini",
				Input: responses.ResponseNewParamsInputUnion{
					OfInputItemList: []responses.ResponseInputItemUnionParam{
						responses.ResponseInputItemParamOfMessage("Extract person info from: Bob, also known as Bobby", responses.EasyInputMessageRoleUser),
					},
				},
				Text: responses.ResponseTextConfigParam{
					Format: responses.ResponseFormatTextConfigUnionParam{
						OfJSONSchema: &responses.ResponseFormatTextJSONSchemaConfigParam{
							Type:        "json_schema",
							Name:        "person_info",
							Schema:      openaiStyleSchema,
							Description: param.NewOpt("Person with optional nickname"),
							Strict:      param.NewOpt(true),
						},
					},
				},
				MaxOutputTokens: nativeopenai.Int(256),
			})
		},
		SDKCall: func(t *testing.T, transport *wireconformance.RecordingTransport) {
			t.Helper()
			provider, err := openai.NewProvider("test-key", openai.WithHTTPClient(&http.Client{Transport: transport}))
			if err != nil {
				t.Fatalf("failed to create provider: %v", err)
			}
			model, err := provider.NewModel("gpt-5-mini", openai.WithMaxTokens(256))
			if err != nil {
				t.Fatalf("failed to create model: %v", err)
			}
			ctx, cancel := newTestCtx()
			defer cancel()
			_, _ = model.Generate(ctx, &llm.Request{
				Messages: []llm.Message{
					llm.NewMessage(llm.RoleUser, llm.NewTextPart("Extract person info from: Bob, also known as Bobby")),
				},
				ResponseFormat: &llm.ResponseFormat{
					Type: llm.ResponseFormatJSONSchema,
					JSONSchema: &llm.JSONSchema{
						Name:        "person_info",
						Description: "Person with optional nickname",
						Schema:      userSchemaBytes,
					},
				},
			})
		},
		IgnorePaths:   defaultIgnorePaths,
		IgnoreHeaders: defaultIgnoreHeaders,
		FixHint:       "providers/openai/schema_mapper.go (AdaptSchemaForOpenAI)",
	}
}

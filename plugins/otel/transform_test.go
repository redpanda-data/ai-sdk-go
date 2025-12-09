package otel

import (
	"encoding/json"
	"testing"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

func TestTransformInputMessages_OTelCompliance(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		messages []llm.Message
		want     string // Expected JSON output
	}{
		{
			name: "text message",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello, world!")),
			},
			want: `[{"role":"user","parts":[{"type":"text","content":"Hello, world!"}]}]`,
		},
		{
			name: "message with tool call",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant,
					llm.NewTextPart("Let me search for that."),
					llm.NewToolRequestPart(&llm.ToolRequest{
						ID:        "call_123",
						Name:      "search",
						Arguments: json.RawMessage(`{"query":"test"}`),
					}),
				),
			},
			want: `[{"role":"assistant","parts":[{"type":"text","content":"Let me search for that."},{"type":"tool_call","name":"search","id":"call_123","arguments":{"query":"test"}}]}]`,
		},
		{
			name: "message with tool response",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleUser,
					llm.NewToolResponsePart(&llm.ToolResponse{
						ID:     "call_123",
						Name:   "search",
						Result: json.RawMessage(`{"results":["result1","result2"]}`),
					}),
				),
			},
			want: `[{"role":"user","parts":[{"type":"tool_call_response","id":"call_123","response":{"results":["result1","result2"]}}]}]`,
		},
		{
			name: "message with tool response error",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleUser,
					llm.NewToolResponsePart(&llm.ToolResponse{
						ID:    "call_123",
						Name:  "search",
						Error: "API rate limit exceeded",
					}),
				),
			},
			want: `[{"role":"user","parts":[{"type":"tool_call_response","id":"call_123","response":{"error":"API rate limit exceeded"}}]}]`,
		},
		{
			name: "message with reasoning",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleAssistant,
					llm.NewReasoningPart(&llm.ReasoningTrace{
						ID:   "reasoning_123",
						Text: "Let me think about this step by step...",
					}),
					llm.NewTextPart("Here's my answer."),
				),
			},
			want: `[{"role":"assistant","parts":[{"type":"reasoning","content":"Let me think about this step by step..."},{"type":"text","content":"Here's my answer."}]}]`,
		},
		{
			name: "multiple messages",
			messages: []llm.Message{
				llm.NewMessage(llm.RoleSystem, llm.NewTextPart("You are a helpful assistant.")),
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello!")),
				llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Hi there! How can I help you today?")),
			},
			want: `[{"role":"system","parts":[{"type":"text","content":"You are a helpful assistant."}]},{"role":"user","parts":[{"type":"text","content":"Hello!"}]},{"role":"assistant","parts":[{"type":"text","content":"Hi there! How can I help you today?"}]}]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Transform messages
			otelMessages := transformInputMessages(tt.messages)

			// Serialize to JSON
			got, err := json.Marshal(otelMessages)
			if err != nil {
				t.Fatalf("failed to marshal OTel messages: %v", err)
			}

			// Compare JSON output
			if string(got) != tt.want {
				t.Errorf("transformInputMessages() output mismatch\ngot:  %s\nwant: %s", string(got), tt.want)
			}

			// Verify JSON is valid and contains expected fields
			var parsed []map[string]any
			if err := json.Unmarshal(got, &parsed); err != nil {
				t.Fatalf("output is not valid JSON: %v", err)
			}

			// Verify OTel schema compliance for each message
			for i, msg := range parsed {
				// Check required fields
				if _, ok := msg["role"]; !ok {
					t.Errorf("message[%d] missing required field 'role'", i)
				}

				parts, ok := msg["parts"]
				if !ok {
					t.Errorf("message[%d] missing required field 'parts'", i)
					continue
				}

				// Verify parts is an array
				partsArray, ok := parts.([]any)
				if !ok {
					t.Errorf("message[%d] 'parts' is not an array", i)
					continue
				}

				// Verify each part has a 'type' field
				for j, part := range partsArray {
					partMap, ok := part.(map[string]any)
					if !ok {
						t.Errorf("message[%d].parts[%d] is not an object", i, j)
						continue
					}

					partType, ok := partMap["type"]
					if !ok {
						t.Errorf("message[%d].parts[%d] missing required field 'type'", i, j)
						continue
					}

					// Verify type is one of the allowed values
					typeStr, ok := partType.(string)
					if !ok {
						t.Errorf("message[%d].parts[%d] 'type' is not a string", i, j)
						continue
					}

					validTypes := map[string]bool{
						"text":               true,
						"tool_call":          true,
						"tool_call_response": true,
						"reasoning":          true,
					}

					if !validTypes[typeStr] {
						t.Errorf("message[%d].parts[%d] has invalid type '%s'", i, j, typeStr)
					}
				}
			}
		})
	}
}

func TestTransformOutputMessage_OTelCompliance(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name         string
		message      llm.Message
		finishReason string
		want         string // Expected JSON output
	}{
		{
			name:         "simple text response with finish reason",
			message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Hello, world!")),
			finishReason: "stop",
			want:         `{"role":"assistant","parts":[{"type":"text","content":"Hello, world!"}],"finish_reason":"stop"}`,
		},
		{
			name: "response with tool call and finish reason",
			message: llm.NewMessage(llm.RoleAssistant,
				llm.NewToolRequestPart(&llm.ToolRequest{
					ID:        "call_456",
					Name:      "calculate",
					Arguments: json.RawMessage(`{"operation":"add","values":[1,2]}`),
				}),
			),
			finishReason: "tool_call",
			want:         `{"role":"assistant","parts":[{"type":"tool_call","name":"calculate","id":"call_456","arguments":{"operation":"add","values":[1,2]}}],"finish_reason":"tool_call"}`,
		},
		{
			name:         "response stopped due to length",
			message:      llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("This is a very long response that was cut off...")),
			finishReason: "length",
			want:         `{"role":"assistant","parts":[{"type":"text","content":"This is a very long response that was cut off..."}],"finish_reason":"length"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			// Transform message
			otelMessage := transformOutputMessage(tt.message, tt.finishReason)

			// Serialize to JSON
			got, err := json.Marshal(otelMessage)
			if err != nil {
				t.Fatalf("failed to marshal OTel message: %v", err)
			}

			// Compare JSON output
			if string(got) != tt.want {
				t.Errorf("transformOutputMessage() output mismatch\ngot:  %s\nwant: %s", string(got), tt.want)
			}

			// Verify JSON is valid and contains expected fields
			var parsed map[string]any
			if err := json.Unmarshal(got, &parsed); err != nil {
				t.Fatalf("output is not valid JSON: %v", err)
			}

			// Verify required fields for output messages
			if _, ok := parsed["role"]; !ok {
				t.Error("missing required field 'role'")
			}

			if _, ok := parsed["parts"]; !ok {
				t.Error("missing required field 'parts'")
			}

			if _, ok := parsed["finish_reason"]; !ok {
				t.Error("missing required field 'finish_reason'")
			}

			// Verify finish_reason value
			if fr, ok := parsed["finish_reason"].(string); ok {
				if fr != tt.finishReason {
					t.Errorf("finish_reason mismatch: got %q, want %q", fr, tt.finishReason)
				}
			} else {
				t.Error("finish_reason is not a string")
			}
		})
	}
}

func TestTransformPart_AllTypes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		part *llm.Part
		want otelPart
	}{
		{
			name: "text part",
			part: llm.NewTextPart("Hello"),
			want: otelPart{Type: "text", Content: "Hello"},
		},
		{
			name: "tool request part",
			part: llm.NewToolRequestPart(&llm.ToolRequest{
				ID:        "call_789",
				Name:      "weather",
				Arguments: json.RawMessage(`{"location":"NYC"}`),
			}),
			want: otelPart{
				Type:      "tool_call",
				ID:        "call_789",
				Name:      "weather",
				Arguments: json.RawMessage(`{"location":"NYC"}`),
			},
		},
		{
			name: "tool response part",
			part: llm.NewToolResponsePart(&llm.ToolResponse{
				ID:     "call_789",
				Name:   "weather",
				Result: json.RawMessage(`{"temp":72,"condition":"sunny"}`),
			}),
			want: otelPart{
				Type:     "tool_call_response",
				ID:       "call_789",
				Response: json.RawMessage(`{"temp":72,"condition":"sunny"}`),
			},
		},
		{
			name: "reasoning part",
			part: llm.NewReasoningPart(&llm.ReasoningTrace{
				ID:   "reason_123",
				Text: "First, I need to consider...",
			}),
			want: otelPart{
				Type:    "reasoning",
				Content: "First, I need to consider...",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := transformPart(tt.part)

			// Compare each field
			if got.Type != tt.want.Type {
				t.Errorf("Type mismatch: got %q, want %q", got.Type, tt.want.Type)
			}

			if got.Content != tt.want.Content {
				t.Errorf("Content mismatch: got %q, want %q", got.Content, tt.want.Content)
			}

			if got.ID != tt.want.ID {
				t.Errorf("ID mismatch: got %q, want %q", got.ID, tt.want.ID)
			}

			if got.Name != tt.want.Name {
				t.Errorf("Name mismatch: got %q, want %q", got.Name, tt.want.Name)
			}

			if string(got.Arguments) != string(tt.want.Arguments) {
				t.Errorf("Arguments mismatch: got %s, want %s", got.Arguments, tt.want.Arguments)
			}

			if string(got.Response) != string(tt.want.Response) {
				t.Errorf("Response mismatch: got %s, want %s", got.Response, tt.want.Response)
			}
		})
	}
}

// Package a2aconv provides utilities for converting between A2A protocol messages
// and the AI SDK's LLM message format.
package a2aconv

import (
	"encoding/json"

	"github.com/a2aproject/a2a-go/a2a"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// MessageToLLM converts an a2a-go message to LLM SDK message.
func MessageToLLM(msg *a2a.Message) llm.Message {
	// Map role
	var role llm.MessageRole

	switch msg.Role {
	case a2a.MessageRoleUser:
		role = llm.RoleUser
	case a2a.MessageRoleAgent:
		role = llm.RoleAssistant
	case a2a.MessageRoleUnspecified:
		role = llm.RoleUser // fallback
	default:
		role = llm.RoleUser // fallback
	}

	// Convert parts
	parts := make([]*llm.Part, 0, len(msg.Parts))
	for _, part := range msg.Parts {
		if p, ok := part.(a2a.TextPart); ok {
			parts = append(parts, llm.NewTextPart(p.Text))
		}
		// TODO: Add support for tool calls, data parts, etc. when needed
	}

	return llm.NewMessage(role, parts...)
}

// MessagesToLLM converts a slice of a2a-go messages to LLM SDK messages.
func MessagesToLLM(a2aMessages []*a2a.Message) []llm.Message {
	llmMessages := make([]llm.Message, 0, len(a2aMessages))
	for _, msg := range a2aMessages {
		llmMessages = append(llmMessages, MessageToLLM(msg))
	}

	return llmMessages
}

// MessageFromLLM converts an LLM SDK message to a2a-go message.
func MessageFromLLM(llmMsg llm.Message) *a2a.Message {
	// Map role
	var role a2a.MessageRole

	switch llmMsg.Role {
	case llm.RoleUser:
		role = a2a.MessageRoleUser
	case llm.RoleAssistant:
		role = a2a.MessageRoleAgent
	case llm.RoleSystem:
		role = a2a.MessageRoleAgent // System messages become agent messages
	default:
		role = a2a.MessageRoleUser // fallback
	}

	// Convert parts
	parts := make([]a2a.Part, 0, len(llmMsg.Content))
	for _, part := range llmMsg.Content {
		switch {
		case part.IsText():
			parts = append(parts, a2a.TextPart{Text: part.Text})
		case part.IsToolRequest() && part.ToolRequest != nil:
			// Convert tool request to DataPart with structured content
			var args map[string]any
			if err := json.Unmarshal(part.ToolRequest.Arguments, &args); err == nil {
				parts = append(parts, a2a.DataPart{
					Data: map[string]any{
						"type":      "tool_request",
						"id":        part.ToolRequest.ID,
						"name":      part.ToolRequest.Name,
						"arguments": args,
					},
				})
			}
		case part.IsToolResponse() && part.ToolResponse != nil:
			// Convert tool response to DataPart
			var result map[string]any
			if err := json.Unmarshal(part.ToolResponse.Result, &result); err == nil {
				data := map[string]any{
					"type":   "tool_response",
					"id":     part.ToolResponse.ID,
					"name":   part.ToolResponse.Name,
					"result": result,
				}
				if part.ToolResponse.Error != "" {
					data["error"] = part.ToolResponse.Error
				}

				parts = append(parts, a2a.DataPart{
					Data: data,
				})
			}
		}
	}

	return a2a.NewMessage(role, parts...)
}

// MessagesFromLLM converts a slice of LLM SDK messages to a2a-go messages.
func MessagesFromLLM(llmMessages []llm.Message) []*a2a.Message {
	a2aMessages := make([]*a2a.Message, 0, len(llmMessages))
	for _, msg := range llmMessages {
		a2aMessages = append(a2aMessages, MessageFromLLM(msg))
	}

	return a2aMessages
}

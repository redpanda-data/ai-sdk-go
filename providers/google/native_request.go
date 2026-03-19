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

package google

import (
	"encoding/json"
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Request-side native types for JSON unmarshaling. Prefixed with "nativeReq"
// to avoid collision with the response-side native types in native_response.go.

type nativeReqBody struct {
	Contents          []nativeReqContent       `json:"contents"`
	SystemInstruction *nativeReqContent        `json:"systemInstruction,omitempty"`
	Tools             []nativeReqToolGroup     `json:"tools,omitempty"`
	ToolConfig        *nativeReqToolConfig     `json:"toolConfig,omitempty"`
	GenerationConfig  *nativeReqGenerationConf `json:"generationConfig,omitempty"`
	Model             string                   `json:"model,omitempty"`
}

type nativeReqContent struct {
	Role  string           `json:"role,omitempty"`
	Parts []nativeReqPart  `json:"parts"`
}

type nativeReqPart struct {
	Text             string                    `json:"text,omitempty"`
	FunctionCall     *nativeReqFunctionCall    `json:"functionCall,omitempty"`
	FunctionResponse *nativeReqFunctionResp    `json:"functionResponse,omitempty"`
	Thought          bool                      `json:"thought,omitempty"`
}

type nativeReqFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type nativeReqFunctionResp struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response,omitempty"`
}

type nativeReqToolGroup struct {
	FunctionDeclarations []nativeReqFuncDecl `json:"functionDeclarations,omitempty"`
}

type nativeReqFuncDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type nativeReqToolConfig struct {
	FunctionCallingConfig *nativeReqFuncCallConfig `json:"functionCallingConfig,omitempty"`
}

type nativeReqFuncCallConfig struct {
	Mode                 string   `json:"mode,omitempty"`
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type nativeReqGenerationConf struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	TopP             *float64 `json:"topP,omitempty"`
	TopK             *int     `json:"topK,omitempty"`
	MaxOutputTokens  *int     `json:"maxOutputTokens,omitempty"`
	StopSequences    []string `json:"stopSequences,omitempty"`
	CandidateCount   *int     `json:"candidateCount,omitempty"`
	PresencePenalty  *float64 `json:"presencePenalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequencyPenalty,omitempty"`
}

// FromNative parses a raw Gemini generateContent request JSON body into
// a unified llm.Request and extracts the model name.
//
// The model name typically comes from the URL path in Gemini
// (/v1beta/models/{model}:generateContent), not from the request body.
// If a "model" field exists in the body it is returned; otherwise an
// empty string is returned and the caller is expected to extract it
// from the URL.
func (rm *RequestMapper) FromNative(body []byte) (*llm.Request, string, error) {
	var nr nativeReqBody
	if err := json.Unmarshal(body, &nr); err != nil {
		return nil, "", fmt.Errorf("failed to unmarshal native request: %w", err)
	}

	req := &llm.Request{}

	// Parse system instruction
	if nr.SystemInstruction != nil {
		systemMsg, err := parseNativeReqSystemInstruction(nr.SystemInstruction)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse systemInstruction: %w", err)
		}
		if systemMsg != nil {
			req.Messages = append(req.Messages, *systemMsg)
		}
	}

	// Parse content messages
	for i, content := range nr.Contents {
		msg, err := parseNativeReqContent(content)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse content %d: %w", i, err)
		}
		req.Messages = append(req.Messages, *msg)
	}

	// Parse tools
	for _, toolGroup := range nr.Tools {
		for _, decl := range toolGroup.FunctionDeclarations {
			req.Tools = append(req.Tools, llm.ToolDefinition{
				Name:        decl.Name,
				Description: decl.Description,
				Parameters:  decl.Parameters,
			})
		}
	}

	// Parse tool config
	if nr.ToolConfig != nil && nr.ToolConfig.FunctionCallingConfig != nil {
		tc, err := parseNativeReqToolConfig(nr.ToolConfig.FunctionCallingConfig)
		if err != nil {
			return nil, "", fmt.Errorf("failed to parse toolConfig: %w", err)
		}
		req.ToolChoice = tc
	}

	return req, nr.Model, nil
}

// parseNativeReqSystemInstruction converts the Gemini systemInstruction to a system message.
func parseNativeReqSystemInstruction(si *nativeReqContent) (*llm.Message, error) {
	if len(si.Parts) == 0 {
		return nil, nil
	}

	var parts []*llm.Part
	for _, p := range si.Parts {
		parts = append(parts, llm.NewTextPart(p.Text))
	}

	msg := llm.NewMessage(llm.RoleSystem, parts...)
	return &msg, nil
}

// parseNativeReqContent converts a Gemini content entry to a unified llm.Message.
func parseNativeReqContent(content nativeReqContent) (*llm.Message, error) {
	var role llm.MessageRole
	switch content.Role {
	case "user":
		role = llm.RoleUser
	case "model":
		role = llm.RoleAssistant
	default:
		return nil, fmt.Errorf("unsupported content role: %s", content.Role)
	}

	var parts []*llm.Part
	for _, p := range content.Parts {
		part, err := parseNativeReqPart(p, role)
		if err != nil {
			return nil, err
		}
		parts = append(parts, part)
	}

	msg := llm.NewMessage(role, parts...)
	return &msg, nil
}

// parseNativeReqPart converts a single Gemini part to an llm.Part.
func parseNativeReqPart(p nativeReqPart, role llm.MessageRole) (*llm.Part, error) {
	switch {
	case p.FunctionCall != nil:
		args, err := json.Marshal(p.FunctionCall.Args)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal functionCall args: %w", err)
		}
		if len(args) == 0 || string(args) == "null" {
			args = json.RawMessage(`{}`)
		}
		return llm.NewToolRequestPart(&llm.ToolRequest{
			Name:      p.FunctionCall.Name,
			Arguments: args,
		}), nil

	case p.FunctionResponse != nil:
		resp := &llm.ToolResponse{
			ID:   p.FunctionResponse.Name,
			Name: p.FunctionResponse.Name,
		}
		if p.FunctionResponse.Response != nil {
			resultBytes, err := json.Marshal(p.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal functionResponse: %w", err)
			}
			resp.Result = resultBytes
		}
		return llm.NewToolResponsePart(resp), nil

	case p.Thought && p.Text != "":
		return llm.NewReasoningPart(&llm.ReasoningTrace{
			Text: p.Text,
		}), nil

	default:
		// Text part (including empty text)
		return llm.NewTextPart(p.Text), nil
	}
}

// parseNativeReqToolConfig converts Gemini's function calling config to llm.ToolChoice.
func parseNativeReqToolConfig(fcc *nativeReqFuncCallConfig) (*llm.ToolChoice, error) {
	switch fcc.Mode {
	case "AUTO", "auto":
		return &llm.ToolChoice{Type: llm.ToolChoiceAuto}, nil

	case "ANY", "any":
		// If exactly one allowed function name, treat as specific
		if len(fcc.AllowedFunctionNames) == 1 {
			name := fcc.AllowedFunctionNames[0]
			return &llm.ToolChoice{Type: llm.ToolChoiceSpecific, Name: &name}, nil
		}
		return &llm.ToolChoice{Type: llm.ToolChoiceRequired}, nil

	case "NONE", "none":
		return &llm.ToolChoice{Type: llm.ToolChoiceNone}, nil

	default:
		return nil, fmt.Errorf("unsupported function calling mode: %s", fcc.Mode)
	}
}

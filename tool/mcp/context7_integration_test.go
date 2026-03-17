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

package mcp_test

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/mcp"
	"github.com/redpanda-data/ai-sdk-go/tool/mcp/mcptest"
)

const (
	// Context7 MCP endpoint.
	context7Endpoint = "https://mcp.context7.com/mcp"

	// Test timeout for the entire flow.
	testTimeout = 2 * time.Minute

	// Maximum number of tool calling iterations to prevent infinite loops.
	maxToolIterations = 5
)

// TestContext7Integration_EndToEnd tests the complete integration of:
// 1. MCP client connection to Context7 server
// 2. Tool registration in registry
// 3. LLM requesting tool usage
// 4. Tool execution via registry
// 5. Multi-turn conversation with tool results
// 6. Final response generation.
func TestContext7Integration_EndToEnd(t *testing.T) { //nolint:paralleltest // connects to external Context7 API service with rate limits
	// Skip if API keys not available
	openaiAPIKey := openaitest.GetAPIKeyOrSkipTest(t)
	context7APIKey := mcptest.GetContext7APIKeyOrSkipTest(t)

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create Context7 MCP client with streamable transport
	factory := mcp.NewStreamableTransport(
		context7Endpoint,
		mcp.WithHTTPHeaders(map[string]string{
			"CONTEXT7_API_KEY": context7APIKey,
		}),
	)

	registry := tool.NewRegistry(tool.RegistryConfig{})
	client, err := mcp.NewClient("context7", factory, mcp.WithRegistry(registry))
	require.NoError(t, err)

	err = client.Start(ctx)
	if err != nil {
		t.Skipf("skipping test: failed to connect to Context7 MCP server (external service may be unavailable): %v", err)
	}

	defer func() {
		_ = client.Shutdown(context.Background())
	}()

	// Verify tools are registered
	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	require.NotEmpty(t, tools, "Expected Context7 to provide tools")

	// Verify expected tools are present (namespaced with context7__)
	var hasResolveLibrary, hasQueryDocs bool

	for _, tool := range tools {
		switch tool.Name {
		case "context7__resolve-library-id":
			hasResolveLibrary = true
		case "context7__query-docs":
			hasQueryDocs = true
		}
	}

	require.True(t, hasResolveLibrary, "Expected Context7 to provide context7__resolve-library-id tool")
	require.True(t, hasQueryDocs, "Expected Context7 to provide context7__query-docs tool")

	// Create OpenAI provider and model
	provider, err := openai.NewProvider(openaiAPIKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName)
	require.NoError(t, err)

	// Get tool definitions for LLM
	toolDefinitions := registry.List()
	require.NotEmpty(t, toolDefinitions)

	// Create initial conversation
	messages := []llm.Message{
		{
			Role: llm.RoleUser,
			Content: []*llm.Part{
				llm.NewTextPart("I need to understand how to use the useState hook in React. Use the available tools to find the official documentation about React useState hook and summarize the key points for me."),
			},
		},
	}

	// Multi-turn tool calling loop
	var (
		iteration     int
		toolCallCount int
		finalResponse *llm.Response
	)

	for iteration = range maxToolIterations {
		request := &llm.Request{
			Messages: messages,
			Tools:    toolDefinitions,
		}

		response, err := model.Generate(ctx, request)
		require.NoError(t, err, "LLM generation failed at iteration %d", iteration+1)
		require.NotNil(t, response)

		// Check if LLM is done or wants to use tools
		if response.FinishReason == llm.FinishReasonStop {
			finalResponse = response
			break
		}

		if response.FinishReason != llm.FinishReasonToolCalls {
			t.Fatalf("Unexpected finish reason at iteration %d: %s", iteration+1, response.FinishReason)
		}

		// Extract and execute tool requests
		toolRequests := response.ToolRequests()
		require.NotEmpty(t, toolRequests, "LLM indicated tool calls but provided no requests")

		toolResults := make([]*llm.Part, 0, len(toolRequests))
		for _, toolReq := range toolRequests {
			toolCallCount++

			toolResp, err := registry.Execute(ctx, toolReq)
			require.NoError(t, err, "Failed to execute tool %s", toolReq.Name)
			require.NotNil(t, toolResp)

			toolResults = append(toolResults, llm.NewToolResponsePart(toolResp))
		}

		// Add assistant message with tool calls to conversation
		messages = append(messages, llm.Message{
			Role:    llm.RoleAssistant,
			Content: response.Message.Content,
		})

		// Add tool results to conversation
		messages = append(messages, llm.Message{
			Role:    llm.RoleUser,
			Content: toolResults,
		})
	}

	// Validate results
	require.NotNil(t, finalResponse, "LLM did not complete within %d iterations", maxToolIterations)
	assert.Positive(t, toolCallCount, "Expected at least one tool call")

	finalText := finalResponse.TextContent()
	assert.NotEmpty(t, finalText, "Final response should contain text")

	// Verify the response mentions React and hooks
	finalTextLower := strings.ToLower(finalText)
	assert.True(t,
		strings.Contains(finalTextLower, "react") ||
			strings.Contains(finalTextLower, "usestate") ||
			strings.Contains(finalTextLower, "hook") ||
			strings.Contains(finalTextLower, "state"),
		"Final response should mention React or hooks")

	// Verify conversation structure
	assert.GreaterOrEqual(t, len(messages), 3, "Conversation should have at least user + assistant + tool messages")
}

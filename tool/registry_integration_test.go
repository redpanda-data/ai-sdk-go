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

package tool_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/providers/openai/openaitest"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/builtin/webfetch"
)

// testOptions returns common test options that disable security restrictions for testing.
func testOptions() []webfetch.Option {
	return []webfetch.Option{
		webfetch.WithDenyPrivateIPs(false),                     // Allow localhost for testing
		webfetch.WithAllowedSchemes([]string{"https", "http"}), // Allow both schemes
		webfetch.WithAllowedPorts(nil),                         // Allow all ports in tests
	}
}

// TestRegistry_BasicOperations tests the core registry functionality.
func TestRegistry_BasicOperations(t *testing.T) {
	t.Parallel()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	webfetchTool := webfetch.New(testOptions()...)

	// Test registration
	err := registry.Register(webfetchTool)
	require.NoError(t, err)

	// Test listing
	definitions := registry.List()
	assert.Len(t, definitions, 1)
	assert.Equal(t, "webfetch", definitions[0].Name)
	assert.NotEmpty(t, definitions[0].Description)
	assert.NotNil(t, definitions[0].Parameters)

	// Test retrieval
	retrievedTool, err := registry.Get("webfetch")
	require.NoError(t, err)
	assert.NotNil(t, retrievedTool)

	// Test successful execution
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"slideshow": map[string]any{
				"title": "Sample Slide Show",
			},
		})
	}))
	defer server.Close()

	ctx := context.Background()
	params := webfetch.NewParameters(server.URL).WithMethod("GET")
	req := &llm.ToolRequest{
		ID:        "test-success",
		Name:      "webfetch",
		Arguments: params.MustToJSONRawMessage(),
	}

	response, err := registry.Execute(ctx, req)
	require.NoError(t, err)
	assert.Equal(t, "test-success", response.ID)
	assert.Equal(t, "webfetch", response.Name)
	assert.Empty(t, response.Error)
	assert.NotNil(t, response.Result)

	// Test unregistration
	err = registry.Unregister("webfetch")
	require.NoError(t, err)
	assert.Empty(t, registry.List())

	_, err = registry.Get("webfetch")
	assert.ErrorIs(t, err, tool.ErrToolNotFound)
}

// TestRegistry_ErrorConditions tests all error scenarios with sentinel errors.
func TestRegistry_ErrorConditions(t *testing.T) {
	t.Parallel()

	t.Run("registration errors", func(t *testing.T) {
		t.Parallel()
		// Create isolated registry for this subtest to avoid conflicts with parallel subtests
		registry := tool.NewRegistry(tool.RegistryConfig{})

		// Nil tool
		err := registry.Register(nil)
		require.ErrorIs(t, err, tool.ErrToolNil)

		// Duplicate registration
		webfetchTool := webfetch.New(testOptions()...)
		err = registry.Register(webfetchTool)
		require.NoError(t, err)
		err = registry.Register(webfetchTool)
		require.ErrorIs(t, err, tool.ErrToolAlreadyRegistered)

		// Invalid configuration
		err = registry.Register(webfetch.New(testOptions()...), tool.WithTimeout(-1*time.Second))
		require.ErrorIs(t, err, tool.ErrInvalidToolConfig)

		err = registry.Register(webfetch.New(testOptions()...), tool.WithMaxResponseTokens(-100))
		require.ErrorIs(t, err, tool.ErrInvalidToolConfig)
	})

	t.Run("access errors", func(t *testing.T) {
		t.Parallel()
		// Create isolated registry for this subtest
		registry := tool.NewRegistry(tool.RegistryConfig{})

		// Tool not found
		_, err := registry.Get("nonexistent")
		require.ErrorIs(t, err, tool.ErrToolNotFound)

		// Unregister non-existent
		err = registry.Unregister("nonexistent")
		require.ErrorIs(t, err, tool.ErrToolNotFound)
	})

	t.Run("execution errors", func(t *testing.T) {
		t.Parallel()
		// Create isolated registry for this subtest
		registry := tool.NewRegistry(tool.RegistryConfig{})

		// Nil request
		_, err := registry.Execute(context.Background(), nil)
		require.ErrorIs(t, err, tool.ErrToolRequestNil)

		// Non-existent tool execution
		req := &llm.ToolRequest{
			ID:        "test-not-found",
			Name:      "nonexistent",
			Arguments: json.RawMessage(`{}`),
		}
		response, err := registry.Execute(context.Background(), req)
		require.NoError(t, err)
		assert.Equal(t, "test-not-found", response.ID)
		assert.Contains(t, response.Error, tool.ErrToolNotFound.Error())
		assert.Nil(t, response.Result)
	})
}

// TestRegistry_Configuration tests tool configuration options.
func TestRegistry_Configuration(t *testing.T) {
	t.Parallel()
	t.Run("timeout configuration", func(t *testing.T) {
		t.Parallel()

		registry := tool.NewRegistry(tool.RegistryConfig{})

		// Create server that delays response to trigger timeout
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			time.Sleep(1 * time.Second)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "delayed"})
		}))
		defer server.Close()

		// Register tool with short timeout
		webfetchTool := webfetch.New(testOptions()...)
		err := registry.Register(webfetchTool, tool.WithTimeout(100*time.Millisecond))
		require.NoError(t, err)

		params := webfetch.NewParameters(server.URL).WithMethod("GET")
		req := &llm.ToolRequest{
			ID:        "test-timeout",
			Name:      "webfetch",
			Arguments: params.MustToJSONRawMessage(),
		}

		response, err := registry.Execute(t.Context(), req)
		require.NoError(t, err)

		// Webfetch tool handles context cancellation gracefully
		assert.Empty(t, response.Error)
		assert.NotNil(t, response.Result)

		var result map[string]any

		err = json.Unmarshal(response.Result, &result)
		require.NoError(t, err)

		errBool, ok := result["error"].(bool)
		require.True(t, ok, "error should be a bool")
		assert.True(t, errBool)

		message, ok := result["message"].(string)
		require.True(t, ok, "message should be a string")
		assert.Contains(t, message, "context deadline exceeded")
	})

	t.Run("response size limits", func(t *testing.T) {
		t.Parallel()

		registry := tool.NewRegistry(tool.RegistryConfig{})

		// Create server that returns JSON response
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"slideshow": map[string]any{
					"title": "Sample Slide Show",
				},
			})
		}))
		defer server.Close()

		// Test custom message
		webfetchTool := webfetch.New(testOptions()...)
		err := registry.Register(webfetchTool,
			tool.WithMaxResponseTokens(10),
			tool.WithResponseTooLargeMessage("Custom size limit message"),
		)
		require.NoError(t, err)

		params := webfetch.NewParameters(server.URL).WithMethod("GET")
		req := &llm.ToolRequest{
			ID:        "test-size-custom",
			Name:      "webfetch",
			Arguments: params.MustToJSONRawMessage(),
		}

		response, err := registry.Execute(t.Context(), req)
		require.NoError(t, err)
		assert.NotNil(t, response.Result)

		var result map[string]any

		err = json.Unmarshal(response.Result, &result)
		require.NoError(t, err)

		if errorVal, hasError := result["error"]; hasError {
			assert.Equal(t, "response_too_large", errorVal)
			assert.Equal(t, "Custom size limit message", result["message"])
		}

		// Test default message with new registry
		registry2 := tool.NewRegistry(tool.RegistryConfig{})
		err = registry2.Register(webfetch.New(testOptions()...), tool.WithMaxResponseTokens(1))
		require.NoError(t, err)

		req.ID = "test-size-default"
		response, err = registry2.Execute(t.Context(), req)
		require.NoError(t, err)

		err = json.Unmarshal(response.Result, &result)
		require.NoError(t, err)

		if errorVal, hasError := result["error"]; hasError {
			assert.Equal(t, "response_too_large", errorVal)

			message, ok := result["message"].(string)
			require.True(t, ok, "message should be a string")
			assert.Contains(t, message, "Response too large for context window")
		}
	})
}

// TestRegistry_ConcurrentAccess tests thread safety.
func TestRegistry_ConcurrentAccess(t *testing.T) {
	t.Parallel()
	// Create server that handles concurrent requests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"slideshow": map[string]any{
				"title": "Sample Slide Show",
			},
		})
	}))
	defer server.Close()

	registry := tool.NewRegistry(tool.RegistryConfig{})
	webfetchTool := webfetch.New(testOptions()...)
	err := registry.Register(webfetchTool)
	require.NoError(t, err)

	const numGoroutines = 50 // Reduced from 100 for faster tests

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for range numGoroutines {
		go func() {
			defer wg.Done()

			// Test concurrent read operations
			tool, err := registry.Get("webfetch")
			assert.NoError(t, err)
			assert.NotNil(t, tool)

			definitions := registry.List()
			assert.Len(t, definitions, 1)

			// Test concurrent execution
			ctx := context.Background()
			params := webfetch.NewParameters(server.URL).WithMethod("GET")
			req := &llm.ToolRequest{
				ID:        "concurrent-test",
				Name:      "webfetch",
				Arguments: params.MustToJSONRawMessage(),
			}

			response, err := registry.Execute(ctx, req)
			assert.NoError(t, err)
			assert.NotNil(t, response)
		}()
	}

	wg.Wait()
}

// TestRegistry_WebfetchToolWithLLM_Integration tests the complete end-to-end integration
// of webfetch tool with LLM tool calling. This test validates:
// 1. LLM requests a tool call for webfetch
// 2. Registry executes the webfetch tool
// 3. Tool response is sent back to LLM
// 4. LLM generates final response incorporating webfetch results.
func TestRegistry_WebfetchToolWithLLM_Integration(t *testing.T) {
	t.Parallel()
	apiKey := openaitest.GetAPIKeyOrSkipTest(t)

	// Create test server that returns JSON data
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"slideshow": map[string]any{
				"title":  "Sample Slide Show",
				"author": "Yours Truly",
				"date":   "date of publication",
			},
		})
	}))
	defer server.Close()

	// Create OpenAI provider and model
	provider, err := openai.NewProvider(apiKey)
	require.NoError(t, err)

	model, err := provider.NewModel(openaitest.TestModelName) // Use a model that supports tool calling
	require.NoError(t, err)

	// Verify model supports tools
	caps := model.Capabilities()
	require.True(t, caps.Tools, "Model must support tool calling for this test")

	// Create registry with webfetch tool
	registry := tool.NewRegistry(tool.RegistryConfig{})
	webfetchTool := webfetch.New(testOptions()...)
	err = registry.Register(webfetchTool)
	require.NoError(t, err)

	// Get tool definitions for LLM
	toolDefinitions := registry.List()
	require.Len(t, toolDefinitions, 1)
	require.Equal(t, "webfetch", toolDefinitions[0].Name)

	// Create initial LLM request that should trigger webfetch tool usage
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	initialRequest := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Use the webfetch tool to fetch this exact URL: " + server.URL + " - do NOT modify the URL scheme."),
				},
			},
		},
		Tools: toolDefinitions,
		ToolChoice: &llm.ToolChoice{
			Type: llm.ToolChoiceRequired,
		},
	}

	// Step 1: Send initial request to LLM
	response, err := model.Generate(ctx, initialRequest)
	require.NoError(t, err)
	require.NotNil(t, response)

	// Step 2: Verify LLM wants to use tools
	assert.Equal(t, llm.FinishReasonToolCalls, response.FinishReason,
		"LLM should want to use tools for this request")

	// Extract tool requests from response
	toolRequests := response.ToolRequests()
	require.NotEmpty(t, toolRequests, "LLM should have requested tool execution")

	// Find webfetch tool request
	var webfetchRequest *llm.ToolRequest

	for _, req := range toolRequests {
		if req.Name == "webfetch" {
			webfetchRequest = req
			break
		}
	}

	require.NotNil(t, webfetchRequest, "LLM should have requested webfetch tool")
	assert.NotEmpty(t, webfetchRequest.ID, "Tool request should have ID")
	assert.NotEmpty(t, webfetchRequest.Arguments, "Tool request should have arguments")

	// Step 3: Execute tool using registry
	toolResponse, err := registry.Execute(ctx, webfetchRequest)
	require.NoError(t, err)
	require.NotNil(t, toolResponse)
	assert.Equal(t, webfetchRequest.ID, toolResponse.ID, "Tool response ID should match request ID")
	assert.Equal(t, "webfetch", toolResponse.Name, "Tool response name should match")
	assert.Empty(t, toolResponse.Error, "Tool execution should succeed")
	assert.NotNil(t, toolResponse.Result, "Tool should return results")

	// Verify tool response contains expected data structure
	var result map[string]any

	err = json.Unmarshal(toolResponse.Result, &result)
	require.NoError(t, err, "Tool result should be valid JSON")
	assert.Contains(t, result, "url", "Result should contain URL")
	assert.Contains(t, result, "status_code", "Result should contain status code")

	statusCode, ok := result["status_code"].(float64)
	require.True(t, ok, "status_code should be a float64")
	assert.Equal(t, 200, int(statusCode), "Should successfully fetch from test server")

	// Step 4: Send tool response back to LLM
	followUpRequest := &llm.Request{
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Use the webfetch tool to fetch this exact URL: " + server.URL + " - do NOT modify the URL scheme."),
				},
			},
			{
				Role:    llm.RoleAssistant,
				Content: response.Message.Content, // Include the original tool requests
			},
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewToolResponsePart(toolResponse),
				},
			},
		},
		Tools: toolDefinitions,
		ToolChoice: &llm.ToolChoice{
			Type: llm.ToolChoiceNone,
		},
	}

	// Step 5: Get final response from LLM
	finalResponse, err := model.Generate(ctx, followUpRequest)
	require.NoError(t, err)
	require.NotNil(t, finalResponse)

	// Step 6: Verify final response
	assert.Equal(t, llm.FinishReasonStop, finalResponse.FinishReason,
		"LLM should complete normally after receiving tool results")

	finalText := finalResponse.TextContent()
	assert.NotEmpty(t, finalText, "LLM should provide a final response")

	// Verify LLM incorporated the webfetch results
	finalTextLower := strings.ToLower(finalText)
	assert.True(t,
		strings.Contains(finalTextLower, "json") ||
			strings.Contains(finalTextLower, "slideshow") ||
			strings.Contains(finalTextLower, "data"),
		"Final response should reference the fetched data or source. Got: %s", finalText)

	t.Logf("Final LLM response: %s", finalText)
}

// mockTool is a simple mock tool for testing ExecuteAll.
type mockTool struct {
	name     string
	delay    time.Duration
	execFunc func(ctx context.Context, args json.RawMessage) (json.RawMessage, error)
}

func (m *mockTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        m.name,
		Description: "Mock tool for testing",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}}}`),
	}
}

func (*mockTool) IsAsynchronous() bool { return false }

func (m *mockTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if m.execFunc != nil {
		return m.execFunc(ctx, args)
	}

	return json.Marshal(map[string]string{"result": "success"})
}

// TestRegistry_ExecuteAll tests batch execution functionality.
func TestRegistry_ExecuteAll(t *testing.T) {
	t.Parallel()
	// Setup tools once for table-driven tests
	successfulTool := &mockTool{name: "successful-tool"}
	failingTool := &mockTool{
		name: "failing-tool",
		execFunc: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
			return nil, errors.New("simulated failure")
		},
	}

	registry := tool.NewRegistry(tool.RegistryConfig{})
	require.NoError(t, registry.Register(successfulTool))
	require.NoError(t, registry.Register(failingTool))

	testCases := []struct {
		name      string
		requests  []*llm.ToolRequest
		assertion func(t *testing.T, results []*llm.ToolResponse)
	}{
		{
			name:     "empty nil slice",
			requests: nil,
			assertion: func(t *testing.T, _ []*llm.ToolResponse) {
				t.Helper()
			},
		},
		{
			name:     "empty slice",
			requests: []*llm.ToolRequest{},
			assertion: func(t *testing.T, _ []*llm.ToolResponse) {
				t.Helper()
			},
		},
		{
			name: "single request",
			requests: []*llm.ToolRequest{
				{ID: "req-1", Name: "successful-tool", Arguments: json.RawMessage(`{"input":"test"}`)},
			},
			assertion: func(t *testing.T, results []*llm.ToolResponse) {
				t.Helper()
				assert.Equal(t, "req-1", results[0].ID)
				assert.Equal(t, "successful-tool", results[0].Name)
				assert.Empty(t, results[0].Error)
				assert.NotNil(t, results[0].Result)
			},
		},
		{
			name: "multiple requests preserve order",
			requests: []*llm.ToolRequest{
				{ID: "req-1", Name: "successful-tool"},
				{ID: "req-2", Name: "successful-tool"},
				{ID: "req-3", Name: "successful-tool"},
			},
			assertion: func(t *testing.T, results []*llm.ToolResponse) {
				t.Helper()

				for i, result := range results {
					assert.Equal(t, fmt.Sprintf("req-%d", i+1), result.ID, "Order should be preserved")
					assert.Empty(t, result.Error)
				}
			},
		},
		{
			name: "partial failures",
			requests: []*llm.ToolRequest{
				{ID: "req-ok", Name: "successful-tool"},
				{ID: "req-fail", Name: "failing-tool"},
			},
			assertion: func(t *testing.T, results []*llm.ToolResponse) {
				t.Helper()
				assert.Empty(t, results[0].Error)
				assert.Contains(t, results[1].Error, "simulated failure")
			},
		},
		{
			name: "nil request in slice",
			requests: []*llm.ToolRequest{
				{ID: "req-1", Name: "successful-tool"},
				nil,
			},
			assertion: func(t *testing.T, results []*llm.ToolResponse) {
				t.Helper()
				assert.Empty(t, results[0].Error)
				assert.Contains(t, results[1].Error, tool.ErrToolRequestNil.Error())
			},
		},
		{
			name: "nonexistent tool",
			requests: []*llm.ToolRequest{
				{ID: "req-1", Name: "nonexistent-tool"},
			},
			assertion: func(t *testing.T, results []*llm.ToolResponse) {
				t.Helper()
				assert.Contains(t, results[0].Error, tool.ErrToolNotFound.Error())
				assert.Equal(t, "req-1", results[0].ID)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			results := registry.ExecuteAll(context.Background(), tc.requests)
			tc.assertion(t, results)
		})
	}

	t.Run("concurrency limit", func(t *testing.T) {
		t.Parallel()

		registry := tool.NewRegistry(tool.RegistryConfig{})

		var currentConcurrent, maxConcurrent atomic.Int32

		mockTool := &mockTool{
			name:  "test-tool",
			delay: 50 * time.Millisecond,
			execFunc: func(_ context.Context, _ json.RawMessage) (json.RawMessage, error) {
				current := currentConcurrent.Add(1)
				defer currentConcurrent.Add(-1)

				// Track max concurrency
				for {
					maxVal := maxConcurrent.Load()
					if current <= maxVal || maxConcurrent.CompareAndSwap(maxVal, current) {
						break
					}
				}

				time.Sleep(50 * time.Millisecond)

				return json.Marshal(map[string]string{"result": "success"})
			},
		}
		require.NoError(t, registry.Register(mockTool))

		// Create 10 requests
		requests := make([]*llm.ToolRequest, 10)
		for i := range requests {
			requests[i] = &llm.ToolRequest{
				ID:   fmt.Sprintf("req-%d", i),
				Name: "test-tool",
			}
		}

		results := registry.ExecuteAll(
			context.Background(),
			requests,
			tool.WithMaxConcurrency(3),
		)
		require.Len(t, results, 10)

		assert.LessOrEqual(t, maxConcurrent.Load(), int32(3), "Should respect concurrency limit")
		assert.Positive(t, maxConcurrent.Load(), "At least one tool should have run")
	})

	t.Run("context cancellation", func(t *testing.T) {
		t.Parallel()

		registry := tool.NewRegistry(tool.RegistryConfig{})

		mockTool := &mockTool{
			name:  "test-tool",
			delay: 200 * time.Millisecond,
		}
		require.NoError(t, registry.Register(mockTool))

		// Create 5 requests
		requests := make([]*llm.ToolRequest, 5)
		for i := range requests {
			requests[i] = &llm.ToolRequest{
				ID:        fmt.Sprintf("req-%d", i),
				Name:      "test-tool",
				Arguments: json.RawMessage(`{}`),
			}
		}

		// Cancel context after 100ms
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		results := registry.ExecuteAll(ctx, requests)

		// ExecuteAll always returns len(reqs) responses, even on cancellation
		require.Len(t, results, 5)

		// All results should have errors (either execution errors or cancellation errors)
		// Context cancellation errors are encoded in ToolResponse.Error, not as a top-level error
		for _, result := range results {
			assert.NotEmpty(t, result.Error, "All requests should have errors due to cancellation")
		}
	})
}

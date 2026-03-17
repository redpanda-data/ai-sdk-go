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

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/runner"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

func main() {
	// Check for required environment variables
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	// Get the mode from environment variable (default to "interactive")
	mode := os.Getenv("HOOKS_MODE")
	if mode == "" {
		mode = "interactive"
	}

	fmt.Println("=================================================================")
	fmt.Println("  Agent Interceptors Example - Demonstrating the Interceptor System")
	fmt.Println("=================================================================")
	fmt.Println()
	fmt.Println("This example demonstrates the interceptor system with:")
	fmt.Println("  • ObservabilityInterceptor - Tracks metrics and performance")
	fmt.Println("  • TurnTrackerInterceptor - Monitors agent loop turns")
	fmt.Println("  • ToolLoggingInterceptor - Logs tool execution details")
	fmt.Println("  • ToolApprovalInterceptor - Requires approval before tool execution")
	fmt.Println()

	if mode == "auto" {
		fmt.Println("Running in AUTO mode (tool approval bypassed)")
	} else {
		fmt.Println("Running in INTERACTIVE mode (tool approval required)")
		fmt.Println("Set HOOKS_MODE=auto to bypass approval prompts")
	}

	fmt.Println("=================================================================")
	fmt.Println()

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		log.Fatalf("Failed to create provider: %v", err)
	}

	// Create model - using GPT-5
	model, err := provider.NewModel(openai.ModelGPT5)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create tool registry
	registry := tool.NewRegistry(tool.RegistryConfig{})

	// Create runtime secrets that the model cannot know
	secrets := map[string]string{
		"api_key":           "sk-proj-abc123xyz789-secret-key-42",
		"database_password": "P@ssw0rd!2024#Secret",
		"magic_number":      "42",
	}

	// Register our example tools
	tools := []tool.Tool{
		NewTemperatureSensorTool(),
		NewGetSecretValueTool(secrets),
	}

	for _, t := range tools {
		if err := registry.Register(t); err != nil {
			log.Fatalf("Failed to register tool: %v", err)
		}
	}

	// Create interceptors
	observabilityInterceptor := NewObservabilityInterceptor()
	turnTrackerInterceptor := NewTurnTrackerInterceptor(0) // 0 means no custom limit
	toolLoggingInterceptor := NewToolLoggingInterceptor()

	// Tool approval interceptor - require approval for all tools in interactive mode
	autoApprove := mode == "auto"
	toolApprovalInterceptor := NewToolApprovalInterceptor(
		[]string{}, // Empty list means all tools require approval
		autoApprove,
	)

	// Create agent with interceptors
	// All interceptors are configured on the agent. The runner is a thin orchestration layer.
	systemPrompt := `You are a helpful assistant with access to tools.
You have access to a temperature sensor and a secure secrets store.
When the user asks for sensor readings or secret values, you MUST use the available tools.
Without the tools, you cannot access this information - do not make up values.
Be concise and helpful in your responses.`

	ag, err := llmagent.New(
		"assistant",
		systemPrompt,
		model,
		llmagent.WithTools(registry),
		llmagent.WithMaxTurns(10),
		llmagent.WithInterceptors(
			observabilityInterceptor, // Implements ModelInterceptor
			turnTrackerInterceptor,   // Implements TurnInterceptor
			toolApprovalInterceptor,  // Implements ToolInterceptor (prompts for approval) - must be before logging
			toolLoggingInterceptor,   // Implements ToolInterceptor - logs after approval
		),
	)
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create session store (in-memory for this example)
	sessionStore := session.NewInMemoryStore()

	// Create runner (no interceptors needed - agent handles all interceptors)
	r, err := runner.New(ag, sessionStore)
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Run the agent with a test query
	ctx := context.Background()
	userMessage := llm.NewMessage(llm.RoleUser, llm.NewTextPart("What's the current temperature reading from the sensor, and what's the 'api_key' secret value?"))

	fmt.Println("User: What's the current temperature reading from the sensor, and what's the 'api_key' secret value?")
	fmt.Println()
	fmt.Println("Agent response:")
	fmt.Println(strings.Repeat("-", 70))

	// Execute and stream events
	sessionID := "example-session"
	userID := "example-user"

	var finalMessage string
	for evt, err := range r.Run(ctx, userID, sessionID, userMessage) {
		if err != nil {
			log.Fatalf("Error during execution: %v", err)
		}

		// Extract final response from MessageEvent
		switch e := evt.(type) {
		case agent.MessageEvent:
			finalMessage = e.Response.Message.TextContent()
		}
	}

	fmt.Println()
	if finalMessage != "" {
		fmt.Println("Final Response:")
		fmt.Println(strings.Repeat("-", 70))
		fmt.Println(finalMessage)
	}

	fmt.Println()
	fmt.Println("=================================================================")
	fmt.Println("Example completed! Check the logs above to see how interceptors work.")
	fmt.Println("=================================================================")
}

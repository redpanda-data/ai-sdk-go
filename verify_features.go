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

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/llm/fakellm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func main() {
	// 1. Setup a Mock Model to capture the output
	// This allows us to inspect exactly what the agent sends to the LLM.
	fake := fakellm.NewFakeModel()
	fake.When(fakellm.Any()).ThenRespondText("Acknowledged. I have processed the request.")

	// 2. Create an agent with placeholders in the system prompt
	// - {user_name} and {env} will be pulled from metadata.
	// - {current_date} is a built-in variable provided by the SDK.
	systemPrompt := "Hello {user_name}! You are in the {env} environment. Today is {current_date}."
	myAgent, err := llmagent.New("demo-agent", systemPrompt, fake)
	if err != nil {
		panic(fmt.Sprintf("failed to create agent: %v", err))
	}

	// 3. Setup Session Metadata (The data source for templating)
	sess := &session.State{
		ID: "session-123",
		Metadata: map[string]any{
			"user_name": "Felipe",
			"env":       "Production",
		},
	}
	// Add a dummy user message to initiate the execution turn
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("Status check.")))

	// 4. Set Global Instructions via Context
	// This propagates to all agents in the invocation tree (including tools).
	ctx := context.Background()
	globalInstr := "CRITICAL: The response must be professional and under 10 words."
	gctx := agent.ContextWithGlobalInstructions(ctx, globalInstr)

	// 5. Run the Agent
	inv := agent.NewInvocationMetadata(sess, myAgent.Info())
	fmt.Println("=== Running Agent Execution ===")
	for evt, err := range myAgent.Run(gctx, inv) {
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			return
		}
		if msg, ok := evt.(agent.MessageEvent); ok {
			fmt.Printf("Assistant: %s\n", msg.Response.Message.TextContent())
		}
	}

	// 6. Inspect what the LLM actually received as the System Prompt
	// We retrieve this from the fake model's call history.
	calls := fake.Calls()
	if len(calls) == 0 {
		fmt.Println("No calls recorded by the model.")
		return
	}

	var renderedPrompt string
	for _, m := range calls[0].Request.Messages {
		if m.Role == llm.RoleSystem {
			renderedPrompt = m.TextContent()
			break
		}
	}

	fmt.Println("\n=== Final Rendered System Prompt ===")
	fmt.Println(renderedPrompt)
	fmt.Println("====================================")
}

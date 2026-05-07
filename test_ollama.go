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
	"github.com/redpanda-data/ai-sdk-go/providers/openaicompat"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func main() {
	ctx := context.Background()

	// 1. Setup Ollama Provider (OpenAI Compatible)
	// Ollama doesn't require a real API key, but the SDK needs a non-empty string.
	// We point it to the default local Ollama port 11434.
	provider, err := openaicompat.NewProvider("ollama", 
		openaicompat.WithBaseURL("http://localhost:11434/v1"),
	)
	if err != nil {
		panic(fmt.Sprintf("failed to create provider: %v", err))
	}

	// 2. Initialize the Model (using llama3)
	model, err := provider.NewModel("llama3")
	if err != nil {
		panic(fmt.Sprintf("failed to create model: %v", err))
	}

	// 3. Create Agent with placeholders in the system prompt
	prompt := "You are a helpful assistant. Hello {user_name}! Today's date is {current_date}."
	myAgent, err := llmagent.New("local-agent", prompt, model)
	if err != nil {
		panic(fmt.Sprintf("failed to create agent: %v", err))
	}

	// 4. Setup Session with Metadata for templating
	sess := &session.State{
		Metadata: map[string]any{
			"user_name": "Felipe",
		},
	}
	sess.Messages = append(sess.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("Tell me whats the day today accornding to your instructions and a fun fact about Go.")))

	// 5. Add Global Instructions via Context
	gctx := agent.ContextWithGlobalInstructions(ctx, "Keep the response extremely short and professional.")

	// 6. Run the Agent
	inv := agent.NewInvocationMetadata(sess, myAgent.Info())
	fmt.Println("--- Running Agent with Ollama ---")
	fmt.Printf("Metadata being sent: user_name=%s\n",
     sess.Metadata["user_name"])
	for evt, err := range myAgent.Run(gctx, inv) {
		if err != nil {
			fmt.Printf("\nError during execution: %v\n", err)
			return
		}
		
		switch e := evt.(type) {
		case agent.AssistantDeltaEvent:
			// Print streaming token deltas
			if e.Delta.Part.IsText() {
				fmt.Print(e.Delta.Part.Text)
			}
		case agent.InvocationEndEvent:
			fmt.Println("\n--- Execution Finished ---")
			fmt.Printf("Finish Reason: %v\n", e.FinishReason)
		}
	}
}

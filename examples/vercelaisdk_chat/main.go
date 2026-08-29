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

// Chat server for the Vercel AI SDK example: an llmagent with one tool,
// exposed over the UI Message Stream protocol with server-side sessions.
// The React client in ./web talks to it through the Vite dev proxy.
//
//	OPENAI_API_KEY=... go run .
package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/redpanda-data/ai-sdk-go/adapter/vercelaisdk/uimessagestream"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// weatherTool returns canned weather so the example needs no extra API key.
type weatherTool struct{}

func (weatherTool) Definition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Name:        "getWeather",
		Description: "Get the current weather for a city.",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
	}
}

func (weatherTool) Execute(_ context.Context, args json.RawMessage) (json.RawMessage, error) {
	var in struct {
		City string `json:"city"`
	}

	_ = json.Unmarshal(args, &in)

	out, err := json.Marshal(map[string]string{
		"city":       in.City,
		"temp":       "22C",
		"conditions": "sunny",
		"observedAt": time.Now().UTC().Format(time.RFC3339),
	})
	if err != nil {
		return nil, err
	}

	return out, nil
}

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}

	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		log.Fatal(err)
	}

	model, err := provider.NewModel(openai.ModelGPT5Mini)
	if err != nil {
		log.Fatal(err)
	}

	reg := tool.NewRegistry(tool.RegistryConfig{})
	if err := reg.Register(weatherTool{}); err != nil {
		log.Fatal(err)
	}

	ag, err := llmagent.New("assistant",
		"You are a helpful assistant. Use the getWeather tool whenever the user asks about weather.",
		model, llmagent.WithTools(reg))
	if err != nil {
		log.Fatal(err)
	}

	// In-memory store: chats survive page reloads but not a server restart.
	// Swap in a persistent session.Store implementation for real deployments.
	chat := uimessagestream.Handler(ag, session.NewInMemoryStore())

	mux := http.NewServeMux()
	mux.Handle("/api/chat", http.StripPrefix("/api/chat", chat))
	mux.Handle("/api/chat/", http.StripPrefix("/api/chat", chat))

	addr := "127.0.0.1:8080"
	log.Printf("chat server listening on http://%s (POST/GET /api/chat)", addr)
	log.Fatal(http.ListenAndServe(addr, mux)) //nolint:gosec // example server, no timeouts needed
}

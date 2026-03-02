package main

import (
	"context"
	"log"
	"os"

	"github.com/redpanda-data/ai-sdk-go/agentpack"
)

func main() {
	configPath := os.Getenv("AGENT_CONFIG")
	if configPath == "" {
		configPath = "agent.yaml"
	}
	if err := agentpack.Run(context.Background(), configPath); err != nil {
		log.Fatal(err)
	}
}

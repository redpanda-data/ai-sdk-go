package agentpack

import (
	"bufio"
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/google/uuid"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/runner"
)

// runCLI starts a stdin/stdout REPL loop.
func runCLI(ctx context.Context, r *runner.Runner, logger *slog.Logger) error {
	sessionID := uuid.New().String()
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("Agent ready. Type your message (Ctrl+D to quit):")
	fmt.Print("> ")

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			fmt.Print("> ")
			continue
		}

		msg := llm.NewMessage(llm.RoleUser, llm.NewTextPart(line))
		events := r.Run(ctx, "", sessionID, msg)

		var streamed bool
		for event, err := range events {
			if err != nil {
				logger.Error("Agent error", "error", err)
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				break
			}

			switch ev := event.(type) {
			case agent.MessageEvent:
				text := ev.Response.TextContent()
				if text != "" && !streamed {
					fmt.Println(text)
				}
			case agent.AssistantDeltaEvent:
				if ev.Delta.Part != nil && ev.Delta.Part.IsText() {
					fmt.Print(ev.Delta.Part.Text)
					streamed = true
				}
			case agent.InvocationEndEvent:
				if streamed {
					fmt.Println()
				}
			}
		}

		fmt.Print("> ")
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read stdin: %w", err)
	}

	fmt.Println()
	return nil
}

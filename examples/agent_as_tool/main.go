package main

import (
	"context"
	"fmt"
	"iter"
	"log"
	"os"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/agent/llmagent"
	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/providers/openai"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/tool"
	"github.com/redpanda-data/ai-sdk-go/tool/agenttool"
	"github.com/redpanda-data/ai-sdk-go/tool/builtin/webfetch"
)

// usageTracker tracks token usage for an agent.
type usageTracker struct {
	name        string
	totalTokens int
}

// InterceptModel implements agent.ModelInterceptor to track token usage.
func (u *usageTracker) InterceptModel(
	ctx context.Context,
	info *agent.ModelCallInfo,
	next agent.ModelCallHandler,
) agent.ModelCallHandler {
	return &usageModelHandler{
		tracker: u,
		next:    next,
	}
}

type usageModelHandler struct {
	tracker *usageTracker
	next    agent.ModelCallHandler
}

func (h *usageModelHandler) Generate(ctx context.Context, req *llm.Request) (*llm.Response, error) {
	resp, err := h.next.Generate(ctx, req)
	if err == nil && resp.Usage != nil {
		h.tracker.totalTokens += resp.Usage.TotalTokens
		fmt.Printf("  [%s Usage] +%d tokens (cumulative: %d)\n",
			h.tracker.name, resp.Usage.TotalTokens, h.tracker.totalTokens)
	}
	return resp, err
}

func (h *usageModelHandler) GenerateEvents(ctx context.Context, req *llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		var turnUsage int
		for evt, err := range h.next.GenerateEvents(ctx, req) {
			if endEvt, ok := evt.(llm.StreamEndEvent); ok && endEvt.Response != nil && endEvt.Response.Usage != nil {
				turnUsage = endEvt.Response.Usage.TotalTokens
			}
			if !yield(evt, err) {
				return
			}
		}
		if turnUsage > 0 {
			h.tracker.totalTokens += turnUsage
			fmt.Printf("  [%s Usage] +%d tokens (cumulative: %d)\n",
				h.tracker.name, turnUsage, h.tracker.totalTokens)
		}
	}
}

// assistantToolLogger logs tool calls made by the nested assistant agent.
type assistantToolLogger struct{}

// InterceptToolExecution implements agent.ToolInterceptor.
func (a *assistantToolLogger) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponse, error) {
	fmt.Printf("  [Assistant Tool Call] %s\n", info.Req.Name)
	if len(info.Req.Arguments) > 0 && len(info.Req.Arguments) < 200 {
		fmt.Printf("  [Assistant Tool Args] %s\n", string(info.Req.Arguments))
	}

	resp, err := next(ctx, info)

	if err != nil {
		fmt.Printf("  [Assistant Tool Error] %v\n", err)
	} else if resp.Error != "" {
		fmt.Printf("  [Assistant Tool Error] %s\n", resp.Error)
	} else {
		fmt.Printf("  [Assistant Tool Success] %s\n", info.Req.Name)
	}

	return resp, err
}

func main() {
	ctx := context.Background()

	// Get API key from environment
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable not set")
	}

	// Create OpenAI provider
	provider, err := openai.NewProvider(apiKey)
	if err != nil {
		log.Fatalf("failed to create provider: %v", err)
	}

	// Create model
	model, err := provider.NewModel(openai.ModelGPT5_1)
	if err != nil {
		log.Fatalf("failed to create model: %v", err)
	}

	// Create a general-purpose assistant agent with tools
	// This agent can be delegated subtasks for better context management
	assistantTools := tool.NewRegistry(tool.RegistryConfig{})
	if err := assistantTools.Register(webfetch.New()); err != nil {
		log.Fatalf("failed to register webfetch tool: %v", err)
	}

	// Track token usage for assistant agent
	assistantUsage := &usageTracker{name: "Assistant"}
	toolLogger := &assistantToolLogger{}

	assistantAgent, err := llmagent.New(
		"assistant",
		`You are a helpful assistant that can use tools to complete tasks.
When given a task, use available tools as needed and provide a clear, concise response.`,
		model,
		llmagent.WithTools(assistantTools),
		llmagent.WithInterceptors(assistantUsage, toolLogger),
	)
	if err != nil {
		log.Fatalf("failed to create assistant agent: %v", err)
	}

	// Create main agent that can delegate to the assistant for context isolation
	// This enables the main agent to offload subtasks without polluting its context
	mainTools := tool.NewRegistry(tool.RegistryConfig{})
	// Configure a longer timeout (8 minutes) since agent execution can include:
	// - Multiple LLM calls for reasoning and response formatting
	// - Multiple tool calls (like webfetch) with their own timeouts
	// - Complex research or data gathering tasks
	if err := mainTools.Register(agenttool.New(assistantAgent), tool.WithTimeout(8*time.Minute)); err != nil {
		log.Fatalf("failed to register assistant agent: %v", err)
	}

	// Track token usage for main agent
	mainUsage := &usageTracker{name: "Main"}

	mainAgent, err := llmagent.New(
		"main",
		`You are a helpful assistant that can delegate subtasks to a specialized assistant.

When you need to:
- Fetch information from the web
- Perform a complex subtask that would clutter your context
- Isolate work that doesn't need your full conversation history

Use the 'assistant' tool by passing it a clear task description in JSON format like:
{"task": "search for X and summarize the key points"}

The assistant has access to web search and will return a focused response.
This helps keep your context clean and focused on the main task.`,
		model,
		llmagent.WithTools(mainTools),
		llmagent.WithInterceptors(mainUsage),
	)
	if err != nil {
		log.Fatalf("failed to create main agent: %v", err)
	}

	// Create a session for the conversation
	sess := &session.State{
		ID:       "example-session",
		Messages: []llm.Message{},
		Metadata: map[string]any{},
	}

	// Add a user message that requires web research
	userMessage := llm.NewMessage(
		llm.RoleUser,
		llm.NewTextPart("What are the latest developments at Redpanda Data? I need a brief summary."),
	)
	sess.Messages = append(sess.Messages, userMessage)

	// Create invocation metadata
	inv := agent.NewInvocationMetadata(sess, agent.Info{
		Name:        mainAgent.Name(),
		Description: mainAgent.Description(),
	})

	// Run the main agent
	fmt.Println("Agent-as-Tool Example: Context Management Pattern")
	fmt.Println("=================================================")
	fmt.Println()
	fmt.Println("User:", userMessage.TextContent())
	fmt.Println()

	for evt, err := range mainAgent.Run(ctx, inv) {
		if err != nil {
			log.Fatalf("agent error: %v", err)
		}

		switch e := evt.(type) {
		case agent.StatusEvent:
			fmt.Printf("[Status] %s\n", e.Stage)

		case agent.ToolRequestEvent:
			if e.Request.Name == "assistant" {
				fmt.Printf("[Delegating to assistant] Sub-task isolated in fresh context\n")
			} else {
				fmt.Printf("[Tool Call] %s\n", e.Request.Name)
			}

		case agent.ToolResponseEvent:
			if e.Response.Error != "" {
				fmt.Printf("[Error] %s\n", e.Response.Error)
			}

		case agent.MessageEvent:
			fmt.Printf("\nAssistant: %s\n", e.Response.Message.TextContent())

		case agent.InvocationEndEvent:
			fmt.Printf("\n[Done] Finish reason: %s\n", e.FinishReason)
			if e.Usage != nil {
				fmt.Printf("Total tokens used: %d\n", e.Usage.TotalTokens)
			}
		}
	}

	// Print token usage summary
	fmt.Println()
	fmt.Println("=================================================")
	fmt.Println("Token Usage Summary:")
	fmt.Printf("  Main Agent:      %d tokens\n", mainUsage.totalTokens)
	fmt.Printf("  Assistant Agent: %d tokens\n", assistantUsage.totalTokens)
	fmt.Printf("  Total:           %d tokens\n", mainUsage.totalTokens+assistantUsage.totalTokens)
	fmt.Println()
	fmt.Println("Context Isolation Benefit:")
	fmt.Println("  The assistant agent ran with a fresh context,")
	fmt.Println("  keeping the main agent's context clean and focused.")
	fmt.Printf("  Assistant handled %d tokens of work independently.\n", assistantUsage.totalTokens)
}

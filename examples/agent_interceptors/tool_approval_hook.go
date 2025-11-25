package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ToolApprovalInterceptor demonstrates ToolInterceptor with user approval.
// It requires manual approval before executing certain tools, providing
// a human-in-the-loop pattern for sensitive operations.
//
// Use cases:
// - Requiring approval for destructive operations (delete, update)
// - Security/authorization checks
// - Compliance workflows requiring human oversight
// - Testing and debugging (manual tool execution control)
//
// This interceptor demonstrates how to implement tool authorization patterns
// that are common in production AI agents.
type ToolApprovalInterceptor struct {
	// requireApproval determines if approval is needed for specific tools
	requireApproval map[string]bool

	// autoApprove can be set to bypass approval (useful for testing)
	autoApprove bool

	// mu serializes approval prompts when tools execute concurrently
	mu sync.Mutex
}

// NewToolApprovalInterceptor creates an interceptor that requires approval for specified tools.
// If requireApproval is empty, all tools require approval.
// Set autoApprove to true to automatically approve all tools (useful for CI/testing).
func NewToolApprovalInterceptor(requireApproval []string, autoApprove bool) *ToolApprovalInterceptor {
	approvalMap := make(map[string]bool)
	for _, tool := range requireApproval {
		approvalMap[tool] = true
	}

	return &ToolApprovalInterceptor{
		requireApproval: approvalMap,
		autoApprove:     autoApprove,
	}
}

// InterceptToolExecution implements agent.ToolInterceptor.
// It prompts for approval before executing tools that require it.
func (h *ToolApprovalInterceptor) InterceptToolExecution(
	ctx context.Context,
	inv *agent.InvocationMetadata,
	req *llm.ToolRequest,
	next agent.ToolExecutionNext,
) (*llm.ToolResponse, error) {
	// Check if this tool requires approval
	needsApproval := len(h.requireApproval) == 0 || h.requireApproval[req.Name]

	if !needsApproval {
		// No approval needed, execute directly
		return next(ctx, inv, req)
	}

	// Auto-approve if configured (useful for CI/testing)
	if h.autoApprove {
		log.Printf("[ToolApproval] Auto-approved tool %q", req.Name)
		return next(ctx, inv, req)
	}

	// Serialize approval prompts when tools execute concurrently
	// This prevents interleaved output when multiple tools need approval
	h.mu.Lock()
	defer h.mu.Unlock()

	// Display tool execution request
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Printf("[Tool Approval Required]\n")
	fmt.Printf("Tool: %s\n", req.Name)
	fmt.Printf("Arguments:\n")

	// Pretty print the arguments
	var args map[string]any
	if err := json.Unmarshal(req.Arguments, &args); err == nil {
		prettyArgs, _ := json.MarshalIndent(args, "  ", "  ")
		fmt.Printf("  %s\n", string(prettyArgs))
	} else {
		fmt.Printf("  %s\n", string(req.Arguments))
	}

	fmt.Println(strings.Repeat("=", 70))

	// Prompt for approval
	approved := h.promptForApproval()

	if !approved {
		log.Printf("[ToolApproval] Tool %q execution denied by user", req.Name)

		// Return a response indicating the tool was denied
		return &llm.ToolResponse{
			ID:    req.ID,
			Name:  req.Name,
			Error: "Tool execution denied by user",
		}, nil
	}

	log.Printf("[ToolApproval] Tool %q approved by user, executing...", req.Name)
	return next(ctx, inv, req)
}

// promptForApproval prompts the user for approval and returns their decision.
func (h *ToolApprovalInterceptor) promptForApproval() bool {
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\nApprove this tool execution? [y/n]: ")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Printf("Error reading input: %v", err)
			return false
		}

		input = strings.TrimSpace(strings.ToLower(input))

		switch input {
		case "y", "yes":
			return true
		case "n", "no":
			return false
		default:
			fmt.Println("Invalid input. Please enter 'y' for yes or 'n' for no.")
		}
	}
}

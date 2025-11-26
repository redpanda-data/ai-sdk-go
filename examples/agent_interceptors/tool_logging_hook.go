package main

import (
	"context"
	"log"
	"time"

	"github.com/redpanda-data/ai-sdk-go/agent"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ToolLoggingInterceptor demonstrates ToolInterceptor.
// It logs all tool executions with timing, arguments, and results.
//
// Use cases:
// - Audit logging for tool calls
// - Performance monitoring
// - Debugging tool execution
type ToolLoggingInterceptor struct{}

// NewToolLoggingInterceptor creates a new tool logging interceptor.
func NewToolLoggingInterceptor() *ToolLoggingInterceptor {
	return &ToolLoggingInterceptor{}
}

// InterceptToolExecution implements agent.ToolInterceptor.
// It logs tool execution details including timing and results.
func (h *ToolLoggingInterceptor) InterceptToolExecution(
	ctx context.Context,
	info *agent.ToolCallInfo,
	next agent.ToolExecutionNext,
) (*llm.ToolResponse, error) {
	inv := info.Inv
	req := info.Req

	start := time.Now()
	log.Printf("[ToolLogging][Turn %d] Tool %q execution started", inv.Turn(), req.Name)
	log.Printf("[ToolLogging] Arguments: %s", string(req.Arguments))

	resp, err := next(ctx, info)

	duration := time.Since(start)
	if err != nil {
		log.Printf("[ToolLogging] Tool %q failed after %v: %v", req.Name, duration, err)
		return resp, err
	}

	if resp.Error != "" {
		log.Printf("[ToolLogging] Tool %q returned error after %v: %s",
			req.Name, duration, resp.Error)
	} else {
		log.Printf("[ToolLogging] Tool %q completed successfully in %v",
			req.Name, duration)
		log.Printf("[ToolLogging] Result: %s", string(resp.Result))
	}

	return resp, nil
}

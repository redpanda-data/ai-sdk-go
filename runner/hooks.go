package runner

import (
	"fmt"

	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// executeBeforeInvocationHooks runs all registered BeforeInvocation hooks.
//
// Hooks execute in registration order. The first hook to return an error
// stops execution and returns that error.
//
// Returns error if any hook returns an error.
func (r *Runner) executeBeforeInvocationHooks(
	ctx hooks.HookContext,
	userMessage llm.Message,
) error {
	for _, h := range r.config.hooks {
		if hook, ok := h.(hooks.HookBeforeInvocation); ok {
			if err := hook.OnBeforeInvocation(ctx, userMessage); err != nil {
				return fmt.Errorf("before invocation hook failed: %w", err)
			}
		}
	}

	return nil
}

// executeAfterInvocationHooks runs all registered AfterInvocation hooks.
//
// These are observe-only hooks for metrics, logging, etc. They cannot modify
// the invocation result.
//
// Hooks execute in registration order. If any hook returns an error,
// execution stops and returns that error.
func (r *Runner) executeAfterInvocationHooks(
	ctx hooks.HookContext,
	result hooks.InvocationResult,
) error {
	for _, h := range r.config.hooks {
		if hook, ok := h.(hooks.HookAfterInvocation); ok {
			if err := hook.OnAfterInvocation(ctx, result); err != nil {
				return fmt.Errorf("after invocation hook failed: %w", err)
			}
		}
	}

	return nil
}

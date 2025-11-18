package hooks_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Test hook types for validation.
type emptyHook struct{}

type beforeInvocationHook struct{}

func (*beforeInvocationHook) OnBeforeInvocation(ctx hooks.HookContext, msg llm.Message) error {
	return nil
}

type afterInvocationHook struct{}

func (*afterInvocationHook) OnAfterInvocation(ctx hooks.HookContext, result hooks.InvocationResult) error {
	return nil
}

type multiHook struct{}

func (*multiHook) OnBeforeInvocation(ctx hooks.HookContext, msg llm.Message) error {
	return nil
}

func (*multiHook) OnAfterInvocation(ctx hooks.HookContext, result hooks.InvocationResult) error {
	return nil
}

func TestImplementsAnyHook(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		hook hooks.Hook
		want bool
	}{
		{
			name: "empty hook implements no interfaces",
			hook: &emptyHook{},
			want: false,
		},
		{
			name: "hook implements BeforeInvocation",
			hook: &beforeInvocationHook{},
			want: true,
		},
		{
			name: "hook implements AfterInvocation",
			hook: &afterInvocationHook{},
			want: true,
		},
		{
			name: "hook implements multiple interfaces",
			hook: &multiHook{},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := hooks.ImplementsAnyHook(tt.hook)
			assert.Equal(t, tt.want, got)
		})
	}
}

package hooks_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"github.com/redpanda-data/ai-sdk-go/agent/hooks"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

func TestHookContext_Metadata(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		fn   func(t *testing.T)
	}{
		{
			name: "set and retrieve metadata",
			fn: func(t *testing.T) {
				t.Helper()

				ctx := hooks.NewHookContext(
					context.Background(),
					"inv-123",
					"session-123",
					0,
					time.Now(),
					&session.State{},
				)

				// Set metadata
				ctx.SetMetadata("key1", "value1")
				ctx.SetMetadata("key2", 42)

				// Retrieve metadata
				metadata := ctx.Metadata()
				assert.Equal(t, "value1", metadata["key1"])
				assert.Equal(t, 42, metadata["key2"])
			},
		},
		{
			name: "metadata isolated between contexts",
			fn: func(t *testing.T) {
				t.Helper()

				ctx1 := hooks.NewHookContext(
					context.Background(),
					"inv-1",
					"session-1",
					0,
					time.Now(),
					&session.State{},
				)

				ctx2 := hooks.NewHookContext(
					context.Background(),
					"inv-2",
					"session-2",
					0,
					time.Now(),
					&session.State{},
				)

				// Set different metadata in each context
				ctx1.SetMetadata("key", "value1")
				ctx2.SetMetadata("key", "value2")

				// Verify isolation
				assert.Equal(t, "value1", ctx1.Metadata()["key"])
				assert.Equal(t, "value2", ctx2.Metadata()["key"])
			},
		},
		{
			name: "overwrite existing metadata",
			fn: func(t *testing.T) {
				t.Helper()

				ctx := hooks.NewHookContext(
					context.Background(),
					"inv-123",
					"session-123",
					0,
					time.Now(),
					&session.State{},
				)

				// Set initial value
				ctx.SetMetadata("key", "initial")
				assert.Equal(t, "initial", ctx.Metadata()["key"])

				// Overwrite
				ctx.SetMetadata("key", "updated")
				assert.Equal(t, "updated", ctx.Metadata()["key"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.fn(t)
		})
	}
}

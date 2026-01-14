package kvstore_test

import (
	"context"
	"testing"

	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
)

// redpandaLowMemory returns options for running Redpanda with reduced memory footprint.
// This helps avoid OOM kills in CI environments with limited resources.
func redpandaLowMemory() testcontainers.CustomizeRequestOption {
	return testcontainers.WithCmd(
		"redpanda", "start",
		"--mode=dev-container",
		"--smp=1",
		"--memory=512M",
		"--reserve-memory=0M",
	)
}

// requireRedpandaContainer starts a Redpanda container and logs its output on failure.
func requireRedpandaContainer(ctx context.Context, t *testing.T) *redpanda.Container {
	t.Helper()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest",
		redpandaLowMemory(),
		redpanda.WithAutoCreateTopics(),
	)
	if err != nil {
		if container != nil {
			if logs, logErr := container.Logs(ctx); logErr == nil {
				buf := make([]byte, 4096)
				n, _ := logs.Read(buf)
				t.Logf("Container logs:\n%s", string(buf[:n]))
				logs.Close()
			}

			_ = container.Terminate(ctx)
		}

		t.Fatalf("Failed to start Redpanda container: %v", err)
	}

	return container
}

func TestKVStore_LoadSaveDelete(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container := requireRedpandaContainer(ctx, t)
	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	store, err := kvstore.NewKVStore(ctx, "test-sessions",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Load non-existent session
	_, err = store.Load(ctx, "nonexistent")
	require.ErrorIs(t, err, session.ErrNotFound)

	// Save a session
	state := &session.State{
		ID: "session-1",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Hello")),
			llm.NewMessage(llm.RoleAssistant, llm.NewTextPart("Hi there")),
		},
		Metadata: map[string]any{
			"user_id": "user-123",
		},
	}
	err = store.Save(ctx, state)
	require.NoError(t, err)

	// Load the session back
	loaded, err := store.Load(ctx, "session-1")
	require.NoError(t, err)
	assert.Equal(t, state.ID, loaded.ID)
	assert.Len(t, loaded.Messages, 2)
	assert.Equal(t, "Hello", loaded.Messages[0].TextContent())
	assert.Equal(t, "Hi there", loaded.Messages[1].TextContent())
	assert.Equal(t, "user-123", loaded.Metadata["user_id"])

	// Update the session
	state.Messages = append(state.Messages, llm.NewMessage(llm.RoleUser, llm.NewTextPart("How are you?")))
	err = store.Save(ctx, state)
	require.NoError(t, err)

	// Load updated session
	loaded, err = store.Load(ctx, "session-1")
	require.NoError(t, err)
	assert.Len(t, loaded.Messages, 3)
	assert.Equal(t, "How are you?", loaded.Messages[2].TextContent())

	// Delete the session
	err = store.Delete(ctx, "session-1")
	require.NoError(t, err)

	// Load should return ErrNotFound
	_, err = store.Load(ctx, "session-1")
	require.ErrorIs(t, err, session.ErrNotFound)

	// Delete non-existent session should be idempotent
	err = store.Delete(ctx, "session-1")
	require.NoError(t, err)
}

func TestKVStore_MultipleSessions(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container := requireRedpandaContainer(ctx, t)
	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	store, err := kvstore.NewKVStore(ctx, "test-multi-sessions",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Save multiple sessions
	for i := range 10 {
		state := &session.State{
			ID: "session-" + string(rune('a'+i)),
			Messages: []llm.Message{
				llm.NewMessage(llm.RoleUser, llm.NewTextPart("Message "+string(rune('A'+i)))),
			},
		}
		err = store.Save(ctx, state)
		require.NoError(t, err)
	}

	// Load all sessions
	for i := range 10 {
		loaded, err := store.Load(ctx, "session-"+string(rune('a'+i)))
		require.NoError(t, err)
		assert.Equal(t, "Message "+string(rune('A'+i)), loaded.Messages[0].TextContent())
	}
}

func TestKVStore_Bootstrap(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container := requireRedpandaContainer(ctx, t)
	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	const topic = "test-bootstrap-sessions"

	// First store: write some sessions
	store1, err := kvstore.NewKVStore(ctx, topic,
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	state := &session.State{
		ID: "bootstrap-session",
		Messages: []llm.Message{
			llm.NewMessage(llm.RoleUser, llm.NewTextPart("Persisted message")),
		},
		Metadata: map[string]any{"key": "value"},
	}
	err = store1.Save(ctx, state)
	require.NoError(t, err)
	store1.Close()

	// Second store: should bootstrap from Kafka
	store2, err := kvstore.NewKVStore(ctx, topic,
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store2.Close()

	// Data should be immediately available
	loaded, err := store2.Load(ctx, "bootstrap-session")
	require.NoError(t, err)
	assert.Equal(t, "Persisted message", loaded.Messages[0].TextContent())
	assert.Equal(t, "value", loaded.Metadata["key"])
}

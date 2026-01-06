package a2a_test

import (
	"testing"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/redpanda-data/common-go/kvstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"

	a2aadapter "github.com/redpanda-data/ai-sdk-go/adapter/a2a"
)

func TestKVTaskStore_SaveGet(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest",
		redpanda.WithAutoCreateTopics(),
	)
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	store, err := a2aadapter.NewKVTaskStore(ctx, "test-a2a-tasks",
		kvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Get non-existent task
	_, err = store.Get(ctx, "nonexistent")
	require.ErrorIs(t, err, a2a.ErrTaskNotFound)

	// Save a task
	task := &a2a.Task{
		ID:        "task-1",
		ContextID: "ctx-1",
		Status: a2a.TaskStatus{
			State: a2a.TaskStateWorking,
		},
		History: []*a2a.Message{
			a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Hello"}),
		},
		Metadata: map[string]any{
			"key": "value",
		},
	}
	err = store.Save(ctx, task)
	require.NoError(t, err)

	// Get the task back
	loaded, err := store.Get(ctx, "task-1")
	require.NoError(t, err)
	assert.Equal(t, task.ID, loaded.ID)
	assert.Equal(t, task.ContextID, loaded.ContextID)
	assert.Equal(t, a2a.TaskStateWorking, loaded.Status.State)
	assert.Len(t, loaded.History, 1)
	assert.Equal(t, "value", loaded.Metadata["key"])

	// Update the task
	task.Status.State = a2a.TaskStateCompleted
	task.History = append(task.History, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Done"}))
	err = store.Save(ctx, task)
	require.NoError(t, err)

	// Get updated task
	loaded, err = store.Get(ctx, "task-1")
	require.NoError(t, err)
	assert.Equal(t, a2a.TaskStateCompleted, loaded.Status.State)
	assert.Len(t, loaded.History, 2)
}

func TestKVTaskStore_MultipleTasks(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest",
		redpanda.WithAutoCreateTopics(),
	)
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	store, err := a2aadapter.NewKVTaskStore(ctx, "test-a2a-multi-tasks",
		kvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Save multiple tasks
	for i := range 10 {
		task := &a2a.Task{
			ID:        a2a.TaskID("task-" + string(rune('a'+i))),
			ContextID: "ctx-" + string(rune('a'+i)),
			Status: a2a.TaskStatus{
				State: a2a.TaskStateSubmitted,
			},
		}
		err = store.Save(ctx, task)
		require.NoError(t, err)
	}

	// Load all tasks
	for i := range 10 {
		loaded, err := store.Get(ctx, a2a.TaskID("task-"+string(rune('a'+i))))
		require.NoError(t, err)
		assert.Equal(t, "ctx-"+string(rune('a'+i)), loaded.ContextID)
	}
}

func TestKVTaskStore_Bootstrap(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest",
		redpanda.WithAutoCreateTopics(),
	)
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	const topic = "test-a2a-bootstrap-tasks"

	// First store: write some tasks
	store1, err := a2aadapter.NewKVTaskStore(ctx, topic,
		kvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	task := &a2a.Task{
		ID:        "bootstrap-task",
		ContextID: "bootstrap-ctx",
		Status: a2a.TaskStatus{
			State:   a2a.TaskStateCompleted,
			Message: a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Done"}),
		},
		Metadata: map[string]any{"key": "value"},
	}
	err = store1.Save(ctx, task)
	require.NoError(t, err)
	store1.Close()

	// Second store: should bootstrap from Kafka
	store2, err := a2aadapter.NewKVTaskStore(ctx, topic,
		kvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store2.Close()

	// Data should be immediately available
	loaded, err := store2.Get(ctx, "bootstrap-task")
	require.NoError(t, err)
	assert.Equal(t, "bootstrap-ctx", loaded.ContextID)
	assert.Equal(t, a2a.TaskStateCompleted, loaded.Status.State)
	assert.Equal(t, "value", loaded.Metadata["key"])
}

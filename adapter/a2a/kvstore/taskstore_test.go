// Copyright 2026 Redpanda Data, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvstore_test

import (
	"context"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"

	"github.com/redpanda-data/ai-sdk-go/adapter/a2a/kvstore"
)

func TestKVTaskStore_SaveGet(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-tasks",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Get non-existent task
	_, _, err = store.Get(ctx, "nonexistent")
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
	mustSave(ctx, t, store, task)

	// Get the task back
	loaded, _, err := store.Get(ctx, "task-1")
	require.NoError(t, err)
	assert.Equal(t, task.ID, loaded.ID)
	assert.Equal(t, task.ContextID, loaded.ContextID)
	assert.Equal(t, a2a.TaskStateWorking, loaded.Status.State)
	assert.Len(t, loaded.History, 1)
	assert.Equal(t, "value", loaded.Metadata["key"])

	// Update the task
	task.Status.State = a2a.TaskStateCompleted
	task.History = append(task.History, a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Done"}))
	mustSave(ctx, t, store, task)

	// Get updated task
	loaded, _, err = store.Get(ctx, "task-1")
	require.NoError(t, err)
	assert.Equal(t, a2a.TaskStateCompleted, loaded.Status.State)
	assert.Len(t, loaded.History, 2)
}

func TestKVTaskStore_MultipleTasks(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-multi-tasks",
		commonkvstore.WithBrokers(brokers),
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
		mustSave(ctx, t, store, task)
	}

	// Load all tasks
	for i := range 10 {
		loaded, _, err := store.Get(ctx, a2a.TaskID("task-"+string(rune('a'+i))))
		require.NoError(t, err)
		assert.Equal(t, "ctx-"+string(rune('a'+i)), loaded.ContextID)
	}
}

func TestKVTaskStore_Bootstrap(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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
	store1, err := kvstore.NewKVTaskStore(ctx, topic,
		commonkvstore.WithBrokers(brokers),
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
	mustSave(ctx, t, store1, task)
	store1.Close()

	// Second store: should bootstrap from Kafka
	store2, err := kvstore.NewKVTaskStore(ctx, topic,
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store2.Close()

	// Data should be immediately available
	loaded, _, err := store2.Get(ctx, "bootstrap-task")
	require.NoError(t, err)
	assert.Equal(t, "bootstrap-ctx", loaded.ContextID)
	assert.Equal(t, a2a.TaskStateCompleted, loaded.Status.State)
	assert.Equal(t, "value", loaded.Metadata["key"])
}

func TestKVTaskStore_ListSortedByTime(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-list-sorted",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Create tasks with different timestamps
	baseTime := time.Now()
	tasks := []*a2a.Task{
		{ID: "task-old", ContextID: "ctx", Status: a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime.Add(-2 * time.Hour))}},
		{ID: "task-mid", ContextID: "ctx", Status: a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime.Add(-1 * time.Hour))}},
		{ID: "task-new", ContextID: "ctx", Status: a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime)}},
	}

	// Save in random order
	mustSave(ctx, t, store, tasks[1]) // mid
	mustSave(ctx, t, store, tasks[0]) // old
	mustSave(ctx, t, store, tasks[2]) // new

	// List should return in descending time order (newest first)
	resp, err := store.List(ctx, &a2a.ListTasksRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 3)
	assert.Equal(t, "task-new", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-mid", string(resp.Tasks[1].ID))
	assert.Equal(t, "task-old", string(resp.Tasks[2].ID))
}

func TestKVTaskStore_ListPagination(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-list-pagination",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	// Create 5 tasks with distinct timestamps
	baseTime := time.Now()
	for i := range 5 {
		task := &a2a.Task{
			ID:        a2a.TaskID("task-" + string(rune('a'+i))),
			ContextID: "ctx",
			Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime.Add(time.Duration(i) * time.Minute))},
		}
		mustSave(ctx, t, store, task)
	}

	// First page: 2 items
	resp, err := store.List(ctx, &a2a.ListTasksRequest{PageSize: 2})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)
	assert.Equal(t, "task-e", string(resp.Tasks[0].ID)) // newest
	assert.Equal(t, "task-d", string(resp.Tasks[1].ID))
	assert.NotEmpty(t, resp.NextPageToken)

	// Second page: 2 items
	resp, err = store.List(ctx, &a2a.ListTasksRequest{PageSize: 2, PageToken: resp.NextPageToken})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)
	assert.Equal(t, "task-c", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-b", string(resp.Tasks[1].ID))
	assert.NotEmpty(t, resp.NextPageToken)

	// Third page: 1 item (last)
	resp, err = store.List(ctx, &a2a.ListTasksRequest{PageSize: 2, PageToken: resp.NextPageToken})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 1)
	assert.Equal(t, "task-a", string(resp.Tasks[0].ID)) // oldest
	assert.Empty(t, resp.NextPageToken)
}

func TestKVTaskStore_ListFilters(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-list-filters",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	baseTime := time.Now()

	// Create tasks with different contexts and states
	tasks := []*a2a.Task{
		{ID: "task-1", ContextID: "ctx-a", Status: a2a.TaskStatus{State: a2a.TaskStateWorking, Timestamp: ptr(baseTime)}},
		{ID: "task-2", ContextID: "ctx-a", Status: a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime.Add(time.Minute))}},
		{ID: "task-3", ContextID: "ctx-b", Status: a2a.TaskStatus{State: a2a.TaskStateWorking, Timestamp: ptr(baseTime.Add(2 * time.Minute))}},
		{ID: "task-4", ContextID: "ctx-b", Status: a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(baseTime.Add(3 * time.Minute))}},
	}
	for _, task := range tasks {
		mustSave(ctx, t, store, task)
	}

	// Filter by ContextID
	resp, err := store.List(ctx, &a2a.ListTasksRequest{ContextID: "ctx-a"})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)

	for _, task := range resp.Tasks {
		assert.Equal(t, "ctx-a", task.ContextID)
	}

	// Filter by Status
	resp, err = store.List(ctx, &a2a.ListTasksRequest{Status: a2a.TaskStateCompleted})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)

	for _, task := range resp.Tasks {
		assert.Equal(t, a2a.TaskStateCompleted, task.Status.State)
	}

	// Filter by LastUpdatedAfter
	cutoff := baseTime.Add(90 * time.Second)
	resp, err = store.List(ctx, &a2a.ListTasksRequest{LastUpdatedAfter: &cutoff})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2) // task-3 and task-4
	assert.Equal(t, "task-4", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-3", string(resp.Tasks[1].ID))

	// Combined filters
	resp, err = store.List(ctx, &a2a.ListTasksRequest{ContextID: "ctx-b", Status: a2a.TaskStateWorking})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 1)
	assert.Equal(t, "task-3", string(resp.Tasks[0].ID))
}

func TestKVTaskStore_ListHistoryAndArtifacts(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-list-history",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	task := &a2a.Task{
		ID:        "task-1",
		ContextID: "ctx",
		Status:    a2a.TaskStatus{State: a2a.TaskStateCompleted, Timestamp: ptr(time.Now())},
		History: []*a2a.Message{
			a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "msg1"}),
			a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "msg2"}),
			a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "msg3"}),
			a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "msg4"}),
		},
		Artifacts: []*a2a.Artifact{
			{Name: "artifact1"},
			{Name: "artifact2"},
		},
	}
	mustSave(ctx, t, store, task)

	// Default: no artifacts, full history
	resp, err := store.List(ctx, &a2a.ListTasksRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 1)
	assert.Len(t, resp.Tasks[0].History, 4)
	assert.Nil(t, resp.Tasks[0].Artifacts)

	// Trim history to last 2 messages
	resp, err = store.List(ctx, &a2a.ListTasksRequest{HistoryLength: 2})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 1)
	assert.Len(t, resp.Tasks[0].History, 2)

	text0, ok := resp.Tasks[0].History[0].Parts[0].(a2a.TextPart)
	require.True(t, ok)
	assert.Equal(t, "msg3", text0.Text)

	text1, ok := resp.Tasks[0].History[1].Parts[0].(a2a.TextPart)
	require.True(t, ok)
	assert.Equal(t, "msg4", text1.Text)

	// Include artifacts
	resp, err = store.List(ctx, &a2a.ListTasksRequest{IncludeArtifacts: true})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 1)
	assert.Len(t, resp.Tasks[0].Artifacts, 2)
}

func TestKVTaskStore_UpdateChangesSortOrder(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	store, err := kvstore.NewKVTaskStore(ctx, "test-a2a-update-sort",
		commonkvstore.WithBrokers(brokers),
	)
	require.NoError(t, err)

	defer store.Close()

	baseTime := time.Now()

	// Create tasks: task-a is oldest, task-b is newest
	taskA := &a2a.Task{ID: "task-a", ContextID: "ctx", Status: a2a.TaskStatus{State: a2a.TaskStateWorking, Timestamp: ptr(baseTime)}}
	taskB := &a2a.Task{ID: "task-b", ContextID: "ctx", Status: a2a.TaskStatus{State: a2a.TaskStateWorking, Timestamp: ptr(baseTime.Add(time.Hour))}}

	mustSave(ctx, t, store, taskA)
	mustSave(ctx, t, store, taskB)

	// Initial order: task-b, task-a
	resp, err := store.List(ctx, &a2a.ListTasksRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)
	assert.Equal(t, "task-b", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-a", string(resp.Tasks[1].ID))

	// Update task-a with newer timestamp - should move to front
	taskA.Status.Timestamp = ptr(baseTime.Add(2 * time.Hour))
	mustSave(ctx, t, store, taskA)

	// New order: task-a, task-b
	resp, err = store.List(ctx, &a2a.ListTasksRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)
	assert.Equal(t, "task-a", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-b", string(resp.Tasks[1].ID))

	// Verify Get still works
	loaded, _, err := store.Get(ctx, "task-a")
	require.NoError(t, err)
	assert.Equal(t, taskA.Status.Timestamp.Unix(), loaded.Status.Timestamp.Unix())
}

func TestKVTaskStore_BootstrapRestoresSortOrder(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
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

	const topic = "test-a2a-bootstrap-sort"

	// First store: create tasks
	store1, err := kvstore.NewKVTaskStore(ctx, topic, commonkvstore.WithBrokers(brokers))
	require.NoError(t, err)

	baseTime := time.Now()
	newTime := baseTime.Add(time.Hour)
	mustSave(ctx, t, store1, &a2a.Task{ID: "task-old", ContextID: "ctx", Status: a2a.TaskStatus{Timestamp: &baseTime}})
	mustSave(ctx, t, store1, &a2a.Task{ID: "task-new", ContextID: "ctx", Status: a2a.TaskStatus{Timestamp: &newTime}})

	_ = store1.Close()

	// Second store: bootstrap from Kafka
	store2, err := kvstore.NewKVTaskStore(ctx, topic, commonkvstore.WithBrokers(brokers))
	require.NoError(t, err)

	defer store2.Close()

	// Sort order should be restored
	resp, err := store2.List(ctx, &a2a.ListTasksRequest{})
	require.NoError(t, err)
	require.Len(t, resp.Tasks, 2)
	assert.Equal(t, "task-new", string(resp.Tasks[0].ID))
	assert.Equal(t, "task-old", string(resp.Tasks[1].ID))
}

func TestKVTaskStore_InvalidPageToken(t *testing.T) { //nolint:tparallel,paralleltest // Serial to reduce container memory pressure
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	ctx := t.Context()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest",
		redpanda.WithAutoCreateTopics(),
	)
	require.NoError(t, err)
	t.Cleanup(func() { _ = container.Terminate(ctx) })

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	const topic = "test-a2a-invalid-token"

	store, err := kvstore.NewKVTaskStore(ctx, topic, commonkvstore.WithBrokers(brokers))
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })

	// Create a task
	mustSave(ctx, t, store, &a2a.Task{
		ID:        "task-1",
		ContextID: "ctx",
		Status:    a2a.TaskStatus{Timestamp: ptr(time.Now())},
	})

	testCases := []struct {
		name      string
		pageToken string
		expectErr string
	}{
		{
			name:      "garbage base64",
			pageToken: "not-valid-base64!@#$",
			expectErr: "invalid page token",
		},
		{
			name:      "valid base64 but not JSON",
			pageToken: "bm90IGpzb24=", // "not json" in base64
			expectErr: "invalid page token",
		},
		{
			name:      "valid JSON but missing sortKey",
			pageToken: "e30=", // "{}" in base64
			expectErr: "invalid page token",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			_, err := store.List(ctx, &a2a.ListTasksRequest{
				PageToken: tc.pageToken,
			})
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.expectErr)
		})
	}
}

func mustSave(ctx context.Context, t *testing.T, store *kvstore.KVTaskStore, task *a2a.Task) {
	t.Helper()

	_, err := store.Save(ctx, task, nil, nil, 0)
	require.NoError(t, err)
}

func ptr[T any](v T) *T {
	return &v
}

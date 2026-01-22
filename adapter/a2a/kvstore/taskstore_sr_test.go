package kvstore_test

import (
	"context"
	"testing"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/bufbuild/protocompile"
	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sr"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/dynamicpb"

	"github.com/redpanda-data/ai-sdk-go/adapter/a2a/kvstore"
)

// TestKVTaskStoreWithSchemaRegistry_SchemaRegistration verifies schema registration.
func TestKVTaskStoreWithSchemaRegistry_SchemaRegistration(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest")
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-a2a-sr-registration"
	expectedSubject := "redpanda-a2a-task-value"

	store, err := kvstore.NewKVTaskStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	// Save a task to trigger schema registration
	now := time.Now()
	testTask := &a2a.Task{
		ID:        "task-registration-test",
		ContextID: "ctx-1",
		Status: a2a.TaskStatus{
			State:     a2a.TaskStateWorking,
			Timestamp: &now,
		},
		History: []*a2a.Message{
			a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Schema registration test"}),
		},
	}

	err = store.Save(ctx, testTask)
	require.NoError(t, err)

	// Verify schema is registered
	subjects, err := srClient.Subjects(ctx)
	require.NoError(t, err)
	require.Contains(t, subjects, expectedSubject)

	subjectSchema, err := srClient.SchemaByVersion(ctx, expectedSubject, -1)
	require.NoError(t, err)
	assert.Equal(t, sr.TypeProtobuf, subjectSchema.Type)
	assert.Positive(t, subjectSchema.ID)
	assert.NotEmpty(t, subjectSchema.Schema.Schema)
	assert.Empty(t, subjectSchema.References, "a2a_task.proto should be self-contained")
}

// TestKVTaskStoreWithSchemaRegistry_RoundTrip tests full serialization round-trip.
func TestKVTaskStoreWithSchemaRegistry_RoundTrip(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest")
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-a2a-sr-roundtrip"

	store, err := kvstore.NewKVTaskStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	now := time.Now()

	testTasks := []*a2a.Task{
		{
			ID:        "task-1",
			ContextID: "ctx-1",
			Status: a2a.TaskStatus{
				State:     a2a.TaskStateSubmitted,
				Timestamp: &now,
			},
		},
		{
			ID:        "task-2",
			ContextID: "ctx-2",
			Status: a2a.TaskStatus{
				State:     a2a.TaskStateWorking,
				Timestamp: &now,
				Message:   a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Working on it"}),
			},
			History: []*a2a.Message{
				a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Hello"}),
				a2a.NewMessage(a2a.MessageRoleAgent, a2a.TextPart{Text: "Hi there"}),
			},
			Metadata: map[string]any{
				"user_id": "user-123",
				"tags":    []any{"test", "roundtrip"},
			},
		},
		{
			ID:        "task-3",
			ContextID: "ctx-3",
			Status: a2a.TaskStatus{
				State:     a2a.TaskStateCompleted,
				Timestamp: &now,
			},
			Artifacts: []*a2a.Artifact{
				{
					Name:        "result.txt",
					Description: "Test artifact",
					Parts:       []a2a.Part{a2a.TextPart{Text: "artifact content"}},
				},
			},
		},
	}

	for _, task := range testTasks {
		err := store.Save(ctx, task)
		require.NoError(t, err)
	}

	for _, original := range testTasks {
		loaded, err := store.Get(ctx, original.ID)
		require.NoError(t, err)

		assert.Equal(t, original.ID, loaded.ID)
		assert.Equal(t, original.ContextID, loaded.ContextID)
		assert.Equal(t, original.Status.State, loaded.Status.State)
		assert.Len(t, loaded.History, len(original.History))
		assert.Len(t, loaded.Artifacts, len(original.Artifacts))

		if original.Metadata != nil {
			assert.Equal(t, original.Metadata, loaded.Metadata)
		}
	}
}

// TestKVTaskStoreWithSchemaRegistry_DynamicDeserialization validates that tasks
// can be deserialized using dynamic proto from Schema Registry, like Console does.
func TestKVTaskStoreWithSchemaRegistry_DynamicDeserialization(t *testing.T) { //nolint:paralleltest // Serial to reduce container memory pressure
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest")
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-a2a-sr-dynamic"
	subject := "redpanda-a2a-task-value"

	store, err := kvstore.NewKVTaskStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	now := time.Now()
	testTask := &a2a.Task{
		ID:        "task-dynamic-test",
		ContextID: "ctx-dynamic",
		Status: a2a.TaskStatus{
			State:     a2a.TaskStateWorking,
			Timestamp: &now,
			Message: &a2a.Message{
				Role:  a2a.MessageRoleAgent,
				Parts: []a2a.Part{a2a.TextPart{Text: "Processing..."}},
			},
		},
		History: []*a2a.Message{
			a2a.NewMessage(a2a.MessageRoleUser, a2a.TextPart{Text: "Hello there"}),
		},
		Metadata: map[string]any{
			"test_key": "test_value",
		},
	}

	err = store.Save(ctx, testTask)
	require.NoError(t, err)

	// Read raw message from Kafka
	kafkaClient, err := kgo.NewClient(
		kgo.SeedBrokers(brokers),
		kgo.ConsumeTopics(topic),
	)
	require.NoError(t, err)

	defer kafkaClient.Close()

	var record *kgo.Record

	for range 10 {
		pollCtx, pollCancel := context.WithTimeout(ctx, time.Second)
		fetches := kafkaClient.PollFetches(pollCtx)

		pollCancel()

		fetches.EachRecord(func(r *kgo.Record) {
			if r.Topic == topic {
				record = r
			}
		})

		if record != nil {
			break
		}

		time.Sleep(100 * time.Millisecond)
	}

	require.NotNil(t, record)

	// Decode Schema Registry wire format
	var header sr.ConfluentHeader
	_, remaining, err := header.DecodeID(record.Value)
	require.NoError(t, err)

	messageIndexes, payload, err := header.DecodeIndex(remaining, 10)
	require.NoError(t, err)

	// Fetch schema from Schema Registry
	schemaMap, err := fetchSchemaWithReferences(ctx, t, srClient, subject)
	require.NoError(t, err)

	// Compile using protocompile like Console does
	compiler := protocompile.Compiler{
		Resolver: protocompile.WithStandardImports(&protocompile.SourceResolver{
			Accessor: protocompile.SourceAccessorFromMap(schemaMap),
		}),
	}

	fds, err := compiler.Compile(ctx, "a2a_task.proto")
	require.NoError(t, err)
	require.NotEmpty(t, fds)

	// Use message index from wire format to select message descriptor
	require.Len(t, messageIndexes, 1)
	messageIndex := messageIndexes[0]

	rootDescriptors := fds[0].Messages()
	require.Greater(t, rootDescriptors.Len(), messageIndex)

	msgDesc := rootDescriptors.Get(messageIndex)
	require.Equal(t, "Task", string(msgDesc.Name()),
		"message at index %d should be Task", messageIndex)

	// Unmarshal using dynamic proto
	taskDynamic := dynamicpb.NewMessage(msgDesc)
	err = proto.Unmarshal(payload, taskDynamic)
	require.NoError(t, err)

	// Verify fields
	assert.Equal(t, string(testTask.ID), taskDynamic.Get(msgDesc.Fields().ByName("id")).String())
	assert.Equal(t, testTask.ContextID, taskDynamic.Get(msgDesc.Fields().ByName("context_id")).String())
}

// fetchSchemaWithReferences fetches a schema and all references from Schema Registry.
func fetchSchemaWithReferences(ctx context.Context, t *testing.T, client *sr.Client, subject string) (map[string]string, error) {
	t.Helper()

	schemaMap := make(map[string]string)

	subjectSchema, err := client.SchemaByVersion(ctx, subject, -1)
	if err != nil {
		return nil, err
	}

	schemaMap["a2a_task.proto"] = subjectSchema.Schema.Schema

	if err := fetchReferences(ctx, t, client, subjectSchema.References, schemaMap); err != nil {
		return nil, err
	}

	return schemaMap, nil
}

// fetchReferences recursively fetches schema references.
func fetchReferences(ctx context.Context, t *testing.T, client *sr.Client, refs []sr.SchemaReference, schemaMap map[string]string) error {
	t.Helper()

	for _, ref := range refs {
		if _, exists := schemaMap[ref.Name]; exists {
			continue
		}

		refSchema, err := client.SchemaByVersion(ctx, ref.Subject, ref.Version)
		if err != nil {
			return err
		}

		schemaMap[ref.Name] = refSchema.Schema.Schema

		if err := fetchReferences(ctx, t, client, refSchema.References, schemaMap); err != nil {
			return err
		}
	}

	return nil
}

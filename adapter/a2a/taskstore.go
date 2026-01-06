package a2a

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/redpanda-data/common-go/kvstore"
	"github.com/redpanda-data/common-go/kvstore/memdb"
)

// Compile-time interface check.
var _ a2asrv.TaskStore = (*KVTaskStore)(nil)

// KVTaskStore provides A2A task storage backed by Kafka via kvstore.
// Writes are persisted to Kafka and block until visible in reads.
// Multiple instances share state through Kafka topic consumption.
type KVTaskStore struct {
	client *kvstore.ResourceClient[*a2a.Task]
	raw    *kvstore.Client
}

// NewKVTaskStore creates a new A2A task store backed by Kafka.
//
// The topic parameter specifies the Kafka topic for task storage.
// Use kvstore.WithBrokers, kvstore.WithReplicationFactor, kvstore.WithKafkaOptions,
// and kvstore.WithLogger to configure the underlying client.
//
// The context is used only for initialization (topic creation, metadata fetches,
// and bootstrap sync). It is not retained after NewKVTaskStore returns.
//
// NewKVTaskStore blocks until the consumer has caught up to the current high watermark,
// ensuring reads are consistent immediately after creation.
//
// The caller MUST call Close() to release resources when done.
func NewKVTaskStore(ctx context.Context, topic string, opts ...kvstore.ClientOption) (*KVTaskStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	// Prepend storage option so user options can override if needed
	allOpts := append([]kvstore.ClientOption{kvstore.WithStorage(storage)}, opts...)

	client, err := kvstore.NewClient(ctx, topic, allOpts...)
	if err != nil {
		return nil, err
	}

	return &KVTaskStore{
		client: kvstore.NewResourceClient(client, &taskSerde{}),
		raw:    client,
	}, nil
}

// Save stores a task.
// Blocks until the write is visible in this client's reads.
func (s *KVTaskStore) Save(ctx context.Context, task *a2a.Task) error {
	return s.client.Put(ctx, []byte(task.ID), task)
}

// Get retrieves a task by ID.
// Returns a2a.ErrTaskNotFound if the task does not exist.
func (s *KVTaskStore) Get(ctx context.Context, taskID a2a.TaskID) (*a2a.Task, error) {
	task, err := s.client.Get(ctx, []byte(taskID))
	if errors.Is(err, kvstore.ErrNotFound) {
		return nil, a2a.ErrTaskNotFound
	}

	if err != nil {
		return nil, err
	}

	return task, nil
}

// Close shuts down the kvstore client and releases resources.
func (s *KVTaskStore) Close() error {
	return s.raw.Close()
}

// taskSerde handles JSON serialization of a2a.Task.
type taskSerde struct{}

func (*taskSerde) Serialize(t *a2a.Task) ([]byte, error) {
	return json.Marshal(t)
}

func (*taskSerde) Deserialize(b []byte) (*a2a.Task, error) {
	var t a2a.Task
	if err := json.Unmarshal(b, &t); err != nil {
		return nil, err
	}

	return &t, nil
}

package session

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/redpanda-data/common-go/kvstore"
	"github.com/redpanda-data/common-go/kvstore/memdb"
)

// Compile-time interface check.
var _ Store = (*KVStore)(nil)

// KVStore provides session storage backed by Kafka via kvstore.
// Writes are persisted to Kafka and block until visible in reads.
// Multiple instances share state through Kafka topic consumption.
type KVStore struct {
	client *kvstore.ResourceClient[*State]
	raw    *kvstore.Client
}

// NewKVStore creates a new session store backed by Kafka.
//
// The topic parameter specifies the Kafka topic for session storage.
// Use kvstore.WithBrokers, kvstore.WithReplicationFactor, kvstore.WithKafkaOptions,
// and kvstore.WithLogger to configure the underlying client.
//
// The context is used only for initialization (topic creation, metadata fetches,
// and bootstrap sync). It is not retained after NewKVStore returns.
//
// NewKVStore blocks until the consumer has caught up to the current high watermark,
// ensuring reads are consistent immediately after creation.
//
// The caller MUST call Close() to release resources when done.
func NewKVStore(ctx context.Context, topic string, opts ...kvstore.ClientOption) (*KVStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	client, err := kvstore.NewClient(ctx, topic, storage, opts...)
	if err != nil {
		return nil, err
	}

	return &KVStore{
		client: kvstore.NewResourceClient(client, &stateSerde{}),
		raw:    client,
	}, nil
}

// Load retrieves a session by ID.
// Returns ErrNotFound if the session does not exist.
func (s *KVStore) Load(ctx context.Context, sessionID string) (*State, error) {
	state, err := s.client.Get(ctx, []byte(sessionID))
	if errors.Is(err, kvstore.ErrNotFound) {
		return nil, ErrNotFound
	}

	if err != nil {
		return nil, err
	}

	return state, nil
}

// Save persists a session.
// Blocks until the write is visible in this client's reads.
func (s *KVStore) Save(ctx context.Context, state *State) error {
	return s.client.Put(ctx, []byte(state.ID), state)
}

// Delete removes a session by ID.
// Blocks until the delete is visible in this client's reads.
// Returns nil if the session doesn't exist (idempotent).
func (s *KVStore) Delete(ctx context.Context, sessionID string) error {
	return s.client.Delete(ctx, []byte(sessionID))
}

// Close shuts down the kvstore client and releases resources.
func (s *KVStore) Close() error {
	return s.raw.Close()
}

// stateSerde handles JSON serialization of session State.
type stateSerde struct{}

func (*stateSerde) Serialize(s *State) ([]byte, error) {
	return json.Marshal(s)
}

func (*stateSerde) Deserialize(b []byte) (*State, error) {
	var s State
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}

	return &s, nil
}

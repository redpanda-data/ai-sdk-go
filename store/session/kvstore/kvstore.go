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

package kvstore

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"

	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/redpanda-data/common-go/kvstore/memdb"
	"github.com/twmb/franz-go/pkg/sr"

	"github.com/redpanda-data/ai-sdk-go/store/session"
	llmpb "github.com/redpanda-data/ai-sdk-go/store/session/kvstore/proto/gen/go/redpanda/llm/v1"
)

// Compile-time interface check.
var _ session.Store = (*KVStore)(nil)

//go:embed proto/session_state.proto
var sessionStateProtoSchema string

// KVStore provides session storage backed by Kafka via kvstore.
// Writes are persisted to Kafka and block until visible in reads.
// Multiple instances share state through Kafka topic consumption.
type KVStore struct {
	client *commonkvstore.ResourceClient[*session.State]
	raw    *commonkvstore.Client
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
func NewKVStore(ctx context.Context, topic string, opts ...commonkvstore.ClientOption) (*KVStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	client, err := commonkvstore.NewClient(ctx, topic, storage, opts...)
	if err != nil {
		return nil, err
	}

	return &KVStore{
		client: commonkvstore.NewResourceClient(client, &stateSerde{}),
		raw:    client,
	}, nil
}

// NewKVStoreWithSchemaRegistry creates a new session store backed by Kafka with Schema Registry integration.
//
// This constructor uses protobuf serialization with Confluent Schema Registry wire format.
// The schema is automatically registered with the Schema Registry on first write.
//
// Parameters:
//   - ctx: Context for initialization only (not retained)
//   - topic: Kafka topic for session storage
//   - srClient: Schema Registry client for schema registration
//   - opts: Additional kvstore options (brokers, replication factor, etc.)
//
// All sessions share the schema subject "redpanda-session-value" to avoid Schema Registry
// accumulation of identical per-session schemas. Each session still uses its own topic for data isolation.
//
// Example:
//
//	srClient, _ := sr.NewClient(sr.URLs("http://localhost:8081"))
//	store, err := session.NewKVStoreWithSchemaRegistry(
//	    ctx,
//	    "agent-sessions",
//	    srClient,
//	    kvstore.WithBrokers("localhost:9092"),
//	)
func NewKVStoreWithSchemaRegistry(
	ctx context.Context,
	topic string,
	srClient *sr.Client,
	opts ...commonkvstore.ClientOption,
) (*KVStore, error) {
	storage, err := memdb.New()
	if err != nil {
		return nil, err
	}

	client, err := commonkvstore.NewClient(ctx, topic, storage, opts...)
	if err != nil {
		return nil, err
	}

	// Create protobuf serde with Schema Registry support
	// Uses shared subject to avoid per-session schema accumulation in Schema Registry.
	// All sessions use identical llmpb.SessionState schema - no need for per-topic subjects.
	// Uses session_state.proto which contains SessionState and all dependencies,
	// ensuring SessionState is at message index 0.
	subject := "redpanda-session-value"

	srSerde, err := commonkvstore.Proto(
		func() *llmpb.SessionState { return &llmpb.SessionState{} },
		commonkvstore.WithSchemaRegistry(
			srClient,
			subject,
			sessionStateProtoSchema,
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create proto serde: %w", err)
	}

	// Wrap with conversion layer (Go types ↔ proto types)
	// Note: We use external conversion functions to avoid import cycles
	serde := newSessionSRSerde(srSerde)

	return &KVStore{
		client: commonkvstore.NewResourceClient(client, serde),
		raw:    client,
	}, nil
}

// Load retrieves a session by ID.
// Returns session.ErrNotFound if the session does not exist.
func (s *KVStore) Load(ctx context.Context, sessionID string) (*session.State, error) {
	state, err := s.client.Get(ctx, []byte(sessionID))
	if errors.Is(err, commonkvstore.ErrNotFound) {
		return nil, session.ErrNotFound
	}

	if err != nil {
		return nil, err
	}

	return state, nil
}

// Save persists a session.
// Blocks until the write is visible in this client's reads.
func (s *KVStore) Save(ctx context.Context, state *session.State) error {
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

func (*stateSerde) Serialize(s *session.State) ([]byte, error) {
	return json.Marshal(s)
}

func (*stateSerde) Deserialize(b []byte) (*session.State, error) {
	var s session.State
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}

	return &s, nil
}

// sessionSRSerde combines conversion and protobuf+SR serialization.
// It converts Go session.State ↔ protobuf llmpb.SessionState, then delegates to proto serde.
//
// To avoid import cycles, conversion functions are provided externally.
// Use newSessionSRSerde() to create with default converters from store/pbconv.
type sessionSRSerde struct {
	inner     commonkvstore.Serde[*llmpb.SessionState]
	toProto   func(*session.State) (*llmpb.SessionState, error)
	fromProto func(*llmpb.SessionState) (*session.State, error)
}

// newSessionSRSerde creates a sessionSRSerde with default converters.
// This function uses import magic to reference store/pbconv without creating a cycle.
func newSessionSRSerde(inner commonkvstore.Serde[*llmpb.SessionState]) *sessionSRSerde {
	// We use a functional approach here to reference the conversion functions
	// from store/pbconv without importing that package (avoiding cycle).
	//
	// The conversion logic is implemented in store/pbconv/converter.go
	// and we reference it through this registration mechanism.
	return &sessionSRSerde{
		inner:     inner,
		toProto:   getToProtoConverter(),
		fromProto: getFromProtoConverter(),
	}
}

// Serialize converts session.State to proto, then serializes with SR wire format.
func (s *sessionSRSerde) Serialize(state *session.State) ([]byte, error) {
	pb, err := s.toProto(state)
	if err != nil {
		return nil, fmt.Errorf("convert to proto: %w", err)
	}

	return s.inner.Serialize(pb)
}

// Deserialize decodes SR wire format, unmarshals proto, then converts to session.State.
func (s *sessionSRSerde) Deserialize(b []byte) (*session.State, error) {
	pb, err := s.inner.Deserialize(b)
	if err != nil {
		return nil, fmt.Errorf("deserialize proto: %w", err)
	}

	return s.fromProto(pb)
}

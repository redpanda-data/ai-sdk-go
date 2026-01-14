package kvstore_test

import (
	"context"
	"encoding/hex"
	"testing"
	"time"

	commonkvstore "github.com/redpanda-data/common-go/kvstore"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/testcontainers/testcontainers-go/modules/redpanda"
	"github.com/twmb/franz-go/pkg/kgo"
	"github.com/twmb/franz-go/pkg/sr"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
)

// TestKVStoreWithSchemaRegistry_WireFormat verifies the Confluent Schema Registry wire format.
// This test ensures that data is stored with the correct wire format encoding:
// [magic_byte=0x00][schema_id:4bytes][message_indexes][protobuf_data].
func TestKVStoreWithSchemaRegistry_WireFormat(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)

	defer cancel()

	// Start Redpanda with Schema Registry
	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest", redpandaLowMemory())
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	// Create Schema Registry client
	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-session-sr-wire-format"

	// Create session store with Schema Registry
	store, err := kvstore.NewKVStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	// Create a test session
	testSession := &session.State{
		ID: "session-wire-format-test",
		Messages: []llm.Message{
			{
				Role: llm.RoleUser,
				Content: []*llm.Part{
					llm.NewTextPart("Test message for wire format verification"),
				},
			},
		},
		Metadata: map[string]any{
			"test": "wire_format",
		},
	}

	// Save the session
	err = store.Save(ctx, testSession)
	require.NoError(t, err)

	// Create Kafka consumer to read raw bytes
	kafkaClient, err := kgo.NewClient(
		kgo.SeedBrokers(brokers),
		kgo.ConsumeTopics(topic),
	)
	require.NoError(t, err)

	defer kafkaClient.Close()

	// Poll for the record
	var record *kgo.Record

	for range 10 {
		fetches := kafkaClient.PollFetches(ctx)
		require.Empty(t, fetches.Errors())

		fetches.EachRecord(func(r *kgo.Record) {
			if string(r.Key) == testSession.ID {
				record = r
			}
		})

		if record != nil {
			break
		}

		time.Sleep(100 * time.Millisecond)
	}

	require.NotNil(t, record, "failed to find session record in Kafka")

	// Verify wire format
	value := record.Value
	t.Logf("Raw value (hex): %s", hex.EncodeToString(value))
	t.Logf("Raw value length: %d bytes", len(value))

	// Check minimum length: magic(1) + schemaID(4) + indexes(at least 1) + data(at least 1)
	require.GreaterOrEqual(t, len(value), 7, "value too short for Schema Registry protobuf format")

	// Verify magic byte (0x00)
	assert.Equal(t, byte(0x00), value[0], "magic byte should be 0x00")
	t.Logf("✓ Magic byte verified: 0x%02x", value[0])

	// Decode using ConfluentHeader
	var header sr.ConfluentHeader

	// Decode schema ID
	schemaID, remaining, err := header.DecodeID(value)
	require.NoError(t, err, "failed to decode schema ID")
	assert.Positive(t, schemaID, "schema ID should be positive")
	t.Logf("✓ Schema ID decoded: %d", schemaID)

	// Decode protobuf message indexes
	messageIndexes, payload, err := header.DecodeIndex(remaining, 10)
	require.NoError(t, err, "failed to decode message indexes")
	require.Len(t, messageIndexes, 1, "expected single message index for top-level message")
	assert.Equal(t, 0, messageIndexes[0], "message index should be 0 for top-level message")
	t.Logf("✓ Message indexes decoded: %v", messageIndexes)
	t.Logf("✓ Protobuf payload length: %d bytes", len(payload))

	// Verify we can read the session back successfully
	loadedSession, err := store.Load(ctx, testSession.ID)
	require.NoError(t, err)
	assert.Equal(t, testSession.ID, loadedSession.ID)
	assert.Len(t, loadedSession.Messages, len(testSession.Messages))
	assert.Equal(t, testSession.Metadata["test"], loadedSession.Metadata["test"])

	t.Log("✓ Wire format verification complete")
}

// TestKVStoreWithSchemaRegistry_SchemaRegistration verifies schema registration.
func TestKVStoreWithSchemaRegistry_SchemaRegistration(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)

	defer cancel()

	// Start Redpanda with Schema Registry
	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest", redpandaLowMemory())
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	// Create Schema Registry client
	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-session-sr-registration"
	expectedSubject := "redpanda-session-value"

	// Create session store with Schema Registry
	store, err := kvstore.NewKVStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	// Save a session to trigger schema registration
	testSession := &session.State{
		ID: "session-registration-test",
		Messages: []llm.Message{
			{
				Role: llm.RoleAssistant,
				Content: []*llm.Part{
					llm.NewTextPart("Schema registration test"),
				},
			},
		},
	}

	err = store.Save(ctx, testSession)
	require.NoError(t, err)

	// Wait a bit for schema registration to complete
	time.Sleep(500 * time.Millisecond)

	// Verify schema is registered in Schema Registry
	subjects, err := srClient.Subjects(ctx)
	require.NoError(t, err)
	require.Contains(t, subjects, expectedSubject, "schema subject should be registered")

	// Get the schema
	subjectSchema, err := srClient.SchemaByVersion(ctx, expectedSubject, -1) // -1 = latest
	require.NoError(t, err)
	assert.Equal(t, sr.TypeProtobuf, subjectSchema.Type, "schema type should be Protobuf")
	assert.Positive(t, subjectSchema.ID, "schema ID should be positive")
	assert.NotEmpty(t, subjectSchema.Schema.Schema, "schema content should not be empty")

	t.Logf("✓ Schema registered: subject=%s, id=%d, type=%s", expectedSubject, subjectSchema.ID, subjectSchema.Type)
	t.Logf("✓ Schema content length: %d bytes", len(subjectSchema.Schema.Schema))
}

// TestKVStoreWithSchemaRegistry_RoundTrip tests full serialization round-trip.
func TestKVStoreWithSchemaRegistry_RoundTrip(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)

	defer cancel()

	// Start Redpanda with Schema Registry
	container, err := redpanda.Run(ctx, "redpandadata/redpanda:latest", redpandaLowMemory())
	require.NoError(t, err)

	defer func() { _ = container.Terminate(ctx) }()

	brokers, err := container.KafkaSeedBroker(ctx)
	require.NoError(t, err)

	schemaRegistryURL, err := container.SchemaRegistryAddress(ctx)
	require.NoError(t, err)

	// Create Schema Registry client
	srClient, err := sr.NewClient(sr.URLs(schemaRegistryURL))
	require.NoError(t, err)

	topic := "test-session-sr-roundtrip"

	// Create session store with Schema Registry
	store, err := kvstore.NewKVStoreWithSchemaRegistry(
		ctx,
		topic,
		srClient,
		commonkvstore.WithBrokers(brokers),
		commonkvstore.WithReplicationFactor(1),
	)
	require.NoError(t, err)

	defer store.Close()

	// Create test sessions with various message types
	testSessions := []*session.State{
		{
			ID: "session-1",
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewTextPart("Simple text message"),
					},
				},
			},
		},
		{
			ID: "session-2",
			Messages: []llm.Message{
				{
					Role: llm.RoleAssistant,
					Content: []*llm.Part{
						llm.NewTextPart("Response with tool call"),
						llm.NewToolRequestPart(&llm.ToolRequest{
							ID:        "req-1",
							Name:      "search",
							Arguments: []byte(`{"query": "test"}`),
						}),
					},
				},
			},
			Metadata: map[string]any{
				"user_id": "user-123",
				"tags":    []any{"test", "roundtrip"},
			},
		},
		{
			ID: "session-3",
			Messages: []llm.Message{
				{
					Role: llm.RoleUser,
					Content: []*llm.Part{
						llm.NewToolResponsePart(&llm.ToolResponse{
							ID:     "req-1",
							Name:   "search",
							Result: []byte(`{"results": ["item1", "item2"]}`),
						}),
					},
				},
			},
		},
	}

	// Save all sessions
	for _, s := range testSessions {
		err := store.Save(ctx, s)
		require.NoError(t, err, "failed to save session %s", s.ID)
	}

	// Load and verify each session
	for _, original := range testSessions {
		loaded, err := store.Load(ctx, original.ID)
		require.NoError(t, err, "failed to load session %s", original.ID)

		assert.Equal(t, original.ID, loaded.ID)
		assert.Len(t, loaded.Messages, len(original.Messages))

		for msgIdx, origMsg := range original.Messages {
			loadedMsg := loaded.Messages[msgIdx]
			assert.Equal(t, origMsg.Role, loadedMsg.Role, "message %d role mismatch", msgIdx)
			assert.Len(t, loadedMsg.Content, len(origMsg.Content), "message %d content length mismatch", msgIdx)

			for partIdx, origPart := range origMsg.Content {
				loadedPart := loadedMsg.Content[partIdx]
				assert.Equal(t, origPart.Kind, loadedPart.Kind, "message %d part %d kind mismatch", msgIdx, partIdx)
			}
		}

		if original.Metadata != nil {
			assert.Equal(t, original.Metadata, loaded.Metadata)
		}

		t.Logf("✓ Session %s round-trip verified", original.ID)
	}
}

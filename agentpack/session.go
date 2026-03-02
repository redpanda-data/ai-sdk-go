package agentpack

import (
	"context"
	"fmt"
	"os"
	"strings"

	commonkvstore "github.com/redpanda-data/common-go/kvstore"

	"github.com/redpanda-data/ai-sdk-go/store/session"
	"github.com/redpanda-data/ai-sdk-go/store/session/kvstore"
)

// newSessionStore creates a session store from environment variables.
//
// Env vars:
//   - SESSION_STORE: memory | redpanda (default: memory)
//   - KAFKA_BROKERS: comma-separated broker list (for redpanda)
//   - KAFKA_TOPIC: topic name (default: agent-sessions)
func newSessionStore(ctx context.Context) (session.Store, func(), error) {
	storeType := os.Getenv("SESSION_STORE")
	if storeType == "" {
		storeType = "memory"
	}

	switch storeType {
	case "memory":
		store := session.NewInMemoryStore()
		return store, func() {}, nil

	case "redpanda":
		brokers := os.Getenv("KAFKA_BROKERS")
		if brokers == "" {
			return nil, nil, fmt.Errorf("KAFKA_BROKERS env var is required when SESSION_STORE=redpanda")
		}

		topic := os.Getenv("KAFKA_TOPIC")
		if topic == "" {
			topic = "agent-sessions"
		}

		brokerList := strings.Split(brokers, ",")
		store, err := kvstore.NewKVStore(ctx, topic, commonkvstore.WithBrokers(brokerList...))
		if err != nil {
			return nil, nil, fmt.Errorf("create redpanda session store: %w", err)
		}
		cleanup := func() { store.Close() }
		return store, cleanup, nil

	default:
		return nil, nil, fmt.Errorf("unknown SESSION_STORE: %s (supported: memory, redpanda)", storeType)
	}
}

// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/twmb/franz-go/pkg/kgo"
)

// Config is shared by Client and Worker.
type Config struct {
	// Brokers are the seed brokers.
	Brokers []string
	// TopicPrefix defaults to DefaultTopicPrefix.
	TopicPrefix string
	// EngineURL is the engine's read-only HTTP base URL, used by Client.Describe
	// and Client.WaitResult. Optional.
	EngineURL string
	// KafkaOpts are appended to every franz-go client built by this package
	// (TLS, SASL, client id, ...).
	KafkaOpts []kgo.Opt
}

func (c Config) topics() Topics {
	return Topics{Prefix: c.TopicPrefix}
}

func (c Config) kafkaOpts(extra ...kgo.Opt) []kgo.Opt {
	opts := []kgo.Opt{
		kgo.SeedBrokers(c.Brokers...),
		kgo.RequiredAcks(kgo.AllISRAcks()),
		kgo.ProducerBatchCompression(kgo.SnappyCompression()),
	}
	opts = append(opts, c.KafkaOpts...)

	return append(opts, extra...)
}

// producer is the minimal synchronous keyed-JSON producer both Client and
// Worker use. Every produce waits for acks=all so a returned nil error means
// the record is durable.
type producer interface {
	Produce(ctx context.Context, topic, key string, value any) error
}

type kgoProducer struct {
	cl *kgo.Client
}

func (p *kgoProducer) Produce(ctx context.Context, topic, key string, value any) error {
	b, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("durable: marshal %s record: %w", topic, err)
	}

	rec := &kgo.Record{Topic: topic, Key: []byte(key), Value: b}
	if err := p.cl.ProduceSync(ctx, rec).FirstErr(); err != nil {
		return fmt.Errorf("durable: produce to %s: %w", topic, err)
	}

	return nil
}

// NewCommand builds a command with a fresh id and timestamp.
func NewCommand(typ, runID string) Command {
	return Command{
		Type:      typ,
		CommandID: uuid.NewString(),
		RunID:     runID,
		SentAt:    time.Now().UTC(),
	}
}

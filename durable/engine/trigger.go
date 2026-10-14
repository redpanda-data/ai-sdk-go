// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/durable"
	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Trigger is the handoff from event streaming to durable execution: it starts
// one run per record on a source topic. The run id is derived from the record
// key (or partition/offset when there is none), so replaying or redelivering
// the source topic never starts duplicate runs.
type Trigger struct {
	// Client config (brokers, prefix, kafka opts).
	Config durable.Config
	// Source is the business topic to consume.
	Source string
	// ConsumerGroup defaults to "durable-trigger-<source>".
	ConsumerGroup string
	// Agent, Version and TaskQueue are passed to StartOptions.
	Agent     string
	Version   string
	TaskQueue string
	// RunIDPrefix is prepended to the derived id. Default "<agent>:".
	RunIDPrefix string
	// Message builds the first user message from a record. Default: the record
	// value as text.
	Message func(rec *kgo.Record) llm.Message
	Logger  *slog.Logger
}

// Run consumes the source topic until ctx ends.
func (t *Trigger) Run(ctx context.Context) error {
	if t.Logger == nil {
		t.Logger = slog.Default()
	}

	if t.ConsumerGroup == "" {
		t.ConsumerGroup = "durable-trigger-" + t.Source
	}

	if t.RunIDPrefix == "" {
		t.RunIDPrefix = t.Agent + ":"
	}

	if t.Message == nil {
		t.Message = func(rec *kgo.Record) llm.Message {
			return llm.NewMessage(llm.RoleUser, llm.NewTextPart(string(rec.Value)))
		}
	}

	client, err := durable.NewClient(t.Config)
	if err != nil {
		return err
	}
	defer client.Close()

	opts := append([]kgo.Opt{
		kgo.SeedBrokers(t.Config.Brokers...),
		kgo.ClientID("durable-trigger"),
		kgo.ConsumerGroup(t.ConsumerGroup),
		kgo.ConsumeTopics(t.Source),
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
	}, t.Config.KafkaOpts...)

	cl, err := kgo.NewClient(opts...)
	if err != nil {
		return fmt.Errorf("trigger: kafka client: %w", err)
	}
	defer cl.Close()

	t.Logger.Info("trigger started", "source", t.Source, "agent", t.Agent)

	for {
		f := cl.PollFetches(ctx)
		if ctx.Err() != nil || f.IsClientClosed() {
			return nil
		}

		f.EachError(func(topic string, p int32, err error) {
			t.Logger.Error("fetch error", "topic", topic, "partition", p, "error", err)
		})

		for iter := f.RecordIter(); !iter.Done(); {
			rec := iter.Next()

			if err := t.start(ctx, client, rec); err != nil {
				t.Logger.Error("start run failed", "error", err)
			}
		}
	}
}

func (t *Trigger) start(ctx context.Context, client *durable.Client, rec *kgo.Record) error {
	id := string(rec.Key)
	if id == "" {
		id = strconv.Itoa(int(rec.Partition)) + "-" + strconv.FormatInt(rec.Offset, 10)
	}

	msg := t.Message(rec)

	return client.Start(ctx, durable.StartOptions{
		RunID:     t.RunIDPrefix + id,
		Agent:     t.Agent,
		Version:   t.Version,
		TaskQueue: t.TaskQueue,
		Message:   &msg,
		Metadata: map[string]any{
			"source_topic":     rec.Topic,
			"source_partition": rec.Partition,
			"source_offset":    rec.Offset,
			"source_key":       string(rec.Key),
		},
	})
}

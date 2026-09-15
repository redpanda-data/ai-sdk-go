// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kerr"
	"github.com/twmb/franz-go/pkg/kgo"
)

// DefaultTopicPrefix is used when Config.TopicPrefix is empty.
const DefaultTopicPrefix = "durable"

// Topics names the topics for one deployment. Runs are keyed by run id on
// every topic, so all records of a run share a partition.
type Topics struct {
	Prefix string
}

// Commands is the topic clients and workers write commands to.
func (t Topics) Commands() string { return t.prefix() + ".commands" }

// Journal is the append-only per-run event log the engine folds into state.
// It must have the same partition count as Commands.
func (t Topics) Journal() string { return t.prefix() + ".journal" }

// Tasks is the task-queue topic workers on queue consume.
func (t Topics) Tasks(queue string) string { return t.prefix() + ".tasks." + queue }

// State is the compacted topic holding the latest RunState per run.
func (t Topics) State() string { return t.prefix() + ".state" }

// Results is the default topic run results are published to.
func (t Topics) Results() string { return t.prefix() + ".results" }

// Config is the compacted topic holding rollout records.
func (t Topics) Config() string { return t.prefix() + ".config" }

// TopicSpec describes how one topic should be created.
type TopicSpec struct {
	Name       string
	Partitions int32
	Compacted  bool
}

// Specs returns the topics a deployment needs, with the given partition count
// for the log topics. Task-queue topics are created on first dispatch.
func (t Topics) Specs(partitions int32) []TopicSpec {
	return []TopicSpec{
		{Name: t.Commands(), Partitions: partitions},
		{Name: t.Journal(), Partitions: partitions},
		{Name: t.State(), Partitions: 1, Compacted: true},
		{Name: t.Results(), Partitions: partitions},
		{Name: t.Config(), Partitions: 1, Compacted: true},
	}
}

// EnsureTopics creates the topics that do not exist yet. Existing topics are
// left untouched, so a mismatched partition count on Commands vs Journal is
// reported as an error rather than silently accepted.
func EnsureTopics(ctx context.Context, cl *kgo.Client, specs []TopicSpec, replicationFactor int16) error {
	adm := kadm.NewClient(cl)

	existing, err := adm.ListTopics(ctx)
	if err != nil {
		return fmt.Errorf("durable: list topics: %w", err)
	}

	var errs []error

	for _, spec := range specs {
		if existing.Has(spec.Name) {
			continue
		}

		configs := map[string]*string{}
		if spec.Compacted {
			configs["cleanup.policy"] = new("compact")
		}

		resp, err := adm.CreateTopic(ctx, spec.Partitions, replicationFactor, configs, spec.Name)
		if err != nil && !errors.Is(err, kerr.TopicAlreadyExists) {
			errs = append(errs, fmt.Errorf("durable: create topic %s: %w", spec.Name, err))

			continue
		}

		if resp.Err != nil && !errors.Is(resp.Err, kerr.TopicAlreadyExists) {
			errs = append(errs, fmt.Errorf("durable: create topic %s: %w", spec.Name, resp.Err))
		}
	}

	if err := errors.Join(errs...); err != nil {
		return err
	}

	return verifyPartitionParity(ctx, adm, specs)
}

// ErrPartitionMismatch is returned when the command and journal topics have
// different partition counts.
var ErrPartitionMismatch = errors.New("durable: commands and journal topics must have the same partition count")

func verifyPartitionParity(ctx context.Context, adm *kadm.Client, specs []TopicSpec) error {
	var cmd, jrn string

	for _, s := range specs {
		switch {
		case strings.HasSuffix(s.Name, ".commands"):
			cmd = s.Name
		case strings.HasSuffix(s.Name, ".journal"):
			jrn = s.Name
		}
	}

	if cmd == "" || jrn == "" {
		return nil
	}

	details, err := adm.ListTopics(ctx, cmd, jrn)
	if err != nil {
		return fmt.Errorf("durable: describe topics: %w", err)
	}

	if len(details[cmd].Partitions) != len(details[jrn].Partitions) {
		return fmt.Errorf("%w: %s=%d %s=%d", ErrPartitionMismatch,
			cmd, len(details[cmd].Partitions), jrn, len(details[jrn].Partitions))
	}

	return nil
}

func (t Topics) prefix() string {
	if t.Prefix == "" {
		return DefaultTopicPrefix
	}

	return t.Prefix
}

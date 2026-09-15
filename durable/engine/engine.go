// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/durable"
)

// Options configures an Engine.
type Options struct {
	Brokers     []string
	TopicPrefix string
	// ConsumerGroup for the command topic. Every engine instance in a
	// deployment shares it; partitions of the command and journal topics are
	// assigned together. Default "durable-engine".
	ConsumerGroup string
	// LeaseDuration is how long a worker may hold a task without reporting
	// before the engine re-dispatches it. Default 5m. Every accepted worker
	// command refreshes the lease.
	LeaseDuration time.Duration
	// Partitions for the command/journal topics when EnsureTopics creates them.
	// Default 4.
	Partitions int32
	// ReplicationFactor for created topics. Default -1 (broker default).
	ReplicationFactor int16
	// HTTPAddr serves the read API and metrics. Empty disables the server.
	HTTPAddr string
	// KafkaOpts are appended to every franz-go client (TLS, SASL, ...).
	KafkaOpts []kgo.Opt
	Logger    *slog.Logger
	// Now overrides the clock (tests).
	Now func() time.Time
}

func (o *Options) defaults() {
	if o.ConsumerGroup == "" {
		o.ConsumerGroup = "durable-engine"
	}

	if o.LeaseDuration <= 0 {
		o.LeaseDuration = 5 * time.Minute
	}

	if o.Partitions <= 0 {
		o.Partitions = 4
	}

	if o.ReplicationFactor == 0 {
		o.ReplicationFactor = -1
	}

	if o.Logger == nil {
		o.Logger = slog.Default()
	}

	if o.Now == nil {
		o.Now = func() time.Time { return time.Now().UTC() }
	}
}

// Engine is the durable execution controller. Create with New, then Run.
type Engine struct {
	opts   Options
	topics durable.Topics
	log    *slog.Logger

	// Metrics is exported for embedding services.
	Metrics Metrics

	mu       sync.RWMutex
	runs     map[string]*run
	timers   timerHeap
	rollouts map[string]durable.Rollout

	cl      *kgo.Client // command consumer + task/state/result producer
	journal *kgo.Client // journal producer, manual partitioning
	wake    chan struct{}
	ready   chan struct{}
}

// New creates an engine. Nothing is connected until Run.
func New(opts Options) *Engine {
	opts.defaults()

	return &Engine{
		opts:     opts,
		topics:   durable.Topics{Prefix: opts.TopicPrefix},
		log:      opts.Logger,
		runs:     map[string]*run{},
		rollouts: map[string]durable.Rollout{},
		wake:     make(chan struct{}, 1),
		ready:    make(chan struct{}),
	}
}

// Ready is closed once the engine has joined the group and finished its first
// journal replay. Useful for tests and readiness probes.
func (e *Engine) Ready() <-chan struct{} { return e.ready }

// Run creates topics, joins the consumer group, replays the journal for the
// assigned partitions and processes commands and timers until ctx ends.
func (e *Engine) Run(ctx context.Context) error {
	admin, err := kgo.NewClient(e.kafkaOpts(kgo.ClientID("durable-engine-admin"))...)
	if err != nil {
		return fmt.Errorf("engine: admin client: %w", err)
	}

	err = durable.EnsureTopics(ctx, admin, e.topics.Specs(e.opts.Partitions), e.opts.ReplicationFactor)

	admin.Close()

	if err != nil {
		return err
	}

	e.journal, err = kgo.NewClient(e.kafkaOpts(
		kgo.ClientID("durable-engine-journal"),
		kgo.RecordPartitioner(kgo.ManualPartitioner()),
	)...)
	if err != nil {
		return fmt.Errorf("engine: journal client: %w", err)
	}
	defer e.journal.Close()

	var readyOnce sync.Once

	e.cl, err = kgo.NewClient(e.kafkaOpts(
		kgo.ClientID("durable-engine"),
		kgo.ConsumerGroup(e.opts.ConsumerGroup),
		kgo.ConsumeTopics(e.topics.Commands()),
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
		kgo.DisableAutoCommit(),
		kgo.BlockRebalanceOnPoll(),
		kgo.RebalanceTimeout(5*time.Minute),
		kgo.AllowAutoTopicCreation(),
		kgo.OnPartitionsAssigned(func(ctx context.Context, _ *kgo.Client, assigned map[string][]int32) {
			e.onAssigned(ctx, assigned[e.topics.Commands()])
			readyOnce.Do(func() { close(e.ready) })
		}),
		kgo.OnPartitionsRevoked(func(_ context.Context, _ *kgo.Client, revoked map[string][]int32) {
			e.onRevoked(revoked[e.topics.Commands()])
		}),
		kgo.OnPartitionsLost(func(_ context.Context, _ *kgo.Client, lost map[string][]int32) {
			e.onRevoked(lost[e.topics.Commands()])
		}),
	)...)
	if err != nil {
		return fmt.Errorf("engine: command client: %w", err)
	}
	defer e.cl.Close()

	configCtx, cancelConfig := context.WithCancel(ctx)
	defer cancelConfig()

	go e.watchConfig(configCtx)

	if e.opts.HTTPAddr != "" {
		srv := &http.Server{Addr: e.opts.HTTPAddr, Handler: e.Handler(), ReadHeaderTimeout: 5 * time.Second}

		var lc net.ListenConfig

		ln, err := lc.Listen(ctx, "tcp", e.opts.HTTPAddr)
		if err != nil {
			return fmt.Errorf("engine: listen %s: %w", e.opts.HTTPAddr, err)
		}

		go func() {
			if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
				e.log.Error("http server stopped", "error", err)
			}
		}()

		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
			defer cancel()

			_ = srv.Shutdown(shutdownCtx)
		}()

		e.log.Info("engine http api listening", "addr", ln.Addr().String())
	}

	e.loop(ctx)

	return nil
}

// loop interleaves command batches and timers on one goroutine, so every state
// change (command-driven or timer-driven) is serialised without extra locking.
func (e *Engine) loop(ctx context.Context) {
	fetches := make(chan kgo.Fetches)

	go func() {
		defer close(fetches)

		for {
			f := e.cl.PollFetches(ctx)
			if ctx.Err() != nil || f.IsClientClosed() {
				return
			}

			select {
			case fetches <- f:
			case <-ctx.Done():
				return
			}
		}
	}()

	for {
		e.mu.RLock()
		next, hasTimer := e.timers.peek()
		e.mu.RUnlock()

		wait := time.Hour
		if hasTimer {
			wait = max(time.Until(next.at), 0)
		}

		t := time.NewTimer(wait)

		select {
		case <-ctx.Done():
			t.Stop()

			return
		case <-e.wake:
			t.Stop()
		case <-t.C:
			e.fireTimers(ctx)
		case f, ok := <-fetches:
			t.Stop()

			if !ok {
				return
			}

			e.processFetches(ctx, f)
			e.cl.AllowRebalance()
		}
	}
}

func (e *Engine) processFetches(ctx context.Context, f kgo.Fetches) {
	f.EachError(func(topic string, p int32, err error) {
		e.log.Error("fetch error", "topic", topic, "partition", p, "error", err)
	})

	var toCommit []*kgo.Record

	for iter := f.RecordIter(); !iter.Done(); {
		rec := iter.Next()

		var cmd durable.Command
		if err := json.Unmarshal(rec.Value, &cmd); err != nil {
			e.log.Error("dropping undecodable command", "partition", rec.Partition, "offset", rec.Offset, "error", err)
			e.Metrics.CommandsError.Add(1)
		} else if err := e.handleCommand(ctx, rec.Partition, cmd); err != nil {
			e.log.Error("command failed", "run_id", cmd.RunID, "type", cmd.Type, "error", err)
		}

		toCommit = append(toCommit, rec)
	}

	if len(toCommit) > 0 {
		if err := e.cl.CommitRecords(ctx, toCommit...); err != nil && ctx.Err() == nil {
			e.log.Error("commit failed; commands will be re-processed and deduplicated", "error", err)
		}
	}
}

// handleCommand runs one command through the state machine, journals the
// resulting records, then performs side effects.
func (e *Engine) handleCommand(ctx context.Context, partition int32, cmd durable.Command) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := e.opts.Now()
	r := e.runs[cmd.RunID]

	if r != nil && r.hasSeen(cmd.CommandID) {
		e.Metrics.CommandsDup.Add(1)

		return nil
	}

	var rollout *durable.Rollout
	if ro, ok := e.rollouts[cmd.Agent]; ok {
		rollout = &ro
	}

	recs, err := transition(r, cmd, now, rollout)
	if err != nil {
		switch {
		case errors.Is(err, errStaleToken):
			e.Metrics.CommandsStale.Add(1)
			e.log.Warn("dropping command from stale attempt", "run_id", cmd.RunID, "type", cmd.Type, "token", cmd.TaskToken)

			return nil
		case errors.Is(err, errRunOpen), errors.Is(err, errRunClosed), errors.Is(err, errUnknownRun):
			e.log.Debug("command ignored", "run_id", cmd.RunID, "type", cmd.Type, "reason", err)

			return nil
		default:
			e.Metrics.CommandsError.Add(1)

			return err
		}
	}

	if len(recs) == 0 {
		if r != nil {
			r.markSeen(cmd.CommandID)
			e.refreshLease(r, now)
		}

		return nil
	}

	if r == nil {
		r = newRun(partition)
		e.runs[cmd.RunID] = r
	}

	if cmd.TaskToken != "" {
		e.refreshLease(r, now)
	}

	if err := e.commit(ctx, r, cmd.RunID, cmd.CommandID, now, recs); err != nil {
		return err
	}

	return e.afterCommit(ctx, r, now)
}

// commit journals records (acks=all) and only then applies them.
func (e *Engine) commit(ctx context.Context, r *run, runID, commandID string, now time.Time, recs []durable.Record) error {
	kafkaRecs := make([]*kgo.Record, 0, len(recs))
	seq := r.st.JournalLen

	for i := range recs {
		seq++
		recs[i].Seq = seq
		recs[i].RunID = runID
		recs[i].At = now
		recs[i].CommandID = commandID

		b, err := json.Marshal(recs[i])
		if err != nil {
			return fmt.Errorf("engine: marshal journal record: %w", err)
		}

		kafkaRecs = append(kafkaRecs, &kgo.Record{
			Topic: e.topics.Journal(), Key: []byte(runID), Value: b, Partition: r.partition,
		})
	}

	if err := e.journal.ProduceSync(ctx, kafkaRecs...).FirstErr(); err != nil {
		return fmt.Errorf("engine: journal append: %w", err)
	}

	for _, rec := range recs {
		r.apply(rec)
		e.observe(rec)
	}

	e.Metrics.JournalRecords.Add(int64(len(recs)))

	return nil
}

func (e *Engine) observe(rec durable.Record) {
	switch rec.Type {
	case durable.RecordRunStarted:
		e.Metrics.RunsStarted.Add(1)
	case durable.RecordRunCompleted:
		e.Metrics.RunsCompleted.Add(1)
	case durable.RecordRunFailed:
		e.Metrics.RunsFailed.Add(1)
	case durable.RecordRunCancelled:
		e.Metrics.RunsCancelled.Add(1)
	case durable.RecordAttemptFailed:
		e.Metrics.AttemptsFailed.Add(1)

		if rec.Exhausted {
			e.Metrics.RunsFailed.Add(1)
		}
	case durable.RecordRunSuspended:
		e.Metrics.Suspensions.Add(1)
	case durable.RecordInputReceived, durable.RecordInputDelivered:
		e.Metrics.InputsReceived.Add(1)
	case durable.RecordTimerFired:
		e.Metrics.TimersFired.Add(1)
	case durable.RecordLeaseExpired:
		e.Metrics.LeasesExpired.Add(1)
	case durable.RecordTaskDispatched:
		e.Metrics.TasksDispatch.Add(1)
	}
}

// afterCommit performs the side effects implied by the new state: dispatch,
// timers, snapshot, result. Must hold e.mu.
func (e *Engine) afterCommit(ctx context.Context, r *run, now time.Time) error {
	var errs []error

	if r.needsDispatch() {
		if err := e.dispatch(ctx, r, now); err != nil {
			errs = append(errs, err)
		}
	}

	e.armTimers(r)

	if err := e.snapshot(ctx, r); err != nil {
		errs = append(errs, err)
	}

	if r.st.Closed() && r.st.ClosedAt != nil && r.st.ClosedAt.Equal(now) {
		if err := e.publishResult(ctx, r); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// dispatch journals task_dispatched then writes the task. If the engine dies
// between the two, the lease recorded in the journal expires and re-dispatches.
func (e *Engine) dispatch(ctx context.Context, r *run, now time.Time) error {
	attempt := r.st.Attempt + 1
	token := durable.TaskToken(r.st.RunID, attempt)
	lease := now.Add(e.opts.LeaseDuration)

	rec := durable.Record{Type: durable.RecordTaskDispatched, TaskToken: token, Attempt: attempt, LeaseExpiresAt: &lease}
	if err := e.commit(ctx, r, r.st.RunID, "", now, []durable.Record{rec}); err != nil {
		return err
	}

	task := durable.Task{
		TaskToken:      token,
		RunID:          r.st.RunID,
		Agent:          r.st.Agent,
		Version:        r.st.Version,
		TaskQueue:      r.st.TaskQueue,
		Attempt:        attempt,
		DispatchedAt:   now,
		LeaseExpiresAt: lease,
		Messages:       r.st.Messages,
		Metadata:       r.st.Metadata,
		ToolResults:    r.st.ToolResults,
		Delivered:      r.st.Delivered,
	}

	return e.produceJSON(ctx, e.topics.Tasks(r.st.TaskQueue), r.st.RunID, task)
}

func (e *Engine) snapshot(ctx context.Context, r *run) error {
	return e.produceJSON(ctx, e.topics.State(), r.st.RunID, r.st)
}

func (e *Engine) publishResult(ctx context.Context, r *run) error {
	topic := r.st.ResultTopic
	if topic == "" {
		topic = e.topics.Results()
	}

	return e.produceJSON(ctx, topic, r.st.RunID, result(r.st))
}

func (e *Engine) produceJSON(ctx context.Context, topic, key string, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("engine: marshal %s record: %w", topic, err)
	}

	if err := e.cl.ProduceSync(ctx, &kgo.Record{Topic: topic, Key: []byte(key), Value: b}).FirstErr(); err != nil {
		return fmt.Errorf("engine: produce to %s: %w", topic, err)
	}

	return nil
}

func (e *Engine) refreshLease(r *run, now time.Time) {
	if r.st.CurrentTaskToken == "" {
		return
	}

	lease := now.Add(e.opts.LeaseDuration)
	r.st.LeaseExpiresAt = &lease
}

// armTimers pushes the timers implied by the run's state. Duplicates are
// harmless: firing re-validates against the state. Must hold e.mu.
func (e *Engine) armTimers(r *run) {
	st := &r.st

	switch {
	case st.Closed():
		return
	case st.CurrentTaskToken != "" && st.LeaseExpiresAt != nil:
		e.timers.push(timer{at: *st.LeaseExpiresAt, runID: st.RunID, kind: timerLease, token: st.CurrentTaskToken})
	case st.Status == durable.StatusRetrying && st.RetryAt != nil:
		e.timers.push(timer{at: *st.RetryAt, runID: st.RunID, kind: timerRetry})
	case st.Status == durable.StatusSuspended && st.Awaiting != nil &&
		st.Awaiting.Kind == durable.AwaitTimer && st.Awaiting.FireAt != nil:
		e.timers.push(timer{at: *st.Awaiting.FireAt, runID: st.RunID, kind: timerWait, toolCallID: st.Awaiting.ToolCallID})
	}

	if st.ExecutionTimeout > 0 {
		e.timers.push(timer{at: st.StartedAt.Add(st.ExecutionTimeout), runID: st.RunID, kind: timerTimeout})
	}

	select {
	case e.wake <- struct{}{}:
	default:
	}
}

func (e *Engine) fireTimers(ctx context.Context) {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := e.opts.Now()

	for _, t := range e.timers.due(now) {
		r, ok := e.runs[t.runID]
		if !ok || r.st.Closed() {
			continue
		}

		if err := e.fireTimer(ctx, r, t, now); err != nil {
			e.log.Error("timer failed", "run_id", t.runID, "kind", t.kind, "error", err)
		}
	}
}

func (e *Engine) fireTimer(ctx context.Context, r *run, t timer, now time.Time) error {
	st := &r.st

	switch t.kind {
	case timerLease:
		if st.CurrentTaskToken != t.token {
			return nil // attempt already finished
		}

		if st.LeaseExpiresAt != nil && st.LeaseExpiresAt.After(now) {
			e.timers.push(timer{at: *st.LeaseExpiresAt, runID: st.RunID, kind: timerLease, token: t.token})

			return nil // lease was refreshed by a worker command
		}

		e.log.Warn("lease expired; re-dispatching", "run_id", st.RunID, "token", t.token)

		rec := durable.Record{Type: durable.RecordLeaseExpired, TaskToken: t.token}
		if err := e.commit(ctx, r, st.RunID, "", now, []durable.Record{rec}); err != nil {
			return err
		}

	case timerRetry:
		if st.Status != durable.StatusRetrying || st.RetryAt == nil || st.RetryAt.After(now) {
			return nil
		}

		// Nothing to journal: task_dispatched is the record of the retry.
		st.Status = durable.StatusRunning
		st.RetryAt = nil

	case timerWait:
		aw := st.Awaiting
		if st.Status != durable.StatusSuspended || aw == nil || aw.ToolCallID != t.toolCallID {
			return nil
		}

		rec := durable.Record{Type: durable.RecordTimerFired, ToolCallID: aw.ToolCallID, Payload: timerPayload(now)}
		if err := e.commit(ctx, r, st.RunID, "", now, []durable.Record{rec}); err != nil {
			return err
		}

	case timerTimeout:
		if st.ExecutionTimeout <= 0 || st.StartedAt.Add(st.ExecutionTimeout).After(now) {
			return nil
		}

		rec := durable.Record{Type: durable.RecordRunFailed, Error: "execution timeout exceeded"}
		if err := e.commit(ctx, r, st.RunID, "", now, []durable.Record{rec}); err != nil {
			return err
		}
	}

	return e.afterCommit(ctx, r, now)
}

// onAssigned replays the journal for newly assigned partitions and arms the
// timers implied by the recovered state.
func (e *Engine) onAssigned(ctx context.Context, partitions []int32) {
	if len(partitions) == 0 {
		return
	}

	start := time.Now()

	if err := e.replay(ctx, partitions); err != nil {
		e.log.Error("journal replay failed", "partitions", partitions, "error", err)

		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	var dispatchable []*run

	for _, r := range e.runs {
		if !containsPartition(partitions, r.partition) {
			continue
		}

		e.armTimers(r)

		if r.needsDispatch() {
			dispatchable = append(dispatchable, r)
		}
	}

	now := e.opts.Now()

	for _, r := range dispatchable {
		if err := e.dispatch(ctx, r, now); err != nil {
			e.log.Error("re-dispatch after replay failed", "run_id", r.st.RunID, "error", err)
		}
	}

	e.Metrics.setReplay(time.Since(start).Milliseconds())
	e.log.Info("journal replay complete", "partitions", partitions, "runs", len(e.runs),
		"redispatched", len(dispatchable), "took", time.Since(start))
}

func (e *Engine) onRevoked(partitions []int32) {
	if len(partitions) == 0 {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	for id, r := range e.runs {
		if containsPartition(partitions, r.partition) {
			delete(e.runs, id)
		}
	}

	kept := e.timers[:0]

	for _, t := range e.timers {
		if r, ok := e.runs[t.runID]; ok && !containsPartition(partitions, r.partition) {
			kept = append(kept, t)
		}
	}

	e.timers = kept
	e.log.Info("partitions revoked; state dropped", "partitions", partitions)
}

// replay reads the journal partitions from the beginning to their current end
// and folds every record.
func (e *Engine) replay(ctx context.Context, partitions []int32) error {
	adm := kadm.NewClient(e.journal)

	ends, err := adm.ListEndOffsets(ctx, e.topics.Journal())
	if err != nil {
		return fmt.Errorf("engine: list journal end offsets: %w", err)
	}

	assign := map[int32]kgo.Offset{}
	remaining := map[int32]int64{}

	for _, p := range partitions {
		end, ok := ends.Lookup(e.topics.Journal(), p)
		if !ok || end.Offset == 0 {
			continue
		}

		assign[p] = kgo.NewOffset().AtStart()
		remaining[p] = end.Offset
	}

	if len(assign) == 0 {
		return nil
	}

	reader, err := kgo.NewClient(e.kafkaOpts(
		kgo.ClientID("durable-engine-replay"),
		kgo.ConsumePartitions(map[string]map[int32]kgo.Offset{e.topics.Journal(): assign}),
	)...)
	if err != nil {
		return fmt.Errorf("engine: replay client: %w", err)
	}
	defer reader.Close()

	e.mu.Lock()
	defer e.mu.Unlock()

	for len(remaining) > 0 {
		f := reader.PollFetches(ctx)
		if ctx.Err() != nil {
			return ctx.Err()
		}

		if err := f.Err(); err != nil {
			return fmt.Errorf("engine: replay fetch: %w", err)
		}

		for iter := f.RecordIter(); !iter.Done(); {
			rec := iter.Next()

			var jr durable.Record
			if err := json.Unmarshal(rec.Value, &jr); err != nil {
				e.log.Error("skipping undecodable journal record", "partition", rec.Partition, "offset", rec.Offset, "error", err)
			} else {
				r, ok := e.runs[jr.RunID]
				if !ok {
					r = newRun(rec.Partition)
					e.runs[jr.RunID] = r
				}

				r.apply(jr)
			}

			if rec.Offset+1 >= remaining[rec.Partition] {
				delete(remaining, rec.Partition)
			}
		}
	}

	return nil
}

// watchConfig follows the config topic and keeps rollouts current.
func (e *Engine) watchConfig(ctx context.Context) {
	cl, err := kgo.NewClient(e.kafkaOpts(
		kgo.ClientID("durable-engine-config"),
		kgo.ConsumeTopics(e.topics.Config()),
		kgo.ConsumeResetOffset(kgo.NewOffset().AtStart()),
	)...)
	if err != nil {
		e.log.Error("config watcher failed to start", "error", err)

		return
	}
	defer cl.Close()

	for {
		f := cl.PollFetches(ctx)
		if ctx.Err() != nil || f.IsClientClosed() {
			return
		}

		for iter := f.RecordIter(); !iter.Done(); {
			rec := iter.Next()
			e.applyConfig(string(rec.Key), rec.Value)
		}
	}
}

func (e *Engine) applyConfig(key string, value []byte) {
	e.mu.Lock()
	defer e.mu.Unlock()

	const prefix = "rollout/"
	if len(key) <= len(prefix) || key[:len(prefix)] != prefix {
		return
	}

	agentName := key[len(prefix):]

	if len(value) == 0 {
		delete(e.rollouts, agentName)

		return
	}

	var ro durable.Rollout
	if err := json.Unmarshal(value, &ro); err != nil {
		e.log.Error("ignoring malformed rollout record", "key", key, "error", err)

		return
	}

	e.rollouts[agentName] = ro
	e.log.Info("rollout updated", "agent", agentName, "targets", len(ro.Targets))
}

func containsPartition(ps []int32, p int32) bool {
	return slices.Contains(ps, p)
}

func (e *Engine) kafkaOpts(extra ...kgo.Opt) []kgo.Opt {
	opts := []kgo.Opt{
		kgo.SeedBrokers(e.opts.Brokers...),
		kgo.RequiredAcks(kgo.AllISRAcks()),
		kgo.ProducerBatchCompression(kgo.SnappyCompression()),
	}
	opts = append(opts, e.opts.KafkaOpts...)

	return append(opts, extra...)
}

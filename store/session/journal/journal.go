// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

// Package journal provides a session.Store backed by an append-only Redpanda
// topic: the durable log form of session persistence.
//
// The kvstore session store writes the whole session on every Save into a
// compacted topic, which keeps only the latest value. This store instead
// appends one record per new message, keyed by session id, so the topic is a
// complete, ordered transcript of every session. Save costs one small record
// per new message rather than a full rewrite, and History exposes the
// transcript that session.State.Messages deliberately does not preserve once
// compaction prunes it.
//
// Every store instance follows the whole topic and materialises sessions in
// memory (like kvstore); Load is a local read. Writes wait for the record to be
// acknowledged (acks=all) and then observed by the local consumer, so a Save
// followed by a Load on the same store returns the saved state.
package journal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/twmb/franz-go/pkg/kadm"
	"github.com/twmb/franz-go/pkg/kerr"
	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/store/session"
)

var _ session.Store = (*Store)(nil)

// Record types.
const (
	recordMessage  = "message"  // one appended message
	recordSnapshot = "snapshot" // full replacement (compaction rewrote history, or metadata changed)
	recordDelete   = "delete"
)

// Record is one journal entry.
type Record struct {
	Type      string `json:"type"`
	SessionID string `json:"session_id"`
	// Seq is the index the message occupies in Messages (type=message).
	Seq            int            `json:"seq,omitempty"`
	Message        *llm.Message   `json:"message,omitempty"`
	Messages       []llm.Message  `json:"messages,omitempty"`
	ConversationID string         `json:"conversation_id,omitempty"`
	Metadata       map[string]any `json:"metadata,omitempty"`
	At             time.Time      `json:"at"`
}

// Options configures a Store.
type Options struct {
	Brokers []string
	// Partitions and ReplicationFactor apply when the topic is created.
	Partitions        int32
	ReplicationFactor int16
	KafkaOpts         []kgo.Opt
	Logger            *slog.Logger
}

// Store is an append-only-topic session store.
type Store struct {
	topic string
	log   *slog.Logger

	prod *kgo.Client
	cons *kgo.Client

	mu       sync.RWMutex
	sessions map[string]*entry
	// history keeps the raw records per session for History.
	history map[string][]Record
	// offsets tracks how far the local consumer has read, per partition.
	offsets map[int32]int64
	cond    *sync.Cond

	cancel context.CancelFunc
	done   chan struct{}
}

type entry struct {
	state     session.State
	updatedAt time.Time
}

// New creates the topic if needed, reads it to the end and starts following it.
func New(ctx context.Context, topic string, opts Options) (*Store, error) {
	if topic == "" {
		return nil, errors.New("journal: topic is required")
	}

	if opts.Logger == nil {
		opts.Logger = slog.Default()
	}

	if opts.Partitions <= 0 {
		opts.Partitions = 1
	}

	if opts.ReplicationFactor == 0 {
		opts.ReplicationFactor = -1
	}

	base := append([]kgo.Opt{kgo.SeedBrokers(opts.Brokers...)}, opts.KafkaOpts...)

	prod, err := kgo.NewClient(append(base, kgo.ClientID("session-journal"), kgo.RequiredAcks(kgo.AllISRAcks()))...)
	if err != nil {
		return nil, fmt.Errorf("journal: producer: %w", err)
	}

	adm := kadm.NewClient(prod)

	resp, err := adm.CreateTopic(ctx, opts.Partitions, opts.ReplicationFactor, nil, topic)
	if err == nil {
		err = resp.Err
	}

	if err != nil && !errors.Is(err, kerr.TopicAlreadyExists) {
		prod.Close()

		return nil, fmt.Errorf("journal: create topic: %w", err)
	}

	ends, err := adm.ListEndOffsets(ctx, topic)
	if err != nil {
		prod.Close()

		return nil, fmt.Errorf("journal: list end offsets: %w", err)
	}

	assign := map[int32]kgo.Offset{}
	target := map[int32]int64{}

	ends.Each(func(o kadm.ListedOffset) {
		assign[o.Partition] = kgo.NewOffset().AtStart()
		target[o.Partition] = o.Offset
	})

	cons, err := kgo.NewClient(append(base,
		kgo.ClientID("session-journal-follower"),
		kgo.ConsumePartitions(map[string]map[int32]kgo.Offset{topic: assign}),
	)...)
	if err != nil {
		prod.Close()

		return nil, fmt.Errorf("journal: consumer: %w", err)
	}

	s := &Store{
		topic:    topic,
		log:      opts.Logger,
		prod:     prod,
		cons:     cons,
		sessions: map[string]*entry{},
		history:  map[string][]Record{},
		offsets:  map[int32]int64{},
		done:     make(chan struct{}),
	}
	s.cond = sync.NewCond(&s.mu)

	for p := range assign {
		s.offsets[p] = -1
	}

	followCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	s.cancel = cancel

	go s.follow(followCtx)

	// Bootstrap: wait until the follower has read to the end offsets.
	for p, end := range target {
		if end == 0 {
			continue
		}

		if err := s.waitFor(ctx, p, end-1); err != nil {
			s.Close()

			return nil, fmt.Errorf("journal: bootstrap: %w", err)
		}
	}

	return s, nil
}

// Close stops the follower and releases clients.
func (s *Store) Close() {
	s.cancel()
	<-s.done
	s.cons.Close()
	s.prod.Close()
}

// waitFor blocks until the follower has consumed offset on partition p.

// Load implements session.Store.
func (s *Store) Load(_ context.Context, sessionID string) (*session.State, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	e, ok := s.sessions[sessionID]
	if !ok {
		return nil, session.ErrNotFound
	}

	st := e.state.Clone()
	st.UpdatedAt = e.updatedAt

	return st, nil
}

// Save implements session.Store. New messages are appended as individual
// records; any other change (pruned history, edited earlier messages, new
// metadata) writes a snapshot.
func (s *Store) Save(ctx context.Context, state *session.State) error {
	if state == nil || state.ID == "" {
		return errors.New("journal: state with ID is required")
	}

	s.mu.RLock()
	cur, exists := s.sessions[state.ID]

	var prevLen int

	appendOnly := exists && isPrefix(cur.state, *state)
	if exists {
		prevLen = len(cur.state.Messages)
	}

	s.mu.RUnlock()

	now := time.Now().UTC()

	var recs []Record

	if appendOnly {
		for i := prevLen; i < len(state.Messages); i++ {
			m := llm.CloneMessage(state.Messages[i])
			recs = append(recs, Record{Type: recordMessage, SessionID: state.ID, Seq: i, Message: &m, At: now})
		}

		if len(recs) == 0 {
			return nil
		}
	} else {
		snap := state.Clone()
		recs = append(recs, Record{
			Type: recordSnapshot, SessionID: state.ID, Messages: snap.Messages,
			ConversationID: snap.ConversationID, Metadata: snap.Metadata, At: now,
		})
	}

	return s.produce(ctx, state.ID, recs)
}

// Delete implements session.Store.
func (s *Store) Delete(ctx context.Context, sessionID string) error {
	return s.produce(ctx, sessionID, []Record{{Type: recordDelete, SessionID: sessionID, At: time.Now().UTC()}})
}

// List implements session.Store. Summaries come from the materialised view.
func (s *Store) List(_ context.Context, req *session.ListRequest) (*session.ListResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	all := make([]session.Summary, 0, len(s.sessions))
	for _, e := range s.sessions {
		all = append(all, session.Summary{
			ID: e.state.ID, ConversationID: e.state.ConversationID,
			Metadata: maps.Clone(e.state.Metadata), UpdatedAt: e.updatedAt,
		})
	}

	sortSummaries(all)

	start := 0

	if req != nil && req.PageToken != "" {
		for i, sm := range all {
			if sm.ID == req.PageToken {
				start = i + 1

				break
			}
		}
	}

	size := 50
	if req != nil && req.PageSize > 0 {
		size = int(req.PageSize)
	}

	end := min(start+size, len(all))
	page := all[start:end]

	resp := &session.ListResponse{Sessions: page}
	if end < len(all) && len(page) > 0 {
		resp.NextPageToken = page[len(page)-1].ID
	}

	return resp, nil
}

// History returns every journal record for a session in order, including
// messages that later compaction removed from the session.
func (s *Store) History(sessionID string) []Record {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := make([]Record, len(s.history[sessionID]))
	copy(out, s.history[sessionID])

	return out
}

func (s *Store) produce(ctx context.Context, key string, recs []Record) error {
	kafkaRecs := make([]*kgo.Record, 0, len(recs))

	for _, r := range recs {
		b, err := json.Marshal(r)
		if err != nil {
			return fmt.Errorf("journal: marshal record: %w", err)
		}

		kafkaRecs = append(kafkaRecs, &kgo.Record{Topic: s.topic, Key: []byte(key), Value: b})
	}

	results := s.prod.ProduceSync(ctx, kafkaRecs...)
	if err := results.FirstErr(); err != nil {
		return fmt.Errorf("journal: produce: %w", err)
	}

	last := results[len(results)-1].Record

	return s.waitFor(ctx, last.Partition, last.Offset)
}

// isPrefix reports whether next only appends messages to cur, with identical
// metadata, so the delta can be journaled as individual message records.
func isPrefix(cur, next session.State) bool {
	if len(next.Messages) < len(cur.Messages) || cur.ConversationID != next.ConversationID {
		return false
	}

	if !metadataEqual(cur.Metadata, next.Metadata) {
		return false
	}

	for i := range cur.Messages {
		a, err1 := json.Marshal(cur.Messages[i])
		b, err2 := json.Marshal(next.Messages[i])

		if err1 != nil || err2 != nil || string(a) != string(b) {
			return false
		}
	}

	return true
}

func metadataEqual(a, b map[string]any) bool {
	if len(a) != len(b) {
		return false
	}

	ja, err1 := json.Marshal(a)
	jb, err2 := json.Marshal(b)

	return err1 == nil && err2 == nil && string(ja) == string(jb)
}

func sortSummaries(all []session.Summary) {
	slices.SortFunc(all, func(a, b session.Summary) int {
		if !a.UpdatedAt.Equal(b.UpdatedAt) {
			return b.UpdatedAt.Compare(a.UpdatedAt)
		}

		return strings.Compare(a.ID, b.ID)
	})
}

func (s *Store) follow(ctx context.Context) {
	defer close(s.done)

	for {
		f := s.cons.PollFetches(ctx)
		if ctx.Err() != nil || f.IsClientClosed() {
			return
		}

		f.EachError(func(_ string, p int32, err error) {
			s.log.Error("journal fetch error", "partition", p, "error", err)
		})

		s.mu.Lock()

		for iter := f.RecordIter(); !iter.Done(); {
			rec := iter.Next()

			var jr Record
			if err := json.Unmarshal(rec.Value, &jr); err != nil {
				s.log.Error("skipping undecodable journal record", "offset", rec.Offset, "error", err)
			} else {
				s.applyLocked(jr)
			}

			s.offsets[rec.Partition] = rec.Offset
		}

		s.mu.Unlock()
		s.cond.Broadcast()
	}
}

func (s *Store) applyLocked(rec Record) {
	switch rec.Type {
	case recordDelete:
		delete(s.sessions, rec.SessionID)
		delete(s.history, rec.SessionID)

		return
	case recordSnapshot:
		st := session.State{ID: rec.SessionID, ConversationID: rec.ConversationID, Messages: rec.Messages, Metadata: rec.Metadata}
		if st.Messages == nil {
			st.Messages = []llm.Message{}
		}

		s.sessions[rec.SessionID] = &entry{state: st, updatedAt: rec.At}
	case recordMessage:
		e, ok := s.sessions[rec.SessionID]
		if !ok {
			e = &entry{state: session.State{ID: rec.SessionID, Messages: []llm.Message{}, Metadata: map[string]any{}}}
			s.sessions[rec.SessionID] = e
		}

		if rec.Message != nil && rec.Seq == len(e.state.Messages) {
			e.state.Messages = append(e.state.Messages, *rec.Message)
		}

		e.updatedAt = rec.At
	}

	s.history[rec.SessionID] = append(s.history[rec.SessionID], rec)
}

func (s *Store) waitFor(ctx context.Context, p int32, offset int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for s.offsets[p] < offset {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		// sync.Cond has no deadline, so a watchdog broadcast bounds the wait
		// and lets the ctx.Err check above run again.
		waitCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)

		go func() {
			<-waitCtx.Done()
			s.cond.Broadcast()
		}()

		s.cond.Wait()
		cancel()
	}

	return nil
}

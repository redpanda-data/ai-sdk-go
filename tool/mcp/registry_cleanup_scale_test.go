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

package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// cleanupScaleResult is one machine-readable matrix row, logged with the
// CLEANUP_SCALE prefix. Logical users are distinct credential partitions;
// concurrency is bounded separately by workers and max_connections.
type cleanupScaleResult struct {
	Legacy              bool    `json:"legacy_protocol"`
	Scenario            string  `json:"scenario"`
	LogicalUsers        int     `json:"logical_users"`
	Operations          int     `json:"operations"`
	Workers             int     `json:"workers"`
	MaxConnections      int     `json:"max_connections"`
	PeakOperations      int64   `json:"peak_concurrent_operations"`
	Verified            int     `json:"verified_results"`
	Errors              int     `json:"errors"`
	DeadlineCohort      int     `json:"deadline_cohort"`
	DeadlineErrors      int     `json:"deadline_errors"`
	DeadlineOverrunMs   float64 `json:"deadline_max_overrun_ms"`
	Opens               int     `json:"opens"`
	Reconnects          int     `json:"reconnects"`
	Deletes             int     `json:"deletes"`
	DeletePeak          int     `json:"delete_peak"`
	Cancels             int     `json:"cancel_notifications"`
	CallsDuringDelete   int     `json:"calls_during_delete"`
	HeldDeleteOverlap   int     `json:"held_delete_overlap_calls"`
	CallPeak            int     `json:"call_peak"`
	LivePeak            int     `json:"live_session_peak"`
	ListenPeak          int     `json:"listen_stream_peak"`
	OccupiedPeak        int     `json:"sampled_resident_plus_closing_peak"`
	ClosingPeak         int     `json:"sampled_closing_peak"`
	WaitingPeak         int     `json:"sampled_waiting_peak"`
	Dials               int     `json:"client_dials"`
	SocketPeak          int     `json:"client_socket_peak"`
	ServerConnPeak      int     `json:"server_conn_peak"`
	GoroutinePeak       int     `json:"sampled_goroutine_peak"`
	ShutdownBlocked     bool    `json:"shutdown_blocked_while_held"`
	ShutdownDrainMs     float64 `json:"shutdown_drain_ms"`
	DurationMs          float64 `json:"duration_ms"`
	ResidentAfterChurn  int     `json:"resident_after_churn"`
	SocketsAfterClose   int     `json:"client_sockets_after_close"`
	ServerConnsAfter    int     `json:"server_conns_after_close"`
	GoroutinesAfter     int     `json:"goroutines_after_close"`
	GoroutinesBaseline  int     `json:"goroutines_baseline"`
	LiveAfterClose      int     `json:"live_sessions_after_close"`
	DeletedOncePerOpen  bool    `json:"deleted_once_per_session"`
	OrphanedSessions    int     `json:"orphaned_sessions"`
	IdentityMismatches  int     `json:"identity_mismatches"`
	PoolSettledAfterAll bool    `json:"pool_settled"`
}

// row is the logged form of r. Metrics that do not apply are null and listed
// under not_applicable: DELETE and live-session metrics for the stateless
// modern protocol, listen streams for legacy, and the deadline cohort,
// sampler and held-shutdown phases, which the one-partition run does not
// execute. A zero that remains is a measured zero.
func (r cleanupScaleResult) row(t *testing.T) []byte {
	t.Helper()

	type plain cleanupScaleResult

	payload, err := json.Marshal(plain(r))
	require.NoError(t, err)

	var row map[string]any
	require.NoError(t, json.Unmarshal(payload, &row))

	na := []string{"listen_stream_peak"}
	if !r.Legacy {
		na = []string{"delete_peak", "calls_during_delete", "live_session_peak", "live_sessions_after_close", "deleted_once_per_session", "orphaned_sessions"}
	}

	if !r.Legacy || r.LogicalUsers <= r.MaxConnections {
		na = append(na, "held_delete_overlap_calls")
	}

	if r.Scenario == cleanupOnePartition {
		na = append(na, "deadline_cohort", "deadline_errors", "deadline_max_overrun_ms", "sampled_resident_plus_closing_peak",
			"sampled_closing_peak", "sampled_waiting_peak", "sampled_goroutine_peak", "shutdown_blocked_while_held",
			"shutdown_drain_ms", "resident_after_churn", "goroutines_after_close", "goroutines_baseline")
	}

	for _, key := range na {
		require.Contains(t, row, key)
		row[key] = nil
	}

	row["not_applicable"] = na

	payload, err = json.Marshal(row)
	require.NoError(t, err)

	return payload
}

const cleanupOnePartition = "one_partition"

type cleanupSampler struct {
	stop chan struct{}
	done chan struct{}

	occupied, closing, waiting, goroutines int
}

// sampleCleanupPool records pool peaks at every registry wakeup, plus a fast
// tick for transitions that do not signal. Exact bounds come from the peer.
func sampleCleanupPool(r *Registry) *cleanupSampler {
	s := &cleanupSampler{stop: make(chan struct{}), done: make(chan struct{})}

	go func() {
		defer close(s.done)

		tick := time.NewTicker(200 * time.Microsecond)
		defer tick.Stop()

		for n := 0; ; n++ {
			r.mu.Lock()
			s.occupied = max(s.occupied, len(r.entries)+r.closing)
			s.closing = max(s.closing, r.closing)
			s.waiting = max(s.waiting, r.waiting)
			changed := r.changed
			r.mu.Unlock()

			if n%64 == 0 {
				s.goroutines = max(s.goroutines, runtime.NumGoroutine())
			}

			select {
			case <-changed:
			case <-tick.C:
			case <-s.stop:
				return
			}
		}
	}()

	return s
}

func (s *cleanupSampler) finish() {
	close(s.stop)
	<-s.done
}

// paceCleanupDeletes holds every legacy DELETE at the peer until a second one
// is waiting, or for one millisecond, so evictions overlap one another and
// later operations without ever waiting for go-sdk's five-second timeout.
func paceCleanupDeletes(peer *cleanupPeer) func() {
	tokens := peer.throttle()
	stop := make(chan struct{})
	done := make(chan struct{})

	go func() {
		defer close(done)

		for {
			peer.mu.Lock()
			inflight, changed := peer.stats.DeleteInflight, peer.changed
			peer.mu.Unlock()

			if inflight < 2 {
				select {
				case <-changed:
					continue
				case <-time.After(time.Millisecond):
				case <-stop:
					return
				}

				if inflight == 0 {
					continue
				}
			}

			select {
			case tokens <- struct{}{}:
			case <-stop:
				return
			}
		}
	}()

	return func() {
		close(stop)
		<-done
		peer.release()
	}
}

func cleanupScaleUser(i int) string { return fmt.Sprintf("user-%05d", i) }

func cleanupEnter(inflight, peak *atomic.Int64) {
	n := inflight.Add(1)
	for p := peak.Load(); n > p; p = peak.Load() {
		if peak.CompareAndSwap(p, n) {
			return
		}
	}
}

func cleanupScaleExecute(ctx context.Context, r *Registry, partition string) error {
	response, err := r.Execute(cleanupCaller(ctx, partition), cleanupRequest())
	if err != nil {
		return err
	}

	if response.IsError || !strings.Contains(string(response.Result), `caller=Bearer `+partition+`"`) {
		return fmt.Errorf("%s got %s", partition, response.Result)
	}

	return nil
}

type cleanupScaleCase struct {
	legacy     bool
	users      int
	opsPerUser int
	workers    int
	slots      int
}

func runCleanupScale(t *testing.T, c cleanupScaleCase) cleanupScaleResult {
	t.Helper()

	result := cleanupScaleResult{
		Legacy: c.legacy, Scenario: "distinct_users", LogicalUsers: c.users,
		Operations: c.users * c.opsPerUser, Workers: c.workers, MaxConnections: c.slots,
		GoroutinesBaseline: runtime.NumGoroutine(),
	}
	started := time.Now()

	f := newCleanupFixture(t, c.legacy, WithRegistryMaxConnections(c.slots))
	sampler := sampleCleanupPool(f.registry)

	var stopPacing func()
	if c.legacy {
		stopPacing = paceCleanupDeletes(f.peer)
	}

	jobs := make(chan string)
	counts := make([]atomic.Int32, c.users)

	var (
		inflight, peak atomic.Int64
		errs           atomic.Int32
		firstErr       atomic.Pointer[error]
		wg             sync.WaitGroup
	)

	for range c.workers {
		wg.Go(func() {
			for partition := range jobs {
				cleanupEnter(&inflight, &peak)

				ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
				err := cleanupScaleExecute(ctx, f.registry, partition)

				cancel()
				inflight.Add(-1)

				if err != nil {
					errs.Add(1)
					firstErr.CompareAndSwap(nil, &err)

					continue
				}

				var index int
				_, _ = fmt.Sscanf(partition, "user-%05d", &index)
				counts[index].Add(1)
			}
		})
	}

	// Hold the first calls at the peer until every slot is in a call and every
	// other worker waits for capacity: the peak is then exactly the workers.
	f.peer.holdCalls()

	fed := make(chan struct{})

	go func() {
		defer close(fed)

		for round := range c.opsPerUser {
			for i := range c.users {
				jobs <- cleanupScaleUser((i + round*7) % c.users)
			}
		}

		close(jobs)
	}()

	busy := min(c.workers, c.slots)

	cleanupEventually(t, "all workers in flight", func() bool {
		return f.peer.snapshot().CallInflight == busy && cleanupPool(f.registry).Waiting == c.workers-busy
	})
	require.Equal(t, int64(c.workers), inflight.Load())
	f.peer.releaseCalls()
	<-fed
	wg.Wait()

	if stopPacing != nil {
		stopPacing()
	}

	if err := firstErr.Load(); err != nil {
		require.NoError(t, *err)
	}

	result.Errors = int(errs.Load())
	result.PeakOperations = peak.Load()

	for i := range counts {
		if n := int(counts[i].Load()); n == c.opsPerUser {
			result.Verified += n
		} else {
			t.Errorf("%s completed %d of %d operations", cleanupScaleUser(i), n, c.opsPerUser)
		}
	}

	state := cleanupSettle(t, f.registry)
	result.ResidentAfterChurn = state.Resident

	var overlapOpens int

	if c.legacy && c.users > c.slots {
		result.HeldDeleteOverlap, overlapOpens = runCleanupHeldDeleteOverlap(t, f, c)
	}

	result.DeadlineCohort, result.DeadlineErrors, result.DeadlineOverrunMs = runCleanupDeadlineCohort(t, f, c, false)

	cohort := result.DeadlineCohort
	for i := range cohort {
		ctx, cancel := context.WithTimeout(t.Context(), cleanupBarrier)
		require.NoError(t, cleanupScaleExecute(ctx, f.registry, fmt.Sprintf("late-%02d", i)))
		cancel()
	}

	cleanupSettle(t, f.registry)

	f.holdCleanup()

	blocked, stop := context.WithTimeout(t.Context(), 20*time.Millisecond)
	result.ShutdownBlocked = errors.Is(f.registry.Shutdown(blocked), context.DeadlineExceeded)

	stop()

	drainStart := time.Now()

	f.peer.release()

	drain, stopDrain := context.WithTimeout(t.Context(), cleanupBarrier)
	require.NoError(t, f.registry.Shutdown(drain))
	stopDrain()

	result.ShutdownDrainMs = float64(time.Since(drainStart).Microseconds()) / 1000

	sampler.finish()

	f.client.transport.CloseIdleConnections()
	cleanupEventually(t, "client sockets closed", func() bool { return f.client.snapshot().Sockets == 0 })
	cleanupEventually(t, "server connections closed", func() bool { return f.peer.snapshot().Conns == 0 })

	peer, client := f.peer.snapshot(), f.client.snapshot()
	result.Opens, result.Deletes, result.DeletePeak, result.Cancels = peer.Opens, peer.DeleteStarted, peer.DeletePeak, peer.Cancels
	result.Reconnects = peer.Opens - c.users - cohort - c.slots - overlapOpens
	result.CallsDuringDelete, result.CallPeak = peer.CallsDuringDelete, peer.CallPeak
	result.LivePeak, result.ListenPeak, result.ServerConnPeak = peer.LivePeak, peer.ListenPeak, peer.ConnPeak
	result.LiveAfterClose, result.ServerConnsAfter = peer.Live, peer.Conns
	result.Dials, result.SocketPeak, result.SocketsAfterClose = client.Dials, client.SocketPeak, client.Sockets
	result.OccupiedPeak, result.ClosingPeak, result.WaitingPeak = sampler.occupied, sampler.closing, sampler.waiting
	result.GoroutinePeak = sampler.goroutines
	result.IdentityMismatches = len(f.peer.identityMismatches())
	result.OrphanedSessions = len(f.peer.orphans())
	result.PoolSettledAfterAll = cleanupPool(f.registry) == cleanupPoolState{}
	result.DeletedOncePerOpen = true

	for _, n := range f.peer.deleteCounts() {
		result.DeletedOncePerOpen = result.DeletedOncePerOpen && n == 1
	}

	cleanupEventually(t, "goroutines released", func() bool { return runtime.NumGoroutine() <= result.GoroutinesBaseline+8 })

	result.GoroutinesAfter = runtime.NumGoroutine()
	result.DurationMs = float64(time.Since(started).Microseconds()) / 1000

	return result
}

// runCleanupHeldDeleteOverlap overlaps a business call with a legacy
// session-termination DELETE held at the peer, independent of churn timing.
// With the pool full, a fresh partition evicts exactly one idle LRU session
// and waits for that slot. A call on a different, most recently used resident
// partition needs no capacity and must complete while the DELETE is still held
// and before the newcomer is admitted. Churn may leave free slots (two workers
// on one partition can both evict), so fresh partitions fill them first. It
// returns the calls the peer served during the held DELETE and the sessions
// the phase opened.
func runCleanupHeldDeleteOverlap(t *testing.T, f *cleanupFixture, c cleanupScaleCase) (int, int) {
	t.Helper()

	const newcomer = "overlap-new"

	ctx, cancel := context.WithTimeout(t.Context(), cleanupBarrier)
	defer cancel()

	free := c.slots - cleanupSettle(t, f.registry).Resident
	for i := range free {
		require.NoError(t, cleanupScaleExecute(ctx, f.registry, fmt.Sprintf("overlap-fill-%02d", i)))
	}

	require.Equal(t, cleanupPoolState{Resident: c.slots}, cleanupSettle(t, f.registry), "the overlap needs a full, idle pool")
	f.peer.await(t, "earlier DELETE handlers finished", func(s cleanupPeerStats) bool { return s.DeleteInflight == 0 })

	var live string

	f.registry.mu.Lock()
	for partition := range f.registry.entries {
		live = partition

		break
	}
	f.registry.mu.Unlock()

	// The most recently used entry is never the LRU eviction victim.
	require.NoError(t, cleanupScaleExecute(ctx, f.registry, live))

	liveOpens := f.peer.openCount(live)

	f.peer.hold()

	admitted := cleanupRun(ctx, func(ctx context.Context) error { return cleanupScaleExecute(ctx, f.registry, newcomer) })

	f.peer.await(t, "the eviction DELETE held", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })
	cleanupEventually(t, "the newcomer waiting for the closing slot", func() bool {
		return cleanupPool(f.registry) == cleanupPoolState{Resident: c.slots - 1, Closing: 1, Waiting: 1}
	})

	held := f.peer.snapshot()
	require.NoError(t, cleanupScaleExecute(ctx, f.registry, live), "a resident partition needs no capacity")

	after := f.peer.snapshot()
	require.Equal(t, 1, after.DeleteInflight, "the eviction DELETE must still be held")
	require.Equal(t, held.DeleteDone, after.DeleteDone)
	require.Equal(t, liveOpens, f.peer.openCount(live), "the live partition must keep its session")
	require.Zero(t, f.peer.openCount(newcomer), "the newcomer must wait for the closing slot")

	state := cleanupPool(f.registry)
	require.Equal(t, c.slots-1, state.Resident)
	require.Equal(t, 1, state.Closing)

	f.peer.release()
	require.NoError(t, cleanupResult(t, "the newcomer", admitted))
	require.Equal(t, 1, f.peer.openCount(newcomer))
	cleanupSettle(t, f.registry)

	return after.CallsDuringDelete - held.CallsDuringDelete, free + 1
}

// holdCleanup blocks every protocol's session cleanup at the peer: legacy
// DELETEs, and the modern listen-stream notifications/cancelled.
func (f *cleanupFixture) holdCleanup() {
	f.peer.mu.Lock()
	f.peer.holdCancel = true
	f.peer.mu.Unlock()
	f.peer.hold()
}

// cleanupHeld counts cleanup requests held at the peer since before: legacy
// DELETEs, or modern listen-stream notifications/cancelled.
func cleanupHeld(now, before cleanupPeerStats, legacy bool) int {
	if legacy {
		return now.DeleteInflight - before.DeleteInflight
	}

	return now.CancelInflight - before.CancelInflight
}

// cleanupDeadline is a deadline the test fires itself: Err reports
// context.DeadlineExceeded once expire is called. It has no clock deadline, so
// a caller can be admitted to a wait before it fires on any schedule.
type cleanupDeadline struct {
	done chan struct{}
	once sync.Once
}

func newCleanupDeadline() *cleanupDeadline { return &cleanupDeadline{done: make(chan struct{})} }

func (*cleanupDeadline) Deadline() (time.Time, bool) { return time.Time{}, false }

func (d *cleanupDeadline) Done() <-chan struct{} { return d.done }

func (d *cleanupDeadline) Err() error {
	select {
	case <-d.done:
		return context.DeadlineExceeded
	default:
		return nil
	}
}

func (*cleanupDeadline) Value(any) any { return nil }

func (d *cleanupDeadline) expire() { d.once.Do(func() { close(d.done) }) }

// cleanupAwait bounds the wait for a caller that held cleanup could block. On
// timeout it releases the peer so the caller and the registry can finish, then
// fails.
func cleanupAwait(t *testing.T, f *cleanupFixture, what string, done <-chan error) error {
	t.Helper()

	timer := time.NewTimer(cleanupBarrier)
	defer timer.Stop()

	select {
	case err := <-done:
		return err
	case <-timer.C:
		f.peer.release()
		t.Fatalf("%s never returned while cleanup was held", what)

		return nil
	}
}

// runCleanupDeadlineCohort holds cleanup at the peer and first parks a primer
// caller behind its own eviction, so an idle session is detached and its
// cleanup is held at the peer before any short deadline starts. A deadline
// waiter is then admitted to the capacity wait, proven by the registry's
// waiting count, before its deadline can fire. Fresh partitions with 50ms
// deadlines then hit the full pool. They must each return their own deadline
// error, open no session and leave every held cleanup in flight, as must the
// admitted waiter once its deadline fires afterwards. That is proven by
// registry and peer state once callers returned, not by wall time: the
// overrun is reported only, since it includes scheduler delay. Watchdogs
// release the peer and fail if a caller never returns. With stall, the test
// holds the pool lock until every 50ms deadline has expired, the worst
// schedule, so no 50ms caller reaches the pool; the admitted waiter does not
// depend on that schedule.
func runCleanupDeadlineCohort(t *testing.T, f *cleanupFixture, c cleanupScaleCase, stall bool) (int, int, float64) {
	t.Helper()

	cohort := min(2*c.slots, 16)

	// Fill every slot with an idle session first, so the cohort can only get
	// capacity by evicting.
	for i := range c.slots {
		ctx, cancel := context.WithTimeout(t.Context(), cleanupBarrier)
		require.NoError(t, cleanupScaleExecute(ctx, f.registry, fmt.Sprintf("warm-%02d", i)))
		cancel()
	}

	require.Equal(t, cleanupPoolState{Resident: c.slots}, cleanupSettle(t, f.registry))

	before := f.peer.snapshot()
	f.holdCleanup()

	const primer = "late-primer"

	primed, stopPrimer := context.WithCancel(t.Context())
	defer stopPrimer()

	primerDone := cleanupRun(primed, func(ctx context.Context) error { return cleanupScaleExecute(ctx, f.registry, primer) })

	cleanupEventually(t, "the primer waiting behind its held eviction", func() bool {
		return cleanupPool(f.registry) == cleanupPoolState{Resident: c.slots - 1, Closing: 1, Waiting: 1} &&
			cleanupHeld(f.peer.snapshot(), before, c.legacy) == 1
	})

	// Only the primer and this waiter acquire here, so a waiting count of two
	// proves the waiter is in the capacity wait before its deadline fires. It
	// may evict one more idle session first, which is held as well.
	const waiter = "late-waiter"

	waiterDeadline := newCleanupDeadline()
	t.Cleanup(waiterDeadline.expire)

	waited := cleanupRun(waiterDeadline, func(ctx context.Context) error { return cleanupScaleExecute(ctx, f.registry, waiter) })

	var admitted cleanupPoolState

	cleanupEventually(t, "the deadline waiter admitted to the capacity wait", func() bool {
		admitted = cleanupPool(f.registry)

		return admitted.Waiting == 2 && admitted.Resident+admitted.Closing == c.slots &&
			cleanupHeld(f.peer.snapshot(), before, c.legacy) == admitted.Closing
	})
	require.NoError(t, waiterDeadline.Err(), "the waiter's deadline must not fire before admission")
	require.Zero(t, f.peer.openCount(waiter))

	const deadline = 50 * time.Millisecond

	type outcome struct {
		err     error
		overrun time.Duration
	}

	outcomes := make(chan outcome, cohort)
	expired := make([]<-chan struct{}, cohort)

	if stall {
		f.registry.mu.Lock()
	}

	for i := range cohort {
		ctx, cancel := context.WithTimeout(t.Context(), deadline)
		expired[i] = ctx.Done()

		go func() {
			defer cancel()

			start := time.Now()

			err := cleanupScaleExecute(ctx, f.registry, fmt.Sprintf("late-%02d", i))
			outcomes <- outcome{err: err, overrun: time.Since(start) - deadline}
		}()
	}

	if stall {
		for _, done := range expired {
			<-done
		}

		f.registry.mu.Unlock()
	}

	var (
		errs    int
		overrun time.Duration
	)

	watchdog := time.NewTimer(cleanupBarrier)
	defer watchdog.Stop()

	for received := range cohort {
		select {
		case got := <-outcomes:
			if assert.ErrorIs(t, got.err, context.DeadlineExceeded) {
				errs++
			}

			overrun = max(overrun, got.overrun)
		case <-watchdog.C:
			// A caller is blocked on held cleanup: release the peer so every
			// goroutine can finish before the test fails.
			f.peer.release()
			t.Fatalf("%d of %d deadline callers never returned while cleanup was held", cohort-received, cohort)
		}
	}

	for i := range cohort {
		assert.Zero(t, f.peer.openCount(fmt.Sprintf("late-%02d", i)), "slots held by closing sessions must not be reused")
	}

	// Closing counts sessions whose Close has not returned, so a positive count
	// after every caller returned shows no caller waited for its eviction. The
	// primer's held eviction makes it positive on any schedule.
	evicted := cleanupPool(f.registry).Closing
	assert.GreaterOrEqual(t, evicted, admitted.Closing, "the primer's and waiter's evictions must still be closing")
	assert.LessOrEqual(t, evicted, c.slots)
	assert.Equal(t, 2, cleanupPool(f.registry).Waiting, "the primer and the admitted deadline waiter must still wait")

	if stall {
		assert.Equal(t, admitted.Closing, evicted, "a stalled cohort must not reach the pool")
	}

	// Held cleanup cannot finish, so every evicted session's cleanup request
	// eventually reaches the peer and stays there.
	cleanupEventually(t, "evictions held at the peer", func() bool {
		return cleanupHeld(f.peer.snapshot(), before, c.legacy) == evicted
	})
	assert.Equal(t, evicted, cleanupPool(f.registry).Closing, "no held cleanup may complete")
	assert.Equal(t, before.DeleteDone, f.peer.snapshot().DeleteDone)

	// The admitted waiter's deadline fires only now, while it waits for
	// capacity behind held cleanup.
	waiterDeadline.expire()
	require.ErrorIs(t, cleanupAwait(t, f, "the admitted deadline waiter", waited), context.DeadlineExceeded)
	assert.Equal(t, cleanupPoolState{Resident: c.slots - evicted, Closing: evicted, Waiting: 1}, cleanupPool(f.registry),
		"the waiter must leave on its deadline without waiting for cleanup")
	assert.Equal(t, evicted, cleanupHeld(f.peer.snapshot(), before, c.legacy), "every eviction must still be held")
	assert.Equal(t, before.DeleteDone, f.peer.snapshot().DeleteDone)
	assert.Zero(t, f.peer.openCount(waiter), "the waiter must not open while its slot is closing")

	stopPrimer()
	require.ErrorIs(t, cleanupAwait(t, f, "the primer", primerDone), context.Canceled)
	assert.Zero(t, f.peer.openCount(primer), "the primer must not open while its slot is closing")
	assert.Equal(t, evicted, cleanupPool(f.registry).Closing, "the canceled primer must not wait for cleanup")

	f.peer.release()

	return cohort, errs, float64(overrun.Microseconds()) / 1000
}

// TestRegistryCleanupDeadlineCohortUnderPoolStall runs the deadline cohort on
// the worst schedule: the test holds the pool lock until every 50ms deadline
// has expired, so no 50ms caller evicts anything. The primer's held eviction
// and the deadline waiter admitted before the stall must still prove that
// deadlines return while cleanup is held at the peer.
func TestRegistryCleanupDeadlineCohortUnderPoolStall(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			c := cleanupScaleCase{legacy: legacy, slots: 4}
			f := newCleanupFixture(t, legacy, WithRegistryMaxConnections(c.slots))

			cohort, errs, _ := runCleanupDeadlineCohort(t, f, c, true)
			assert.Equal(t, 2*c.slots, cohort)
			assert.Equal(t, cohort, errs)
			cleanupSettle(t, f.registry)
		})
	}
}

func checkCleanupScale(t *testing.T, c cleanupScaleCase, r cleanupScaleResult) {
	t.Helper()

	assert.Zero(t, r.Errors)
	assert.Equal(t, r.Operations, r.Verified)
	assert.Equal(t, r.DeadlineCohort, r.DeadlineErrors)
	assert.LessOrEqual(t, r.OccupiedPeak, c.slots)
	assert.Equal(t, int64(min(c.workers, c.users)), r.PeakOperations, "all workers must have run concurrently")
	assert.True(t, r.ShutdownBlocked, "Shutdown must wait for held cleanup")
	assert.True(t, r.PoolSettledAfterAll)
	assert.Zero(t, r.IdentityMismatches)
	assert.Zero(t, r.SocketsAfterClose)
	assert.Zero(t, r.ServerConnsAfter)
	assert.GreaterOrEqual(t, r.Reconnects, 0, "every logical user opened its own session")

	if c.users <= c.slots {
		assert.Zero(t, r.Reconnects, "users that fit must keep their sessions")
	}

	if c.legacy {
		assert.LessOrEqual(t, r.LivePeak, c.slots, "live legacy sessions never exceed capacity")
		assert.LessOrEqual(t, r.DeletePeak, c.slots, "concurrent cleanups are bounded by capacity")
		assert.Equal(t, c.slots, r.LivePeak, "the run must saturate capacity")
		assert.Zero(t, r.LiveAfterClose)
		assert.True(t, r.DeletedOncePerOpen)
		assert.Zero(t, r.OrphanedSessions, "no connect is canceled, so the client knows every session")
		assert.Equal(t, r.Opens, r.Deletes, "each legacy session is terminated exactly once")
	} else {
		assert.Zero(t, r.Deletes, "modern sessions are never DELETEd")
		assert.GreaterOrEqual(t, r.ListenPeak, 1)
	}

	if c.users > c.slots {
		assert.GreaterOrEqual(t, r.ClosingPeak, 1, "evictions must have closed in the background")

		if c.legacy {
			assert.GreaterOrEqual(t, r.DeletePeak, 2, "eviction DELETEs must overlap")
			assert.Equal(t, 1, r.HeldDeleteOverlap, "a resident partition's call must complete while an eviction DELETE is held")
			assert.Positive(t, r.CallsDuringDelete, "operations must progress while cleanup is in flight")
		}
	}
}

// TestRegistryCleanupScale runs 1 to 10,000 distinct logical users through
// real go-sdk clients and loopback MCP peers for both protocols, with bounded
// workers and max_connections, controlled legacy DELETE pacing, a deadline
// cohort while cleanup is held, and a held-then-drained Shutdown. It does not
// open 10,000 simultaneous sessions: live sessions are bounded by capacity.
func TestRegistryCleanupScale(t *testing.T) { //nolint:paralleltest // Bounded heavy matrix; serial to keep peaks and goroutine counts attributable.
	scales := []int{1, 10, 100, 1000, 10000}
	if testing.Short() {
		scales = scales[:4]
	}

	for _, legacy := range []bool{true, false} { //nolint:paralleltest // Serial by design, see above.
		for _, users := range scales {
			c := cleanupScaleCase{legacy: legacy, users: users, opsPerUser: 2, workers: min(users, 16), slots: min(users, 8)}
			if users >= 10000 {
				c.opsPerUser = 1
			}

			t.Run(fmt.Sprintf("legacy=%t/users=%d", legacy, users), func(t *testing.T) {
				r := runCleanupScale(t, c)
				t.Logf("CLEANUP_SCALE %s", r.row(t))
				checkCleanupScale(t, c, r)
			})
		}
	}
}

// TestRegistryCleanupOnePartitionContention holds 32 concurrent calls on one
// partition's single session, shows an intruder cannot evict it, and then
// drives 2,000 operations through that one session with no reconnect.
func TestRegistryCleanupOnePartitionContention(t *testing.T) { //nolint:paralleltest // Serial with the scale matrix.
	for _, legacy := range []bool{true, false} { //nolint:paralleltest // Serial with the scale matrix.
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			const (
				workers    = 32
				operations = 2000
			)

			started := time.Now()
			f := newCleanupFixture(t, legacy, WithRegistryMaxConnections(1))
			f.whoami(t, "hot")
			f.peer.holdCalls()

			jobs := make(chan struct{})

			var (
				wg             sync.WaitGroup
				verified       atomic.Int32
				inflight, peak atomic.Int64
			)

			for range workers {
				wg.Go(func() {
					for range jobs {
						cleanupEnter(&inflight, &peak)

						ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
						if assert.NoError(t, cleanupScaleExecute(ctx, f.registry, "hot")) {
							verified.Add(1)
						}

						cancel()
						inflight.Add(-1)
					}
				})
			}

			feed := make(chan struct{})

			go func() {
				defer close(feed)

				for range operations {
					jobs <- struct{}{}
				}

				close(jobs)
			}()

			f.peer.await(t, "32 concurrent calls on one session", func(s cleanupPeerStats) bool { return s.CallInflight == workers })

			cold, stop := context.WithTimeout(t.Context(), 50*time.Millisecond)
			require.ErrorIs(t, cleanupScaleExecute(cold, f.registry, "cold"), context.DeadlineExceeded)
			stop()
			require.Zero(t, f.peer.openCount("cold"), "a leased session must not be evicted")

			f.peer.release()
			<-feed
			wg.Wait()
			require.NoError(t, f.registry.Close())
			f.client.transport.CloseIdleConnections()
			cleanupEventually(t, "client sockets closed", func() bool { return f.client.snapshot().Sockets == 0 })
			cleanupEventually(t, "server connections closed", func() bool { return f.peer.snapshot().Conns == 0 })

			stats, client := f.peer.snapshot(), f.client.snapshot()
			r := cleanupScaleResult{
				Legacy: legacy, Scenario: cleanupOnePartition,
				LogicalUsers: 1, Operations: operations, Workers: workers, MaxConnections: 1,
				PeakOperations: peak.Load(), Verified: int(verified.Load()), Opens: stats.Opens,
				Deletes: stats.DeleteStarted, Cancels: stats.Cancels, CallPeak: stats.CallPeak, LivePeak: stats.LivePeak,
				ListenPeak: stats.ListenPeak, Dials: client.Dials, SocketPeak: client.SocketPeak, ServerConnPeak: stats.ConnPeak,
				SocketsAfterClose: client.Sockets, ServerConnsAfter: stats.Conns, LiveAfterClose: stats.Live,
				IdentityMismatches: len(f.peer.identityMismatches()), PoolSettledAfterAll: cleanupPool(f.registry) == cleanupPoolState{},
				DeletedOncePerOpen: legacy && len(f.peer.deleteCounts()) == 1 && f.peer.deleteCounts()[f.peer.sessionsOf("hot")[0]] == 1,
				DurationMs:         float64(time.Since(started).Microseconds()) / 1000,
			}
			t.Logf("CLEANUP_SCALE %s", r.row(t))

			assert.Equal(t, operations, r.Verified)
			assert.Equal(t, int64(workers), r.PeakOperations)
			assert.Equal(t, workers, r.CallPeak)
			assert.Equal(t, 1, f.peer.openCount("hot"), "contention must share one session")
			assert.True(t, r.PoolSettledAfterAll)
			assert.Zero(t, r.IdentityMismatches)
			assert.Zero(t, r.SocketsAfterClose)
			assert.Zero(t, r.ServerConnsAfter)

			if legacy {
				assert.True(t, r.DeletedOncePerOpen, "only Close terminates the shared session, once")
				assert.Equal(t, 1, r.LivePeak)
				assert.Zero(t, r.LiveAfterClose)
			} else {
				assert.Zero(t, r.Deletes)
			}
		})
	}
}

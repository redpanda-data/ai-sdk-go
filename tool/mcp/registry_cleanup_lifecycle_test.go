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
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type cleanupPoolState struct {
	Resident int `json:"resident"`
	Closing  int `json:"closing"`
	Draining int `json:"draining"`
	Waiting  int `json:"waiting"`
}

func cleanupPool(r *Registry) cleanupPoolState {
	r.mu.Lock()
	defer r.mu.Unlock()

	return cleanupPoolState{Resident: len(r.entries), Closing: r.closing, Draining: r.draining, Waiting: r.waiting}
}

// cleanupSettle is a barrier on the registry's own wakeups until no detached
// close, draining lease or capacity waiter remains.
func cleanupSettle(t *testing.T, r *Registry) cleanupPoolState {
	t.Helper()

	timer := time.NewTimer(cleanupBarrier)
	defer timer.Stop()

	for {
		r.mu.Lock()
		state := cleanupPoolState{Resident: len(r.entries), Closing: r.closing, Draining: r.draining, Waiting: r.waiting}
		changed := r.changed
		r.mu.Unlock()

		require.GreaterOrEqual(t, state.Closing, state.Draining, "closing includes draining")
		require.GreaterOrEqual(t, state.Draining, 0)
		require.LessOrEqual(t, state.Resident+state.Closing, r.config.maxConnections)

		if state.Closing == 0 && state.Waiting == 0 {
			return state
		}

		select {
		case <-changed:
		case <-timer.C:
			t.Fatalf("registry never settled: %+v", state)
		}
	}
}

// cleanupEventually polls a convergence condition, such as sockets closing,
// that has no wakeup of its own.
func cleanupEventually(t *testing.T, what string, cond func() bool) {
	t.Helper()

	deadline := time.Now().Add(cleanupBarrier)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("never converged: %s", what)
		}

		time.Sleep(5 * time.Millisecond)
	}
}

// TestRegistryLeasedSessionIsNeverEvicted holds Alice's tool call at the
// peer. Bob cannot take the only slot from a leased session; Alice's call
// finishes on her original session, and only then is that session evicted.
func TestRegistryLeasedSessionIsNeverEvicted(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
	f.whoami(t, "alice")
	session := f.peer.sessionsOf("alice")[0]
	f.peer.holdCalls()

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "alice"), cleanupBarrier)
	defer cancel()

	inflight := cleanupRun(ctx, func(ctx context.Context) error {
		response, err := f.registry.Execute(ctx, cleanupRequest())
		if err == nil && !strings.Contains(string(response.Result), "caller=Bearer alice") {
			err = fmt.Errorf("wrong caller: %s", response.Result)
		}

		return err
	})

	f.peer.await(t, "Alice's held call", func(s cleanupPeerStats) bool { return s.CallInflight == 1 })

	bob, stop := context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
	defer stop()

	_, err := f.registry.List(bob)
	require.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Zero(t, f.peer.snapshot().DeleteStarted)
	assert.Zero(t, f.peer.openCount("bob"))

	f.peer.release()
	require.NoError(t, cleanupResult(t, "Alice's call", inflight))
	f.whoami(t, "bob")
	f.client.awaitDeletes(t, 1)
	assert.Equal(t, map[string]int{session: 1}, f.peer.deleteCounts())
	assert.Equal(t, 1, f.peer.openCount("alice"))
	cleanupSettle(t, f.registry)
}

// TestRegistryConcurrentRetireAndRelease races several leases retiring and
// releasing one live session against a capacity waiter for another partition,
// repeatedly. Each session must be closed exactly once, accounting must
// return to zero, and the waiter must be woken every time without a deadline.
func TestRegistryConcurrentRetireAndRelease(t *testing.T) {
	t.Parallel()

	const (
		rounds = 12
		leases = 4
	)

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))

	for round := range rounds {
		partition, waiter := fmt.Sprintf("retire-%d", round), fmt.Sprintf("waiter-%d", round)

		ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), partition), cleanupBarrier)
		ctx, end, err := f.registry.begin(ctx)
		require.NoError(t, err)

		entries := make([]*registryEntry, leases)
		releases := make([]func(), leases)

		for i := range leases {
			entries[i], releases[i], err = f.registry.acquire(ctx, partition)
			require.NoError(t, err)
			require.Same(t, entries[0], entries[i])
		}

		waited := cleanupRun(cleanupCaller(ctx, waiter), func(ctx context.Context) error {
			response, err := f.registry.Execute(ctx, cleanupRequest())
			if err == nil && !strings.Contains(string(response.Result), "caller=Bearer "+waiter+`"`) {
				err = fmt.Errorf("wrong caller: %s", response.Result)
			}

			return err
		})

		var wg sync.WaitGroup

		start := make(chan struct{})

		for i := range leases {
			wg.Go(func() {
				<-start

				if i%2 == 0 {
					f.registry.retireClosed(entries[i], sdkmcp.ErrConnectionClosed)
				}

				releases[i]()
			})
		}

		close(start)
		wg.Wait()
		require.NoError(t, cleanupResult(t, waiter, waited))
		end()
		cancel()

		state := cleanupSettle(t, f.registry)
		require.Equal(t, cleanupPoolState{Resident: 1}, state, "round %d", round)
	}

	f.client.awaitDeletes(t, 2*rounds-1)

	counts := f.peer.deleteCounts()
	require.Len(t, counts, 2*rounds-1, "every session but the last waiter's is closed")

	for session, n := range counts {
		require.Equal(t, 1, n, session)
	}

	stats := f.peer.snapshot()
	assert.Equal(t, 2*rounds, stats.Opens)
	assert.Equal(t, 1, stats.LivePeak, "one slot must never hold two live sessions")
	assert.Empty(t, f.peer.identityMismatches())
}

// TestRegistryShutdownRacesAcquisitions shuts down while workers churn through
// more partitions than slots. Shutdown must drain operations, connects and
// detached closes: afterwards every opened legacy session has exactly one
// DELETE, no open is admitted, and every worker stops on ErrClosed or on the
// registry's cancellation of its in-flight operation.
func TestRegistryShutdownRacesAcquisitions(t *testing.T) {
	t.Parallel()

	const (
		workers    = 12
		partitions = 24
		slots      = 4
	)

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(slots))
	started := make(chan struct{})

	var (
		wg        sync.WaitGroup
		mu        sync.Mutex
		successes int
		closed    int
		canceled  int
		startOnce sync.Once
	)

	for w := range workers {
		wg.Go(func() {
			for i := 0; ; i++ {
				partition := fmt.Sprintf("p%d", (w*7+i)%partitions)

				ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), partition), cleanupBarrier)
				response, err := f.registry.Execute(ctx, cleanupRequest())

				cancel()

				if err == nil && response.IsError {
					switch result := string(response.Result); {
					case strings.Contains(result, context.Canceled.Error()):
						err = context.Canceled
					case strings.Contains(result, ErrClosed.Error()):
						err = ErrClosed
					default:
						err = fmt.Errorf("tool error: %s", result)
					}
				}

				mu.Lock()
				switch {
				case err == nil:
					assert.Contains(t, string(response.Result), "caller=Bearer "+partition+`"`)

					successes++
					if successes == 2*workers {
						startOnce.Do(func() { close(started) })
					}
				case errors.Is(err, ErrClosed):
					closed++
				case errors.Is(err, context.Canceled):
					canceled++
				default:
					t.Errorf("%s: %v", partition, err)
				}
				mu.Unlock()

				if err != nil {
					return
				}
			}
		})
	}

	select {
	case <-started:
	case <-time.After(cleanupBarrier):
		t.Fatal("workers never made progress")
	}

	ctx, cancel := context.WithTimeout(t.Context(), cleanupBarrier)
	defer cancel()

	require.NoError(t, f.registry.Shutdown(ctx))

	final := f.peer.snapshot()

	wg.Wait()
	assert.Equal(t, workers, closed+canceled, "every worker must stop on shutdown")
	assert.Equal(t, final.Opens, f.peer.snapshot().Opens, "no open after Shutdown returned")
	assert.LessOrEqual(t, final.LivePeak, slots)
	assert.Equal(t, slots, final.LivePeak, "the churn must saturate capacity")

	// Shutdown cancels connects in flight. If that lands after the peer
	// assigned a session but before go-sdk read the initialize response, the
	// client never learns the id and no client can DELETE it; such a session
	// has no request from the client at all. Every other session must be
	// terminated exactly once before Shutdown returns.
	orphans := f.peer.orphans()
	assert.Len(t, orphans, final.Live, "Shutdown returned before every known session was terminated")
	assert.Equal(t, final.Sessions-len(orphans), final.DeleteStarted)

	for session, n := range f.peer.deleteCounts() {
		assert.Equal(t, 1, n, session)
	}

	if len(orphans) > 0 {
		assert.Positive(t, canceled, "only a canceled connect can orphan a session")
	}

	assert.Equal(t, cleanupPoolState{}, cleanupPool(f.registry))
	t.Logf("successes=%d closed=%d canceled=%d orphaned_by_canceled_initialize=%d peer=%+v", successes, closed, canceled, len(orphans), final)
}

// TestRegistryCleanupLeavesNoGoroutinesOrSockets runs blocked and released
// eviction rounds, then closes the registry. Every goroutine and socket the
// rounds created must be gone. It runs serially so the goroutine count is
// not shared with parallel tests.
func TestRegistryCleanupLeavesNoGoroutinesOrSockets(t *testing.T) { //nolint:paralleltest // Counts process-wide goroutines.
	f := newCleanupFixture(t, true, WithRegistryMaxConnections(2))
	baseline := runtime.NumGoroutine()

	for round := range 3 {
		for i := range 2 {
			f.list(t, fmt.Sprintf("resident-%d-%d", round, i))
		}

		f.peer.hold()

		for i := range 2 {
			short, stop := context.WithTimeout(cleanupCaller(t.Context(), fmt.Sprintf("late-%d-%d", round, i)), 50*time.Millisecond)
			_, err := f.registry.List(short)

			stop()
			require.ErrorIs(t, err, context.DeadlineExceeded)
		}

		f.peer.await(t, "eviction blocked", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })
		f.peer.release()
		cleanupSettle(t, f.registry)
	}

	require.NoError(t, f.registry.Close())
	f.client.transport.CloseIdleConnections()

	stats := f.peer.snapshot()
	assert.Zero(t, stats.Live)
	assert.Equal(t, stats.Sessions, stats.DeleteDone)
	cleanupEventually(t, "client sockets closed", func() bool { return f.client.snapshot().Sockets == 0 })
	cleanupEventually(t, "server connections closed", func() bool { return f.peer.snapshot().Conns == 0 })
	cleanupEventually(t, fmt.Sprintf("goroutines back to %d", baseline), func() bool { return runtime.NumGoroutine() <= baseline })
	t.Logf("baseline goroutines=%d final=%d client=%+v peer=%+v", baseline, runtime.NumGoroutine(), f.client.snapshot(), f.peer.snapshot())
}

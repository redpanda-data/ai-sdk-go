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
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

type retirementPoolState struct {
	Entries  int `json:"resident_entries"`
	Closing  int `json:"closing"`
	Draining int `json:"draining"`
	Waiting  int `json:"waiting"`
}

func retirementState(r *Registry) retirementPoolState {
	r.mu.Lock()
	defer r.mu.Unlock()

	return retirementPoolState{Entries: len(r.entries), Closing: r.closing, Draining: r.draining, Waiting: r.waiting}
}

// retirementLease acquires inside an operation window, as Snapshot and tool
// execution do. The returned release is idempotent.
func retirementLease(ctx context.Context, tb testing.TB, r *Registry, owner string) (*registryEntry, func()) {
	tb.Helper()

	entry, release, err := retirementTryLease(retirementCaller(ctx, owner, "one"), r, owner)
	require.NoError(tb, err)

	return entry, release
}

func retirementTryLease(ctx context.Context, r *Registry, owner string) (*registryEntry, func(), error) {
	opCtx, end, err := r.begin(ctx)
	if err != nil {
		return nil, nil, err
	}

	entry, release, err := r.acquire(opCtx, owner)
	if err != nil {
		end()

		return nil, nil, err
	}

	var once sync.Once

	return entry, func() { once.Do(func() { release(); end() }) }, nil
}

func retirementSettle(ctx context.Context, tb testing.TB, r *Registry) {
	tb.Helper()

	retirementAwait(ctx, tb, "detached sessions to close", func() bool {
		state := retirementState(r)

		return state.Closing == 0 && state.Draining == 0 && state.Waiting == 0
	})
}

// retirementPing proves a leased session is still open. It avoids tools/list,
// which the registry only issues under the entry's catalog gate.
func retirementPing(ctx context.Context, entry *registryEntry, owner string) error {
	return entry.session.Ping(retirementCaller(ctx, owner, "one"), nil)
}

func TestRegistryRetirementClassification(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name   string
		err    error
		retire bool
	}{
		{name: "connection closed", err: sdkmcp.ErrConnectionClosed, retire: true},
		{name: "session missing", err: sdkmcp.ErrSessionMissing, retire: true},
		{name: "wrapped connection closed", err: fmt.Errorf("mcp: call tool: %w", fmt.Errorf("%w: calling %q", sdkmcp.ErrConnectionClosed, registryAgentCallMethod)), retire: true},
		{name: "wrapped session missing", err: fmt.Errorf("sending %q: %w", registryAgentCallMethod, sdkmcp.ErrSessionMissing), retire: true},
		{name: "joined session missing", err: errors.Join(io.ErrUnexpectedEOF, sdkmcp.ErrSessionMissing), retire: true},
		{name: "nil"},
		{name: "same text without identity", err: errors.New(sdkmcp.ErrConnectionClosed.Error())},
		{name: "formatted without wrapping", err: fmt.Errorf("lost: %v", sdkmcp.ErrSessionMissing)}, //nolint:errorlint // Deliberately drops the error identity.
		{name: "canceled", err: context.Canceled},
		{name: "deadline", err: fmt.Errorf("mcp: call tool: %w", context.DeadlineExceeded)},
		{name: "tool result error", err: &tool.ResultError{Result: json.RawMessage(`{"isError":true}`)}},
		{name: "transient HTTP status", err: errors.New(`sending "tools/call": rejected: Service Unavailable`)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, true, false)
			registry := newRetirementRegistry(t, peer)

			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()

			old, releaseOld := retirementLease(ctx, t, registry, "alice")
			oldID := old.session.ID()
			registry.retireClosed(old, tc.err)

			want := retirementPoolState{Entries: 1}
			if tc.retire {
				want = retirementPoolState{Closing: 1, Draining: 1}
			}

			require.Equal(t, want, retirementState(registry))

			next, releaseNext := retirementLease(ctx, t, registry, "alice")
			if tc.retire {
				require.NotSame(t, old, next, "a retired entry must never be returned to a new acquisition")
				assert.Equal(t, 2, peer.handshakesOf("alice"))
				require.NoError(t, retirementPing(ctx, old, "alice"), "the old lease still owns a usable session")
				assert.Zero(t, peer.deletesOf(oldID), "a leased session is not closed")
			} else {
				require.Same(t, old, next)
				assert.Equal(t, 1, peer.handshakesOf("alice"))
			}

			require.NoError(t, retirementPing(ctx, next, "alice"))

			releaseNext()
			releaseOld()
			retirementSettle(ctx, t, registry)

			current, leases := retirementEntry(registry, "alice")
			require.Same(t, next, current)
			assert.Zero(t, leases)
			assert.Zero(t, old.leases)

			if tc.retire {
				assert.Equal(t, 1, peer.deletesOf(oldID), "the retired session closes exactly once after its last lease")
				assert.Zero(t, peer.deletesOf(next.session.ID()))
			} else {
				assert.Zero(t, peer.deletesOf(oldID))
			}
		})
	}
}

func TestRegistryRetirementDrainsAfterLastLease(t *testing.T) {
	t.Parallel()

	peer := newRetirementPeer(t, true, false)
	registry := newRetirementRegistry(t, peer)

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()

	old, releaseFirst := retirementLease(ctx, t, registry, "alice")
	second, releaseSecond := retirementLease(ctx, t, registry, "alice")
	require.Same(t, old, second)

	registry.retireClosed(old, sdkmcp.ErrConnectionClosed)
	registry.retireClosed(old, fmt.Errorf("again: %w", sdkmcp.ErrSessionMissing))
	require.Equal(t, retirementPoolState{Closing: 1, Draining: 1}, retirementState(registry), "both holders' reports retire once")

	replacement, releaseReplacement := retirementLease(ctx, t, registry, "alice")
	require.NotSame(t, old, replacement)

	registry.retireClosed(old, sdkmcp.ErrConnectionClosed)
	require.Equal(t, retirementPoolState{Entries: 1, Closing: 1, Draining: 1}, retirementState(registry), "a late report about the old entry leaves the replacement alone")

	releaseFirst()
	releaseFirst()
	require.Equal(t, retirementPoolState{Entries: 1, Closing: 1, Draining: 1}, retirementState(registry))
	require.NoError(t, retirementPing(ctx, old, "alice"), "the remaining holder unwinds on an open session")
	assert.Zero(t, peer.deletesOf(old.session.ID()))

	releaseSecond()
	retirementSettle(ctx, t, registry)
	assert.Equal(t, 1, peer.deletesOf(old.session.ID()))
	assert.Zero(t, old.leases)

	current, leases := retirementEntry(registry, "alice")
	require.Same(t, replacement, current)
	assert.Equal(t, 1, leases)
	require.NoError(t, retirementPing(ctx, replacement, "alice"))

	releaseReplacement()
	assert.Equal(t, retirementPoolState{Entries: 1}, retirementState(registry))
	assert.Zero(t, peer.deletesOf(replacement.session.ID()))
	assert.Equal(t, 2, peer.handshakesOf("alice"))
}

// TestRegistryRetirementConcurrentRetireAndRelease races retirement reports, releases
// and new acquisitions on one partition, round after round. Every holder uses
// its session while leased, so a session closed under a lease would fail.
func TestRegistryRetirementConcurrentRetireAndRelease(t *testing.T) {
	t.Parallel()

	const rounds, holders, acquirers = 30, 16, 8

	peer := newRetirementPeer(t, true, false)
	registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

	ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
	defer cancel()

	created := make(map[*registryEntry]bool)

	for round := range rounds {
		entries := make([]*registryEntry, holders)
		releases := make([]func(), holders)

		for i := range holders {
			entries[i], releases[i] = retirementLease(ctx, t, registry, "alice")
		}

		retired := entries[0]
		created[retired] = true

		for _, entry := range entries {
			require.Same(t, retired, entry)
		}

		start := make(chan struct{})
		errs := make(chan error, holders+acquirers)
		obtained := make(chan *registryEntry, acquirers)

		var wg sync.WaitGroup

		for i := range holders {
			wg.Go(func() {
				<-start

				if i%2 == 0 {
					registry.retireClosed(retired, fmt.Errorf("round %d: %w", round, sdkmcp.ErrConnectionClosed))
				}

				errs <- retirementPing(ctx, retired, "alice")

				releases[i]()
			})
		}

		for range acquirers {
			wg.Go(func() {
				<-start

				entry, release, err := retirementTryLease(retirementCaller(ctx, "alice", "one"), registry, "alice")
				if err != nil {
					errs <- err

					return
				}

				errs <- retirementPing(ctx, entry, "alice")

				obtained <- entry

				release()
			})
		}

		close(start)
		wg.Wait()
		close(errs)
		close(obtained)

		for err := range errs {
			require.NoError(t, err, "round %d", round)
		}

		replacements := make(map[*registryEntry]bool)

		for entry := range obtained {
			if entry != retired {
				replacements[entry] = true
				created[entry] = true
			}
		}

		require.LessOrEqual(t, len(replacements), 1, "round %d: one retirement yields at most one replacement", round)

		retirementSettle(ctx, t, registry)
		assert.True(t, retired.retired)
		assert.Zero(t, retired.leases)
		assert.Equal(t, 1, peer.deletesOf(retired.session.ID()), "round %d: the retired session closes exactly once", round)

		registry.mu.Lock()
		for _, entry := range registry.entries {
			assert.False(t, entry.retired, "round %d: a retired entry is resident", round)
			assert.Zero(t, entry.leases)
		}
		registry.mu.Unlock()
	}

	assert.Equal(t, len(created), peer.handshakesOf("alice"), "every handshake produced exactly one observed entry")
	assert.LessOrEqual(t, peer.stats().PeakSessions, 2)

	peer.mu.Lock()
	for id, deletes := range peer.deletes {
		assert.LessOrEqual(t, deletes, 1, id)
	}
	peer.mu.Unlock()
}

func TestRegistryRetirementCapacityWaiters(t *testing.T) {
	t.Parallel()

	t.Run("one slot", func(t *testing.T) {
		t.Parallel()

		peer := newRetirementPeer(t, true, false)
		registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(1))

		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
		defer cancel()

		old, releaseOld := retirementLease(ctx, t, registry, "alice")
		registry.retireClosed(old, sdkmcp.ErrSessionMissing)

		for _, owner := range []string{"alice", "bob"} {
			short, stop := context.WithTimeout(retirementCaller(ctx, owner, "one"), 50*time.Millisecond)
			_, _, err := retirementTryLease(short, registry, owner)

			stop()
			require.ErrorIs(t, err, context.DeadlineExceeded, owner)
		}

		require.Equal(t, retirementPoolState{Closing: 1, Draining: 1}, retirementState(registry), "expired waiters leave no trace")

		canceled, cancelWaiter := context.WithCancel(retirementCaller(ctx, "bob", "one"))
		abandoned := make(chan error, 1)

		go func() {
			_, _, err := retirementTryLease(canceled, registry, "bob")
			abandoned <- err
		}()

		retirementAwait(ctx, t, "the canceled waiter to wait", func() bool { return retirementState(registry).Waiting == 1 })
		cancelWaiter()
		require.ErrorIs(t, retirementReceive(ctx, t, abandoned, "the canceled waiter"), context.Canceled)
		require.Equal(t, retirementPoolState{Closing: 1, Draining: 1}, retirementState(registry))

		type leased struct {
			owner   string
			entry   *registryEntry
			release func()
			err     error
		}

		results := make(chan leased, 2)

		for _, owner := range []string{"alice", "bob"} {
			go func() {
				entry, release, err := retirementTryLease(retirementCaller(ctx, owner, "one"), registry, owner)
				results <- leased{owner: owner, entry: entry, release: release, err: err}
			}()
		}

		retirementAwait(ctx, t, "both waiters to wait", func() bool { return retirementState(registry).Waiting == 2 })
		assert.Equal(t, 1, peer.handshakesOf("alice"))
		assert.Zero(t, peer.handshakesOf("bob"))
		assert.Zero(t, peer.deletesOf(old.session.ID()))

		releaseOld()

		first := retirementReceive(ctx, t, results, "the first waiter")
		require.NoError(t, first.err)
		retirementAwait(ctx, t, "the second waiter to keep waiting", func() bool { return retirementState(registry).Waiting == 1 })
		assert.Equal(t, 1, peer.deletesOf(old.session.ID()))

		first.release()

		second := retirementReceive(ctx, t, results, "the second waiter")
		require.NoError(t, second.err)
		assert.NotEqual(t, first.owner, second.owner)
		assert.Equal(t, 1, peer.deletesOf(first.entry.session.ID()), "the second waiter evicted the idle first")

		second.release()
		retirementSettle(ctx, t, registry)

		assert.Equal(t, retirementPoolState{Entries: 1}, retirementState(registry))
		assert.Equal(t, 2, peer.handshakesOf("alice"))
		assert.Equal(t, 1, peer.handshakesOf("bob"))
		assert.Equal(t, 1, peer.stats().PeakSessions)
	})

	t.Run("draining slot does not block eviction of an idle one", func(t *testing.T) {
		t.Parallel()

		peer := newRetirementPeer(t, true, false)
		registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

		ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
		defer cancel()

		carol, releaseCarol := retirementLease(ctx, t, registry, "carol")
		releaseCarol()

		old, releaseOld := retirementLease(ctx, t, registry, "alice")
		registry.retireClosed(old, sdkmcp.ErrConnectionClosed)

		bob, releaseBob := retirementLease(ctx, t, registry, "bob")
		retirementAwait(ctx, t, "carol's eviction to finish", func() bool { return retirementState(registry).Closing == 1 })

		assert.Equal(t, 1, peer.deletesOf(carol.session.ID()))
		assert.Zero(t, peer.deletesOf(old.session.ID()), "the draining session stays open for its holder")
		assert.Equal(t, retirementPoolState{Entries: 1, Closing: 1, Draining: 1}, retirementState(registry))
		require.NoError(t, retirementPing(ctx, bob, "bob"))
		require.NoError(t, retirementPing(ctx, old, "alice"))

		releaseBob()
		releaseOld()
		retirementSettle(ctx, t, registry)
		assert.Equal(t, 1, peer.deletesOf(old.session.ID()))
		assert.Equal(t, 2, peer.stats().PeakSessions)
	})
}

// TestRegistryRetirementEvictionDoesNotCascade holds eviction DELETEs at the peer. A
// waiting caller woken by unrelated releases must keep waiting for the close
// it already started instead of evicting another idle partition.
func TestRegistryRetirementEvictionDoesNotCascade(t *testing.T) {
	t.Parallel()

	peer := newRetirementPeer(t, true, false)
	registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()

	alice, releaseAlice := retirementLease(ctx, t, registry, "alice")
	releaseAlice()

	bob, releaseBob := retirementLease(ctx, t, registry, "bob")
	releaseBob()

	var heldDeletes atomic.Int64

	unblock := make(chan struct{})

	peer.setFault(func(req *http.Request, _ retirementWire) int {
		if req.Method == http.MethodDelete {
			heldDeletes.Add(1)

			select {
			case <-unblock:
			case <-req.Context().Done():
			}
		}

		return 0
	})

	type leased struct {
		entry   *registryEntry
		release func()
		err     error
	}

	waiter := func(owner string) <-chan leased {
		result := make(chan leased, 1)

		go func() {
			entry, release, err := retirementTryLease(retirementCaller(ctx, owner, "one"), registry, owner)
			result <- leased{entry: entry, release: release, err: err}
		}()

		return result
	}

	carol := waiter("carol")

	retirementAwait(ctx, t, "carol to evict alice and wait", func() bool {
		return heldDeletes.Load() == 1 && retirementState(registry).Waiting == 1
	})

	for range 20 {
		_, release := retirementLease(ctx, t, registry, "bob")
		release()
	}

	current, _ := retirementEntry(registry, "bob")
	require.Same(t, bob, current, "spurious wakeups must not evict another idle partition")
	assert.EqualValues(t, 1, heldDeletes.Load())
	assert.Equal(t, retirementPoolState{Entries: 1, Closing: 1, Waiting: 1}, retirementState(registry))

	dave := waiter("dave")

	retirementAwait(ctx, t, "dave to evict bob", func() bool {
		return heldDeletes.Load() == 2 && retirementState(registry).Waiting == 2
	})

	close(unblock)

	for _, result := range []<-chan leased{carol, dave} {
		got := retirementReceive(ctx, t, result, "a waiter")
		require.NoError(t, got.err)
		got.release()
	}

	retirementSettle(ctx, t, registry)
	assert.Equal(t, 1, peer.deletesOf(alice.session.ID()))
	assert.Equal(t, 1, peer.deletesOf(bob.session.ID()))
	assert.Equal(t, 1, peer.handshakesOf("carol"))
	assert.Equal(t, 1, peer.handshakesOf("dave"))
	assert.Equal(t, 2, peer.stats().PeakSessions, "closing sessions count toward the bound")
}

// TestRegistryRetirementShutdown shuts down while one
// partition is retired and leased, another is live and leased, and a third
// caller waits for capacity.
func TestRegistryRetirementShutdown(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)
			registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()

			calls := make(map[string]tool.Tool)

			for _, owner := range []string{"alice", "bob"} {
				caller := retirementCaller(ctx, owner, "one")
				captured, err := registry.Snapshot(caller)
				require.NoError(t, err)

				calls[owner] = retirementTool(caller, t, captured, owner)
			}

			bob, _ := retirementEntry(registry, "bob")
			_, aliceHeld := retirementStartHeld(ctx, t, calls["alice"], "alice", "alice-held")
			_, bobHeld := retirementStartHeld(ctx, t, calls["bob"], "bob", "bob-held")

			retirementLoseSession(retirementCaller(ctx, "alice", "one"), t, peer, calls["alice"], "alice", "lost")
			require.Equal(t, retirementPoolState{Entries: 1, Closing: 1, Draining: 1}, retirementState(registry))

			carol := make(chan error, 1)

			go func() {
				_, err := registry.Snapshot(retirementCaller(ctx, "carol", "one"))
				carol <- err
			}()

			retirementAwait(ctx, t, "carol to wait for capacity", func() bool { return retirementState(registry).Waiting == 1 })

			shutdown, stop := context.WithTimeout(ctx, 10*time.Second)
			defer stop()

			require.NoError(t, registry.Shutdown(shutdown))

			require.ErrorIs(t, retirementReceive(ctx, t, aliceHeld, "alice's held call"), context.Canceled)
			require.ErrorIs(t, retirementReceive(ctx, t, bobHeld, "bob's held call"), context.Canceled)

			err := retirementReceive(ctx, t, carol, "carol's waiting snapshot")
			require.True(t, errors.Is(err, ErrClosed) || errors.Is(err, context.Canceled), "carol: %v", err)

			assert.Equal(t, retirementPoolState{}, retirementState(registry))
			assert.Zero(t, peer.handshakesOf("carol"))
			assert.Zero(t, peer.stats().Duplicates)

			if legacy {
				stats := peer.stats()
				assert.Zero(t, stats.LiveSessions)
				assert.Equal(t, 1, peer.deletesOf(bob.session.ID()), "shutdown closes the live session once")
				assert.Equal(t, 1, stats.Deletes, "the lost session needs no DELETE")
			} else {
				retirementAwait(ctx, t, "listen streams to end", func() bool { return peer.stats().Listens == 0 })
			}

			_, err = registry.Snapshot(retirementCaller(ctx, "alice", "one"))
			require.ErrorIs(t, err, ErrClosed)
			require.NoError(t, registry.Close())
			require.NoError(t, registry.Shutdown(shutdown))
		})
	}
}

// TestRegistryRetirementWaiterBurst queues 10,000 live callers behind one
// slot that a retired, still-leased session holds. No caller connects while it
// drains; canceled callers unwind; every same-partition caller shares the one
// replacement; and Shutdown unwinds a second queued burst before closing.
func TestRegistryRetirementWaiterBurst(t *testing.T) {
	t.Parallel()

	total := 10000
	if retirementRaceEnabled {
		total = 6000
	}

	distinct := total * 2 / 5
	joiners := total - distinct

	peer := newRetirementPeer(t, true, false)
	registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(1))

	ctx, cancel := context.WithTimeout(t.Context(), 2*time.Minute)
	defer cancel()

	old, releaseOld := retirementLease(ctx, t, registry, "alice")
	registry.retireClosed(old, sdkmcp.ErrSessionMissing)

	type outcome struct {
		entry   *registryEntry
		release func()
		err     error
	}

	lease := func(caller context.Context, owner string, results chan<- outcome) {
		entry, release, err := retirementTryLease(caller, registry, owner)
		results <- outcome{entry: entry, release: release, err: err}
	}

	abandon, stopAbandoned := context.WithCancel(ctx)
	abandoned := make(chan outcome, distinct)
	joined := make(chan outcome, joiners)

	for i := range distinct {
		owner := fmt.Sprintf("burst-%d", i)
		go lease(retirementCaller(abandon, owner, "one"), owner, abandoned)
	}

	for range joiners {
		go lease(retirementCaller(ctx, "alice", "two"), "alice", joined)
	}

	retirementAwait(ctx, t, "every caller to wait for the draining slot", func() bool { return retirementState(registry).Waiting == total })
	require.Equal(t, retirementPoolState{Closing: 1, Draining: 1, Waiting: total}, retirementState(registry))
	assert.Equal(t, 1, peer.stats().Handshakes, "no caller connects while the retired session holds the slot")

	stopAbandoned()

	for range distinct {
		require.ErrorIs(t, retirementReceive(ctx, t, abandoned, "a canceled caller").err, context.Canceled)
	}

	require.Equal(t, retirementPoolState{Closing: 1, Draining: 1, Waiting: joiners}, retirementState(registry), "canceled callers leave the queue")

	releaseOld()

	var replacement *registryEntry

	releases := make([]func(), 0, joiners)

	for range joiners {
		got := retirementReceive(ctx, t, joined, "a queued same-partition caller")
		require.NoError(t, got.err)

		if replacement == nil {
			replacement = got.entry
		}

		require.Same(t, replacement, got.entry)

		releases = append(releases, got.release)
	}

	_, leases := retirementEntry(registry, "alice")
	assert.Equal(t, joiners, leases)
	assert.Equal(t, 2, peer.stats().Handshakes, "every queued caller shares one replacement")
	assert.Equal(t, 1, peer.deletesOf(old.session.ID()))

	late := make(chan outcome, distinct)

	for i := range distinct {
		owner := fmt.Sprintf("late-%d", i)
		go lease(retirementCaller(ctx, owner, "one"), owner, late)
	}

	retirementAwait(ctx, t, "the late burst to wait behind the leased replacement", func() bool { return retirementState(registry).Waiting == distinct })

	shutdown := make(chan error, 1)

	go func() { shutdown <- registry.Shutdown(ctx) }()

	for range distinct {
		err := retirementReceive(ctx, t, late, "a caller unwound by Shutdown").err
		require.True(t, errors.Is(err, ErrClosed) || errors.Is(err, context.Canceled), "late caller: %v", err)
	}

	select {
	case err := <-shutdown:
		t.Fatalf("Shutdown returned while %d leases were held: %v", joiners, err)
	default:
	}

	for _, release := range releases {
		release()
	}

	require.NoError(t, retirementReceive(ctx, t, shutdown, "Shutdown"))
	assert.Equal(t, retirementPoolState{}, retirementState(registry))

	stats := peer.stats()
	assert.Equal(t, 2, stats.Sessions)
	assert.Equal(t, 2, stats.Handshakes)
	assert.Zero(t, stats.LiveSessions)
	assert.Equal(t, 1, peer.deletesOf(replacement.session.ID()))
}

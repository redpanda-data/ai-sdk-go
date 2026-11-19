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
	"maps"
	"net/http"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

// retirementScaleReport is one machine-readable row of the scale evidence.
// Logical users and concurrency are reported separately: users are served
// through a bounded worker pool, never as simultaneous network sessions.
type retirementScaleReport struct {
	Scenario         string              `json:"scenario"`
	Mode             string              `json:"mode"`
	LogicalUsers     int                 `json:"logical_users"`
	Workers          int                 `json:"workers"`
	MaxConnections   int                 `json:"max_connections"`
	OverlapBarrier   int                 `json:"overlap_barrier"`
	WorkerPeak       int64               `json:"worker_peak_concurrency"`
	LogicalOps       int                 `json:"logical_operations"`
	Attempts         int64               `json:"attempts"`
	Succeeded        int64               `json:"succeeded_attempts"`
	Rejected         map[string]int64    `json:"rejected_attempts_by_class"`
	OtherSamples     []string            `json:"other_error_samples,omitempty"`
	FaultedOps       int                 `json:"faulted_operations"`
	Reconnects       int                 `json:"reconnects"`
	ClientPeak       int                 `json:"client_peak_resident_sampled"`
	Final            retirementPoolState `json:"final_pool"`
	Server           retirementPeerStats `json:"server"`
	ServerAfterClose retirementPeerStats `json:"server_after_close"`
	GoroutinesBefore int                 `json:"goroutines_before"`
	GoroutinesAfter  int                 `json:"goroutines_after_close"`
	HeapInuseMiB     uint64              `json:"heap_inuse_mib_after"`
	DurationMS       int64               `json:"duration_ms"`
}

func retirementWriteReport(t *testing.T, report retirementScaleReport) {
	t.Helper()

	line, err := json.Marshal(report)
	require.NoError(t, err)
	t.Logf("scale report: %s", line)

	path := os.Getenv("RETIREMENT_SCALE_REPORT")
	if path == "" {
		return
	}

	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600) //nolint:gosec // The operator chooses the evidence file.
	require.NoError(t, err)

	defer file.Close()

	_, err = file.Write(append(line, '\n'))
	require.NoError(t, err)
}

func retirementClass(err error) string {
	switch {
	case errors.Is(err, sdkmcp.ErrSessionMissing):
		return "session_missing"
	case errors.Is(err, sdkmcp.ErrConnectionClosed):
		return "connection_closed"
	case strings.Contains(err.Error(), http.StatusText(http.StatusBadRequest)):
		// The injected 400 itself, or another call the dead stream failed.
		return "bad_request"
	case errors.Is(err, context.Canceled), errors.Is(err, context.DeadlineExceeded):
		return "context"
	default:
		return "other"
	}
}

type retirementScaleCounters struct {
	attempts  atomic.Int64
	succeeded atomic.Int64
	inFlight  atomic.Int64
	peak      atomic.Int64

	mu       sync.Mutex
	rejected map[string]int64
	others   []string
	samples  int
	failures []string
}

func newRetirementScaleCounters() *retirementScaleCounters {
	return &retirementScaleCounters{rejected: make(map[string]int64)}
}

func (c *retirementScaleCounters) enter() func() {
	current := c.inFlight.Add(1)
	for {
		peak := c.peak.Load()
		if current <= peak || c.peak.CompareAndSwap(peak, current) {
			break
		}
	}

	return func() { c.inFlight.Add(-1) }
}

func (c *retirementScaleCounters) record(err error) {
	c.attempts.Add(1)

	if err == nil {
		c.succeeded.Add(1)

		return
	}

	class := retirementClass(err)

	c.mu.Lock()

	c.rejected[class]++
	if class == "other" && len(c.others) < 5 {
		c.others = append(c.others, err.Error())
	}
	c.mu.Unlock()
}

func (c *retirementScaleCounters) failf(format string, args ...any) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if len(c.failures) < 20 {
		c.failures = append(c.failures, fmt.Sprintf(format, args...))
	}
}

func (c *retirementScaleCounters) sample(r *Registry) {
	state := retirementState(r)

	c.mu.Lock()
	c.samples = max(c.samples, state.Entries+state.Closing)
	c.mu.Unlock()
}

func newRetirementScaleRegistry(t *testing.T, peer *retirementPeer, connections, workers int) (*Registry, *http.Transport) {
	t.Helper()

	transport := &http.Transport{MaxIdleConns: 4 * workers, MaxIdleConnsPerHost: 4 * workers, IdleConnTimeout: 30 * time.Second}
	t.Cleanup(transport.CloseIdleConnections)

	registry, err := NewRegistry("retire", peer.url,
		WithCredentials(retirementCredentials),
		WithRegistryMaxConnections(connections),
		WithRegistryHTTPClient(&http.Client{Transport: transport}),
	)
	require.NoError(t, err)

	return registry, transport
}

// retirementCatalogMatches verifies the identity-specific catalog: exactly the
// caller's tool, with a schema that names and pins the caller.
func retirementCatalogMatches(ctx context.Context, captured tool.Registry, owner string) error {
	definitions, err := captured.List(ctx)
	if err != nil {
		return err
	}

	if len(definitions) != 1 || definitions[0].Name != "retire__"+retirementToolName(owner) || definitions[0].Description != "identity of "+owner {
		return fmt.Errorf("%s received catalog %v", owner, registryNames(definitions))
	}

	var schema struct {
		Properties map[string]struct {
			Const string `json:"const"`
		} `json:"properties"`
	}

	if err := json.Unmarshal(definitions[0].Parameters, &schema); err != nil {
		return err
	}

	if len(schema.Properties) != 2 || schema.Properties["owner_"+owner].Const != owner {
		return fmt.Errorf("%s received schema %s", owner, definitions[0].Parameters)
	}

	return nil
}

func retirementScaleFault(peer *retirementPeer) {
	peer.setFault(func(_ *http.Request, wire retirementWire) int {
		if wire.rpcMethod != registryAgentCallMethod || !strings.HasSuffix(wire.op, "/fault#0") {
			return 0
		}

		if peer.legacy {
			// The server loses the session; the SDK handler then answers 404.
			peer.terminateSession(wire.owner, wire.session)

			return 0
		}

		return http.StatusBadRequest
	})
}

// TestRegistryRetirementScale serves 1 to 10,000 logical users, each with its
// own credentials, identity-specific schema and results, through a bounded
// worker pool and a small connection pool. Every fifth user loses its session
// during a tool call; the application retries with a new operation ID.
//
//nolint:tparallel,paralleltest // Scales run one at a time to bound sockets and memory.
func TestRegistryRetirementScale(t *testing.T) {
	t.Parallel()

	scales := []int{1, 10, 100, 1000, 10000}
	if testing.Short() {
		scales = []int{1, 10, 100}
	}

	for _, legacy := range []bool{true, false} {
		for _, users := range scales {
			mode := "modern"
			if legacy {
				mode = "legacy"
			}

			t.Run(fmt.Sprintf("%s/users=%d", mode, users), func(t *testing.T) {
				runRetirementScale(t, legacy, users)
			})
		}
	}
}

func runRetirementScale(t *testing.T, legacy bool, users int) {
	t.Helper()

	const opsPerUser, maxAttempts = 3, 4

	workers, connections := min(users, 32), min(users, 16)
	if users == 10 {
		connections = 4
	}

	barrier := min(workers, connections)
	goroutinesBefore := runtime.NumGoroutine()
	started := time.Now()

	peer := newRetirementPeer(t, legacy, false)
	registry, transport := newRetirementScaleRegistry(t, peer, connections, workers)
	retirementScaleFault(peer)

	overlap := newRetirementBarrier(barrier)

	peer.setGate(func(ctx context.Context, _ retirementExecution) error { return overlap.wait(ctx) })

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()

	counters := newRetirementScaleCounters()
	completed := make([]atomic.Int32, users)
	faulted := 0
	jobs := make(chan int)

	var wg, entered sync.WaitGroup

	entered.Add(workers)

	for range workers {
		wg.Go(func() {
			first := true

			for index := range jobs {
				owner := fmt.Sprintf("u%05d", index)
				caller := retirementCaller(ctx, owner, fmt.Sprintf("s%d", index))

				exit := counters.enter()

				if first {
					first = false

					entered.Done()
					entered.Wait()
				}

				captured, err := registry.Snapshot(caller)
				counters.record(err)
				counters.sample(registry)

				if err == nil {
					err = retirementCatalogMatches(caller, captured, owner)
				}

				if err != nil {
					exit()
					counters.failf("%s snapshot: %v", owner, err)

					continue
				}

				call, err := captured.Get(caller, "retire__"+retirementToolName(owner))
				if err != nil {
					exit()
					counters.failf("%s get: %v", owner, err)

					continue
				}

				done := int32(0)

				for op := range opsPerUser {
					name := fmt.Sprintf("%s/%d", owner, op)
					if op == 1 && index%5 == 2 {
						name = owner + "/fault"
					}

					for attempt := range maxAttempts {
						id := fmt.Sprintf("%s#%d", name, attempt)
						out, err := call.Execute(caller, retirementArgs(id))
						counters.record(err)
						counters.sample(registry)

						if err == nil {
							if want := owner + "|Bearer " + owner + ":" + fmt.Sprintf("s%d", index) + "|" + id; !strings.Contains(string(out), want) {
								counters.failf("%s result %s, want %s", id, out, want)
							} else {
								done++
							}

							break
						}
					}
				}

				exit()
				completed[index].Store(done)
			}
		})
	}

	for index := range users {
		if index%5 == 2 {
			faulted++
		}

		jobs <- index
	}

	close(jobs)
	wg.Wait()

	counters.mu.Lock()
	failures := append([]string(nil), counters.failures...)

	rejected := make(map[string]int64, len(counters.rejected))
	maps.Copy(rejected, counters.rejected)

	clientPeak := counters.samples
	counters.mu.Unlock()

	require.Empty(t, failures)
	require.Empty(t, counters.others, "distinct users never share a session, so every failure is classified")

	for index := range users {
		require.EqualValues(t, opsPerUser, completed[index].Load(), "user %d did not complete every operation", index)
	}

	retirementSettle(ctx, t, registry)

	final := retirementState(registry)
	server := peer.stats()

	require.NoError(t, registry.Close())
	transport.CloseIdleConnections()
	retirementAwait(ctx, t, "server resources to drain after Close", func() bool {
		stats := peer.stats()

		return stats.LiveSessions == 0 && stats.Listens == 0 && stats.Executing == 0
	})

	var mem runtime.MemStats

	runtime.ReadMemStats(&mem)

	report := retirementScaleReport{
		Scenario: "distinct-users", Mode: map[bool]string{true: "legacy", false: "modern"}[legacy],
		LogicalUsers: users, Workers: workers, MaxConnections: connections, OverlapBarrier: barrier,
		WorkerPeak: counters.peak.Load(), LogicalOps: users * (1 + opsPerUser), Attempts: counters.attempts.Load(),
		Succeeded: counters.succeeded.Load(), Rejected: rejected, FaultedOps: faulted, Reconnects: server.Handshakes - users,
		ClientPeak: clientPeak, Final: final, Server: server, ServerAfterClose: peer.stats(),
		GoroutinesBefore: goroutinesBefore, GoroutinesAfter: runtime.NumGoroutine(), HeapInuseMiB: mem.HeapInuse >> 20,
		DurationMS: time.Since(started).Milliseconds(),
	}
	retirementWriteReport(t, report)

	assert.EqualValues(t, users*(1+opsPerUser), report.Succeeded, "every snapshot and operation succeeded exactly once")
	assert.Equal(t, int64(workers), report.WorkerPeak, "all workers ran at once")
	assert.Equal(t, barrier, server.PeakExecuting, "the overlap barrier held exactly that many server executions at once")
	assert.LessOrEqual(t, clientPeak, connections, "resident plus closing sessions stay within the bound")
	assert.LessOrEqual(t, final.Entries, connections)
	assert.Equal(t, retirementPoolState{Entries: final.Entries}, final)
	assert.Zero(t, server.Duplicates, "no operation ID executed twice")
	assert.Zero(t, server.Leaks, "no cross-user execution or session use")
	assert.Equal(t, report.Succeeded-int64(users), int64(server.Executions), "only successful attempts executed business calls")
	assert.Zero(t, rejected["other"]+rejected["context"])
	assert.GreaterOrEqual(t, report.Reconnects, faulted, "every lost session was replaced")

	if legacy {
		want := map[string]int64{}
		if faulted > 0 {
			want["session_missing"] = int64(faulted)
		}

		assert.Equal(t, want, rejected, "exactly one failed attempt per lost session")
		assert.Equal(t, faulted, server.Kills)
		assert.Equal(t, connections, server.PeakSessions, "the server saw the full bound in use and never more")
		assert.Equal(t, final.Entries, server.LiveSessions)
		assert.Equal(t, server.Handshakes, server.Sessions)
	} else {
		assert.EqualValues(t, faulted, rejected["bad_request"])
		assert.LessOrEqual(t, rejected["connection_closed"], int64(faulted), "at most one retiring probe per lost connection")
		assert.Equal(t, faulted, server.Injected)
	}
}

// TestRegistryRetirementSinglePartitionContention drives one partition from
// 64 workers while its session is repeatedly lost. Every loss must produce
// exactly one replacement however many callers contend for it.
//
//nolint:tparallel,paralleltest // Contention runs one at a time to bound sockets and memory.
func TestRegistryRetirementSinglePartitionContention(t *testing.T) {
	t.Parallel()

	ops := 2048
	if testing.Short() {
		ops = 512
	}

	for _, legacy := range []bool{true, false} {
		for _, connections := range []int{1, 2} {
			mode := "modern"
			if legacy {
				mode = "legacy"
			}

			t.Run(fmt.Sprintf("%s/max=%d", mode, connections), func(t *testing.T) {
				runRetirementContention(t, legacy, connections, ops)
			})
		}
	}
}

func runRetirementContention(t *testing.T, legacy bool, connections, ops int) {
	t.Helper()

	const workers, maxAttempts, faultEvery = 64, 16, 64

	goroutinesBefore := runtime.NumGoroutine()
	started := time.Now()

	peer := newRetirementPeer(t, legacy, false)
	registry, transport := newRetirementScaleRegistry(t, peer, connections, workers)
	retirementScaleFault(peer)

	overlap := newRetirementBarrier(workers)

	peer.setGate(func(ctx context.Context, _ retirementExecution) error { return overlap.wait(ctx) })

	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Minute)
	defer cancel()

	caller := retirementCaller(ctx, "alice", "one")
	captured, err := registry.Snapshot(caller)
	require.NoError(t, err)
	require.NoError(t, retirementCatalogMatches(caller, captured, "alice"))

	call := retirementTool(caller, t, captured, "alice")
	counters := newRetirementScaleCounters()
	completed := make([]atomic.Bool, ops)
	jobs := make(chan int)
	faulted := 0

	// The first wave proves overlap on a healthy session; losses start after.
	faults := func(index int) bool { return index >= workers && index%faultEvery == faultEvery/4 }

	var wg, entered sync.WaitGroup

	entered.Add(workers)

	for range workers {
		wg.Go(func() {
			first := true

			for index := range jobs {
				name := fmt.Sprintf("alice/%d", index)
				if faults(index) {
					name += "/fault"
				}

				exit := counters.enter()

				if first {
					first = false

					entered.Done()
					entered.Wait()
				}

				for attempt := range maxAttempts {
					id := fmt.Sprintf("%s#%d", name, attempt)
					out, err := call.Execute(caller, retirementArgs(id))
					counters.record(err)
					counters.sample(registry)

					if err == nil {
						if !strings.Contains(string(out), "alice|Bearer alice:one|"+id) {
							counters.failf("%s result %s", id, out)
						} else {
							completed[index].Store(true)
						}

						break
					}
				}

				exit()
			}
		})
	}

	for index := range ops {
		if faults(index) {
			faulted++
		}

		jobs <- index
	}

	close(jobs)
	wg.Wait()

	counters.mu.Lock()
	failures := append([]string(nil), counters.failures...)

	rejected := make(map[string]int64, len(counters.rejected))
	maps.Copy(rejected, counters.rejected)

	clientPeak := counters.samples
	counters.mu.Unlock()

	require.Empty(t, failures)

	for index := range ops {
		require.True(t, completed[index].Load(), "operation %d never succeeded", index)
	}

	retirementSettle(ctx, t, registry)

	final := retirementState(registry)
	server := peer.stats()

	require.NoError(t, registry.Close())
	transport.CloseIdleConnections()
	retirementAwait(ctx, t, "server resources to drain after Close", func() bool {
		stats := peer.stats()

		return stats.LiveSessions == 0 && stats.Listens == 0 && stats.Executing == 0
	})

	var mem runtime.MemStats

	runtime.ReadMemStats(&mem)

	report := retirementScaleReport{
		Scenario: fmt.Sprintf("single-partition-max%d", connections), Mode: map[bool]string{true: "legacy", false: "modern"}[legacy],
		LogicalUsers: 1, Workers: workers, MaxConnections: connections, OverlapBarrier: workers,
		WorkerPeak: counters.peak.Load(), LogicalOps: ops, Attempts: counters.attempts.Load(), Succeeded: counters.succeeded.Load(),
		Rejected: rejected, OtherSamples: counters.others, FaultedOps: faulted, Reconnects: server.Handshakes - 1, ClientPeak: clientPeak, Final: final,
		Server: server, ServerAfterClose: peer.stats(), GoroutinesBefore: goroutinesBefore, GoroutinesAfter: runtime.NumGoroutine(),
		HeapInuseMiB: mem.HeapInuse >> 20, DurationMS: time.Since(started).Milliseconds(),
	}
	retirementWriteReport(t, report)

	// A call the client already failed may still run at the server while its
	// retry executes, so the server's peak can exceed the worker count.
	assert.EqualValues(t, ops, report.Succeeded)
	assert.Equal(t, int64(workers), report.WorkerPeak)
	assert.True(t, overlap.opened(), "all workers executed on one session at once")
	assert.GreaterOrEqual(t, server.PeakExecuting, workers)
	assert.LessOrEqual(t, clientPeak, connections)
	assert.Equal(t, retirementPoolState{Entries: 1}, final)
	assert.Zero(t, server.Duplicates, "no operation ID executed twice")
	assert.Zero(t, server.Leaks)
	assert.GreaterOrEqual(t, report.Reconnects, 1, "sessions were lost and replaced")

	if legacy {
		assert.Equal(t, server.Kills+1, server.Handshakes, "exactly one replacement per lost session")
		assert.Equal(t, 1, server.PeakSessions, "one live server session at a time")
		assert.Positive(t, server.Kills)
		assert.LessOrEqual(t, server.Kills, faulted)
	} else {
		assert.LessOrEqual(t, server.Handshakes, server.Injected+1, "no replacement without a lost connection")
		assert.Positive(t, server.Injected)
		assert.LessOrEqual(t, server.Injected, faulted)
	}
}

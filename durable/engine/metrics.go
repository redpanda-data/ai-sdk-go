// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"sync/atomic"
)

// Metrics are exposed in Prometheus text format on /metrics. The engine has no
// dependency on a metrics library; embedding services (which usually have one)
// can read the counters through Snapshot.
type Metrics struct {
	RunsStarted    atomic.Int64
	RunsCompleted  atomic.Int64
	RunsFailed     atomic.Int64
	RunsCancelled  atomic.Int64
	TasksDispatch  atomic.Int64
	LeasesExpired  atomic.Int64
	AttemptsFailed atomic.Int64
	Suspensions    atomic.Int64
	InputsReceived atomic.Int64
	TimersFired    atomic.Int64
	CommandsStale  atomic.Int64
	CommandsDup    atomic.Int64
	CommandsError  atomic.Int64
	JournalRecords atomic.Int64

	mu         sync.Mutex
	replayMSec int64
}

// Snapshot returns every counter by metric name.
func (m *Metrics) Snapshot() map[string]int64 {
	m.mu.Lock()
	replay := m.replayMSec
	m.mu.Unlock()

	return map[string]int64{
		"durable_runs_started_total":       m.RunsStarted.Load(),
		"durable_runs_completed_total":     m.RunsCompleted.Load(),
		"durable_runs_failed_total":        m.RunsFailed.Load(),
		"durable_runs_cancelled_total":     m.RunsCancelled.Load(),
		"durable_tasks_dispatched_total":   m.TasksDispatch.Load(),
		"durable_leases_expired_total":     m.LeasesExpired.Load(),
		"durable_attempts_failed_total":    m.AttemptsFailed.Load(),
		"durable_run_suspensions_total":    m.Suspensions.Load(),
		"durable_inputs_received_total":    m.InputsReceived.Load(),
		"durable_timers_fired_total":       m.TimersFired.Load(),
		"durable_commands_stale_total":     m.CommandsStale.Load(),
		"durable_commands_duplicate_total": m.CommandsDup.Load(),
		"durable_commands_error_total":     m.CommandsError.Load(),
		"durable_journal_records_total":    m.JournalRecords.Load(),
		"durable_replay_last_milliseconds": replay,
	}
}

// WriteTo writes the counters in Prometheus text exposition format.
func (m *Metrics) WriteTo(w io.Writer) (int64, error) {
	snap := m.Snapshot()

	names := make([]string, 0, len(snap))
	for k := range snap {
		names = append(names, k)
	}

	sort.Strings(names)

	var total int64

	for _, name := range names {
		typ := "counter"
		if name == "durable_replay_last_milliseconds" {
			typ = "gauge"
		}

		n, err := fmt.Fprintf(w, "# TYPE %s %s\n%s %d\n", name, typ, name, snap[name])
		total += int64(n)

		if err != nil {
			return total, err
		}
	}

	return total, nil
}

func (m *Metrics) setReplay(ms int64) {
	m.mu.Lock()
	m.replayMSec = ms
	m.mu.Unlock()
}

// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package engine

import (
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/redpanda-data/ai-sdk-go/durable"
)

// Handler returns the read-only HTTP API:
//
//	GET /healthz
//	GET /metrics
//	GET /v1/runs?status=<status>&limit=<n>
//	GET /v1/runs/{id}
//
// Mutations are never accepted over HTTP; the command topic is the write path.
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = e.Metrics.WriteTo(w)
	})
	mux.HandleFunc("GET /v1/runs", e.handleList)
	mux.HandleFunc("GET /v1/runs/{id}", e.handleGet)

	return mux
}

func (e *Engine) handleGet(w http.ResponseWriter, r *http.Request) {
	st, ok := e.Describe(r.PathValue("id"))
	if !ok {
		http.Error(w, `{"error":"run not found"}`, http.StatusNotFound)

		return
	}

	writeJSON(w, st)
}

func (e *Engine) handleList(w http.ResponseWriter, r *http.Request) {
	status := r.URL.Query().Get("status")

	limit := 100

	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			limit = n
		}
	}

	runs := e.List(status)
	if len(runs) > limit {
		runs = runs[:limit]
	}

	writeJSON(w, map[string]any{"runs": runs})
}

// Describe returns the current state of a run.
func (e *Engine) Describe(runID string) (durable.RunState, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	r, ok := e.runs[runID]
	if !ok {
		return durable.RunState{}, false
	}

	return r.st, true
}

// List returns run states, newest first, optionally filtered by status.
func (e *Engine) List(status string) []durable.RunState {
	e.mu.RLock()
	defer e.mu.RUnlock()

	out := make([]durable.RunState, 0, len(e.runs))

	for _, r := range e.runs {
		if status != "" && !strings.EqualFold(status, r.st.Status) {
			continue
		}

		st := r.st
		st.Messages = nil // summaries omit the payload; fetch one run for messages
		out = append(out, st)
	}

	sort.Slice(out, func(i, j int) bool {
		if out[i].UpdatedAt.Equal(out[j].UpdatedAt) {
			return out[i].RunID < out[j].RunID
		}

		return out[i].UpdatedAt.After(out[j].UpdatedAt)
	})

	return out
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(v); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

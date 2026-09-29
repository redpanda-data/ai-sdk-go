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

package a2a

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
)

// DeltaCoalescing bounds how long a streamed text delta waits before it
// becomes an A2A artifact event.
//
// The first delta of an artifact is sent immediately. Later deltas are
// buffered and sent as one append on a size trigger, when Interval has
// passed (a timer covers a provider stall), or on any non-delta event. Each
// flushed event carries at most one TextPart. A failed write drops its
// text, as it does with coalescing off.
//
// Executor.Cancel does not flush: up to one Interval of buffered text is
// dropped, not saved. The stored task still matches what was streamed.
type DeltaCoalescing struct {
	// Interval is the longest buffered text waits before it is sent. Zero
	// turns coalescing off: every delta is sent immediately, as before, and
	// MaxBytes has no effect. A negative value counts as zero.
	Interval time.Duration
	// MaxBytes sends the buffer once it holds at least this many bytes. Zero
	// disables the size trigger. It has no effect when Interval is zero. A
	// negative value counts as zero.
	MaxBytes int
}

// deltaWriter buffers one streamed artifact's text for a single
// processEvents call. The mutex serializes the event loop and the flush
// timer.
type deltaWriter struct {
	mu sync.Mutex

	reqCtx *a2asrv.RequestContext
	queue  eventqueue.Queue
	log    *slog.Logger
	cfg    DeltaCoalescing

	artifactID a2a.ArtifactID
	buf        strings.Builder
	lastSent   time.Time
	timer      *time.Timer
	closed     bool
}

func newDeltaWriter(reqCtx *a2asrv.RequestContext, queue eventqueue.Queue, log *slog.Logger, cfg DeltaCoalescing) *deltaWriter {
	if cfg.Interval < 0 {
		cfg.Interval = 0
	}

	if cfg.MaxBytes < 0 {
		cfg.MaxBytes = 0
	}

	return &deltaWriter{
		reqCtx: reqCtx,
		queue:  queue,
		log:    log,
		cfg:    cfg,
	}
}

// delta handles one streamed text chunk. With coalescing off it is sent
// immediately, as its own artifact event.
func (w *deltaWriter) delta(ctx context.Context, text string) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return
	}

	if w.cfg.Interval == 0 {
		w.sendImmediateLocked(ctx, text)
		return
	}

	if w.artifactID == "" {
		// Keep the ID only when the artifact exists, so later appends
		// never target an artifact the task does not have.
		artifact := a2a.NewArtifactEvent(w.reqCtx, a2a.TextPart{Text: text})
		w.lastSent = time.Now()

		if w.queueWrite(ctx, artifact) == nil {
			w.artifactID = artifact.Artifact.ID
		}

		return
	}

	w.buf.WriteString(text)

	elapsed := time.Since(w.lastSent)
	if (w.cfg.MaxBytes > 0 && w.buf.Len() >= w.cfg.MaxBytes) || elapsed >= w.cfg.Interval {
		w.flushLocked(ctx, false)
		return
	}

	w.armTimerLocked(ctx, w.cfg.Interval-elapsed)
}

// sendImmediateLocked is the coalescing-off path, unchanged from before.
// Caller holds w.mu.
func (w *deltaWriter) sendImmediateLocked(ctx context.Context, text string) {
	var artifact *a2a.TaskArtifactUpdateEvent

	if w.artifactID == "" {
		artifact = a2a.NewArtifactEvent(w.reqCtx, a2a.TextPart{Text: text})
		w.artifactID = artifact.Artifact.ID
	} else {
		artifact = a2a.NewArtifactUpdateEvent(w.reqCtx, w.artifactID, a2a.TextPart{Text: text})
	}

	_ = w.queueWrite(ctx, artifact)
}

// flushLocked sends the buffered text as one append and empties the
// buffer, also when the write fails: a retry could send the text twice,
// because the queue can fail after it delivered the event. An empty buffer
// with lastChunk false does nothing. Caller holds w.mu.
func (w *deltaWriter) flushLocked(ctx context.Context, lastChunk bool) {
	if w.buf.Len() == 0 && !lastChunk {
		return
	}

	var artifact *a2a.TaskArtifactUpdateEvent
	if w.buf.Len() > 0 {
		artifact = a2a.NewArtifactUpdateEvent(w.reqCtx, w.artifactID, a2a.TextPart{Text: w.buf.String()})
	} else {
		artifact = a2a.NewArtifactUpdateEvent(w.reqCtx, w.artifactID)
	}

	artifact.LastChunk = lastChunk

	_ = w.queueWrite(ctx, artifact)

	w.buf.Reset()
	w.lastSent = time.Now()
	w.stopTimerLocked()
}

// endArtifact flushes into a LastChunk event and clears the artifact ID.
func (w *deltaWriter) endArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.artifactID == "" {
		return
	}

	w.flushLocked(ctx, true)
	w.artifactID = ""
}

// resetArtifact flushes as a non-final append and clears the artifact ID.
func (w *deltaWriter) resetArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.flushLocked(ctx, false)
	w.artifactID = ""
}

// write flushes any pending text, then writes ev and returns its error.
func (w *deltaWriter) write(ctx context.Context, ev a2a.Event) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.closed {
		w.flushLocked(ctx, false)
	}

	return w.queue.Write(ctx, ev)
}

// onTimer is the flush timer's callback; it does nothing once closed.
func (w *deltaWriter) onTimer(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return
	}

	w.flushLocked(ctx, false)
}

// close stops the timer. processEvents defers it, so the timer never
// writes after processEvents returns.
func (w *deltaWriter) close() {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.stopTimerLocked()
	w.closed = true
}

func (w *deltaWriter) stopTimerLocked() {
	if w.timer != nil {
		w.timer.Stop()
	}
}

// armTimerLocked arms (or re-arms) the flush timer for d. The closure
// captures ctx instead of storing it on the struct.
func (w *deltaWriter) armTimerLocked(ctx context.Context, d time.Duration) {
	if w.timer == nil {
		w.timer = time.AfterFunc(d, func() { w.onTimer(ctx) })
		return
	}

	w.timer.Reset(d)
}

func (w *deltaWriter) queueWrite(ctx context.Context, ev a2a.Event) error {
	err := w.queue.Write(ctx, ev)
	if err != nil {
		w.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
	}

	return err
}

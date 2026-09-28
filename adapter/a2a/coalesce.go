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
// Interval is the longest a delta waits; zero turns coalescing off and
// every delta is sent immediately, byte for byte, as before. MaxBytes
// flushes once the buffer holds at least that many bytes; zero disables
// the size trigger. Negative values count as zero. Each flushed event
// carries exactly one TextPart.
//
// Cancel does not flush: up to one Interval of buffered text is dropped,
// not saved. The stored task still matches what was actually streamed.
type DeltaCoalescing struct {
	Interval time.Duration
	MaxBytes int
}

// deltaWriter buffers one streamed artifact's text for a single
// processEvents call and flushes it on a size trigger, an age trigger, or
// any non-delta event.
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

// newDeltaWriter treats a negative Interval or MaxBytes in cfg as zero.
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

// delta buffers one streamed text chunk. With coalescing off it is sent
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
		artifact := a2a.NewArtifactEvent(w.reqCtx, a2a.TextPart{Text: text})
		w.artifactID = artifact.Artifact.ID
		w.lastSent = time.Now()
		_ = w.queueWrite(ctx, artifact)

		return
	}

	w.buf.WriteString(text)

	elapsed := time.Since(w.lastSent)
	sizeTrigger := w.cfg.MaxBytes > 0 && w.buf.Len() >= w.cfg.MaxBytes
	ageTrigger := elapsed >= w.cfg.Interval

	if sizeTrigger || ageTrigger {
		w.flushLocked(ctx, false)
		return
	}

	w.armTimerLocked(ctx, w.cfg.Interval-elapsed)
}

// sendImmediateLocked sends text as its own artifact event. Caller holds w.mu.
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

// flushLocked sends the buffered text as one append. An empty buffer with
// lastChunk false does nothing. A failed write keeps the buffer for the
// next attempt. Caller holds w.mu.
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

	if err := w.queueWrite(ctx, artifact); err != nil {
		return
	}

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

// onTimer is the flush timer's callback; it no-ops once closed is set.
func (w *deltaWriter) onTimer(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return
	}

	w.flushLocked(ctx, false)
}

// close stops the timer and marks the writer closed.
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

// armTimerLocked arms (or re-arms) the flush timer for d. The AfterFunc
// closure captures ctx instead of storing it on the struct.
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

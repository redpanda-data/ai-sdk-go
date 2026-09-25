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

// DeltaCoalescing bounds how long a streamed text delta waits before the
// executor turns it into an A2A artifact event.
//
// Interval is the longest a streamed text delta waits before it is sent; a
// zero Interval turns coalescing off, and the executor sends one artifact
// event per delta, byte for byte, as it always has. MaxBytes is a flush
// threshold, not a hard cap: once the buffer holds at least that many
// bytes it flushes at once, but the single delta that crosses the
// threshold can itself be larger than MaxBytes, so one flushed event's
// text can exceed it. A zero MaxBytes means no size trigger. Negative
// values in either field count as zero.
//
// Per streamed artifact, the first delta goes out at once, so the time to
// first token does not change. Every later delta is buffered and flushed
// on the size trigger, on the age trigger, or on a MessageEvent,
// StreamReset, or any other event that ends or interrupts the artifact.
// Every flushed event carries exactly one TextPart.
//
// tasks/get can lag the live stream by up to Interval, because buffered
// text is not in the stored task until the next flush. Executor.Cancel
// flushes the buffer before writing the canceled status only when it is
// handed the exact same queue the running task is writing to; a2a-go's
// distributed (cluster) mode already gives Execute and Cancel different
// queues for the same task, in which case Cancel writes the status
// directly, as it always has, and up to Interval of already-streamed text
// can be missing from the stored artifact. Nothing already saved is lost.
type DeltaCoalescing struct {
	Interval time.Duration
	MaxBytes int
}

// deltaWriter buffers one streamed artifact's text for one processEvents
// call and flushes it on a size trigger, an age trigger, or any event that
// is not itself a delta. It holds no context.Context: the flush timer's
// AfterFunc closure captures the ctx that was live the first time the
// timer was armed for the current buffer, and keeps using that same ctx
// across any later Reset (Reset reschedules the existing callback; it
// does not rebind its captured ctx).
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

// newDeltaWriter creates a deltaWriter that writes to queue on behalf of
// reqCtx. A negative Interval or MaxBytes in cfg is treated as zero.
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

// delta handles one streamed text chunk. With coalescing off it sends the
// chunk at once, exactly as the executor always has. Otherwise the first
// chunk of an artifact still goes out at once; later chunks buffer until a
// size or age trigger flushes them.
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

// sendImmediateLocked reproduces the executor's original per-delta
// behavior: every chunk becomes its own artifact event. The caller must
// hold w.mu.
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

// flushLocked sends the buffered text as one append to the current
// artifact. With an empty buffer it does nothing unless lastChunk is set,
// in which case it sends the empty LastChunk event that ends the
// artifact. If the write fails, the buffer, lastSent and timer are left
// as they are, so a later flush (the next delta, the retry timer, or a
// terminal write) gets another chance to deliver the text instead of
// losing it. The caller must hold w.mu.
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

// endArtifact flushes any pending text into the artifact's LastChunk event
// and forgets the artifact ID, so the next delta opens a new artifact.
func (w *deltaWriter) endArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.artifactID == "" {
		return
	}

	w.flushLocked(ctx, true)
	w.artifactID = ""
}

// resetArtifact flushes any pending text as a non-final append to the
// current artifact and forgets its ID, without sending a LastChunk event.
// A model_call status uses this: the old artifact is done, but nothing
// else tells the client so.
func (w *deltaWriter) resetArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.flushLocked(ctx, false)
	w.artifactID = ""
}

// write flushes any pending text and then writes ev, in that order, so
// buffered text always reaches the queue before the event that follows
// it, and returns ev's own write error (not the flush's, which is already
// logged by queueWrite). A closed writer skips the flush and only passes
// ev through, which lets the cancel branch's own canceled status still
// get written.
func (w *deltaWriter) write(ctx context.Context, ev a2a.Event) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.closed {
		w.flushLocked(ctx, false)
	}

	return w.queue.Write(ctx, ev)
}

// cancel flushes any pending text, then writes ev with a write bounded to
// 30 seconds (the same bound the executor's own context-canceled branch
// uses for its background context), so a stuck queue cannot make Cancel
// hold w.mu forever. closed is set, and the timer stopped, only once ev
// is actually written: a failed cancel leaves the writer running normally
// rather than silently disabling it.
func (w *deltaWriter) cancel(ctx context.Context, ev a2a.Event) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.flushLocked(ctx, false)

	writeCtx, stop := context.WithTimeout(ctx, 30*time.Second)
	defer stop()

	if err := w.queue.Write(writeCtx, ev); err != nil {
		return err
	}

	w.closed = true
	w.stopTimerLocked()

	return nil
}

// onTimer is the time.AfterFunc callback. It flushes the pending text once
// Interval passes without a new delta, unless the writer is already
// closed, so a timer that fires after the request ends is harmless.
func (w *deltaWriter) onTimer(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return
	}

	w.flushLocked(ctx, false)
}

// close stops the timer and marks the writer closed, so a timer callback
// already in flight becomes a no-op once it acquires w.mu.
func (w *deltaWriter) close() {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.stopTimerLocked()
	w.closed = true
}

// stopTimerLocked stops the pending flush timer, if any. The caller must
// hold w.mu.
func (w *deltaWriter) stopTimerLocked() {
	if w.timer != nil {
		w.timer.Stop()
	}
}

// armTimerLocked arms (or re-arms) the flush timer for d. The AfterFunc
// closure captures ctx rather than storing it on the struct; Reset reuses
// that same closure, so onTimer keeps running with the ctx captured the
// first time this artifact's timer was armed, not a later one. The caller
// must hold w.mu.
func (w *deltaWriter) armTimerLocked(ctx context.Context, d time.Duration) {
	if w.timer == nil {
		w.timer = time.AfterFunc(d, func() { w.onTimer(ctx) })
		return
	}

	w.timer.Reset(d)
}

// queueWrite writes ev to the queue and logs a failure the same way the
// executor's write closure always has. The caller must hold w.mu.
func (w *deltaWriter) queueWrite(ctx context.Context, ev a2a.Event) error {
	err := w.queue.Write(ctx, ev)
	if err != nil {
		w.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
	}

	return err
}

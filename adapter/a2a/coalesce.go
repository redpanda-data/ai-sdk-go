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
	"unicode/utf8"

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
// flushed event carries one non-empty TextPart. A failed write drops its
// text, as it does with coalescing off.
//
// With or without coalescing, the last character of the text streamed so far
// is held back until the next send, so the event that closes an artifact
// always has text to carry LastChunk: A2A v1.0 rejects an artifact update
// without parts.
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
	// held is the last character of the artifact's text, not yet sent.
	held     string
	lastSent time.Time
	timer    *time.Timer
	closed   bool
}

// sendMode says how much of the pending text a send writes.
type sendMode int

const (
	// sendHold writes all but the last character, which stays held.
	sendHold sendMode = iota
	// sendDrain writes everything without closing the artifact.
	sendDrain
	// sendClose writes everything and marks the artifact complete.
	sendClose
)

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

	if w.closed || text == "" {
		return
	}

	if w.cfg.Interval == 0 || w.artifactID == "" {
		w.sendLocked(ctx, text, sendHold)
		return
	}

	w.buf.WriteString(text)

	elapsed := time.Since(w.lastSent)
	if (w.cfg.MaxBytes > 0 && w.buf.Len() >= w.cfg.MaxBytes) || elapsed >= w.cfg.Interval {
		w.flushLocked(ctx, sendHold)
		return
	}

	w.armTimerLocked(ctx, w.cfg.Interval-elapsed)
}

// sendLocked writes the held character, then text, as the artifact's next
// event, keeping the last character back in sendHold mode. Nothing is written
// when that leaves no text. The first event creates the artifact. With
// coalescing on, the ID is kept only when that write succeeds, so later
// appends never target an artifact the task does not have, and the held
// character is dropped with the failed write. Caller holds w.mu.
func (w *deltaWriter) sendLocked(ctx context.Context, text string, mode sendMode) {
	text = w.held + text
	w.held = ""

	if mode == sendHold {
		_, size := utf8.DecodeLastRuneInString(text)
		text, w.held = text[:len(text)-size], text[len(text)-size:]
	}

	if text == "" {
		return
	}

	var artifact *a2a.TaskArtifactUpdateEvent

	creates := w.artifactID == ""
	if creates {
		artifact = a2a.NewArtifactEvent(w.reqCtx, a2a.TextPart{Text: text})
	} else {
		artifact = a2a.NewArtifactUpdateEvent(w.reqCtx, w.artifactID, a2a.TextPart{Text: text})
	}

	artifact.LastChunk = mode == sendClose
	w.lastSent = time.Now()

	err := w.queueWrite(ctx, artifact)

	switch {
	case !creates:
	case err == nil || w.cfg.Interval == 0:
		w.artifactID = artifact.Artifact.ID
	default:
		w.held = ""
	}
}

// flushLocked sends the buffered text and empties the buffer, also when the
// write fails: a retry could send the text twice, because the queue can fail
// after it delivered the event. Caller holds w.mu.
func (w *deltaWriter) flushLocked(ctx context.Context, mode sendMode) {
	if w.buf.Len() == 0 && mode == sendHold {
		return
	}

	text := w.buf.String()
	w.buf.Reset()
	w.sendLocked(ctx, text, mode)
	w.stopTimerLocked()
}

// endArtifact sends the remaining text as the artifact's LastChunk and
// clears the artifact ID.
func (w *deltaWriter) endArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.artifactID == "" && w.held == "" {
		return
	}

	w.flushLocked(ctx, sendClose)
	w.artifactID = ""
}

// resetArtifact sends the remaining text as a non-final append and clears
// the artifact ID.
func (w *deltaWriter) resetArtifact(ctx context.Context) {
	w.mu.Lock()
	defer w.mu.Unlock()

	w.flushLocked(ctx, sendDrain)
	w.artifactID = ""
}

// write sends any pending text, then writes ev and returns its error.
func (w *deltaWriter) write(ctx context.Context, ev a2a.Event) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if !w.closed {
		w.flushLocked(ctx, sendDrain)
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

	w.flushLocked(ctx, sendHold)
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

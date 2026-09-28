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
	"time"

	"github.com/a2aproject/a2a-go/a2a"
	"github.com/a2aproject/a2a-go/a2asrv"
	"github.com/a2aproject/a2a-go/a2asrv/eventqueue"
)

// DeltaCoalescing bounds how long a streamed text delta waits before it
// becomes an A2A artifact event.
//
// The first delta of an artifact is sent immediately. Later deltas are
// buffered and sent as one append on a size trigger, an age trigger, or any
// non-delta event. The age is checked when the next event arrives, so text
// can wait longer than Interval while the provider sends nothing. Each
// flushed event carries at most one TextPart. A failed write drops its text,
// as it does with coalescing off.
//
// Executor.Cancel does not flush: the unsent buffer is dropped, not saved.
// The stored task still matches what was actually streamed.
type DeltaCoalescing struct {
	// Interval is how long buffered text waits before it is sent. Zero turns
	// coalescing off: every delta is sent immediately, as before, and
	// MaxBytes has no effect. A negative value counts as zero.
	Interval time.Duration
	// MaxBytes sends the buffer once it holds at least this many bytes. Zero
	// disables the size trigger. It has no effect when Interval is zero. A
	// negative value counts as zero.
	MaxBytes int
}

// deltaWriter buffers one streamed artifact's text for a single
// processEvents call. It is not safe for concurrent use.
type deltaWriter struct {
	reqCtx *a2asrv.RequestContext
	queue  eventqueue.Queue
	log    *slog.Logger
	cfg    DeltaCoalescing

	artifactID a2a.ArtifactID
	buf        strings.Builder
	lastSent   time.Time
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
	if w.cfg.Interval == 0 {
		w.sendImmediate(ctx, text)
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

	if w.cfg.MaxBytes > 0 && w.buf.Len() >= w.cfg.MaxBytes {
		w.flush(ctx, false)
	}
}

// tick sends the buffer once it is older than Interval. processEvents calls
// it after every event.
func (w *deltaWriter) tick(ctx context.Context) {
	if w.buf.Len() > 0 && time.Since(w.lastSent) >= w.cfg.Interval {
		w.flush(ctx, false)
	}
}

// sendImmediate is the coalescing-off path, unchanged from before.
func (w *deltaWriter) sendImmediate(ctx context.Context, text string) {
	var artifact *a2a.TaskArtifactUpdateEvent

	if w.artifactID == "" {
		artifact = a2a.NewArtifactEvent(w.reqCtx, a2a.TextPart{Text: text})
		w.artifactID = artifact.Artifact.ID
	} else {
		artifact = a2a.NewArtifactUpdateEvent(w.reqCtx, w.artifactID, a2a.TextPart{Text: text})
	}

	_ = w.queueWrite(ctx, artifact)
}

// flush sends the buffered text as one append and empties the buffer, also
// when the write fails: a retry could send the text twice, because the
// queue can fail after it delivered the event. An empty buffer with
// lastChunk false does nothing.
func (w *deltaWriter) flush(ctx context.Context, lastChunk bool) {
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
}

// endArtifact flushes into a LastChunk event and clears the artifact ID.
func (w *deltaWriter) endArtifact(ctx context.Context) {
	if w.artifactID == "" {
		return
	}

	w.flush(ctx, true)
	w.artifactID = ""
}

// resetArtifact flushes as a non-final append and clears the artifact ID.
func (w *deltaWriter) resetArtifact(ctx context.Context) {
	w.flush(ctx, false)
	w.artifactID = ""
}

// write flushes any pending text, then writes ev and returns its error.
func (w *deltaWriter) write(ctx context.Context, ev a2a.Event) error {
	w.flush(ctx, false)

	return w.queue.Write(ctx, ev)
}

func (w *deltaWriter) queueWrite(ctx context.Context, ev a2a.Event) error {
	err := w.queue.Write(ctx, ev)
	if err != nil {
		w.log.ErrorContext(ctx, "Failed to write to queue", "error", err)
	}

	return err
}

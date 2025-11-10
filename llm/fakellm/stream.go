package fakellm

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// fakeStream implements llm.EventStream with configurable events and timing.
type fakeStream struct {
	ctx    context.Context
	cancel context.CancelFunc
	events []llm.StreamEvent
	delay  time.Duration

	mu     sync.Mutex
	index  int
	closed bool
}

// newFakeStream creates a new fake stream that emits the given events.
func newFakeStream(parent context.Context, events []llm.StreamEvent, interChunkDelay time.Duration) llm.EventStream {
	ctx, cancel := context.WithCancel(parent)

	return &fakeStream{
		ctx:    ctx,
		cancel: cancel,
		events: events,
		delay:  interChunkDelay,
	}
}

// Recv returns the next event or io.EOF when complete.
func (s *fakeStream) Recv() (llm.StreamEvent, error) {
	// Fast checks without holding the lock
	select {
	case <-s.ctx.Done():
		// Check if it was a normal cancellation or stream closure
		s.mu.Lock()
		wasClosed := s.closed
		s.mu.Unlock()

		if wasClosed {
			return nil, llm.ErrStreamClosed
		}

		return nil, s.ctx.Err()
	default:
	}

	// Check state and determine if we need to delay
	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return nil, llm.ErrStreamClosed
	}

	i := s.index
	done := i >= len(s.events)
	needDelay := s.delay > 0 && i > 0
	s.mu.Unlock()

	if done {
		return nil, io.EOF
	}

	// Simulate inter-chunk delay outside the lock
	if needDelay {
		timer := time.NewTimer(s.delay)
		defer timer.Stop()

		select {
		case <-s.ctx.Done():
			s.mu.Lock()
			wasClosed := s.closed
			s.mu.Unlock()

			if wasClosed {
				return nil, llm.ErrStreamClosed
			}

			return nil, s.ctx.Err()
		case <-timer.C:
		}
	}

	// Reacquire lock and check state again after delay
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil, llm.ErrStreamClosed
	}

	if s.index >= len(s.events) {
		return nil, io.EOF
	}

	event := s.events[s.index]
	s.index++

	return event, nil
}

// Close releases resources and marks the stream as closed.
// This will unblock any goroutines waiting in Recv().
func (s *fakeStream) Close() error {
	s.mu.Lock()

	if s.closed {
		s.mu.Unlock()
		return nil
	}

	s.closed = true
	s.mu.Unlock()

	// Cancel the context to unblock any waiting Recv() calls
	s.cancel()

	return nil
}

// streamWrapper wraps an EventStream and calls a function on Close.
type streamWrapper struct {
	llm.EventStream

	onClose func()
	once    sync.Once
}

// Close calls the onClose function exactly once.
func (sw *streamWrapper) Close() error {
	var err error

	sw.once.Do(func() {
		err = sw.EventStream.Close()
		if sw.onClose != nil {
			sw.onClose()
		}
	})

	return err
}

// errorAfterStream wraps an EventStream and returns an error after N events.
// This allows testing mid-stream errors where some chunks are successfully
// delivered before the error occurs.
type errorAfterStream struct {
	inner llm.EventStream
	n     int
	seen  int
	err   error
}

// newErrorAfterStream creates a stream that emits n events from inner, then returns err.
func newErrorAfterStream(inner llm.EventStream, n int, err error) llm.EventStream {
	return &errorAfterStream{
		inner: inner,
		n:     n,
		err:   err,
	}
}

// Recv returns events from the inner stream until n events have been seen,
// then returns the configured error.
func (e *errorAfterStream) Recv() (llm.StreamEvent, error) {
	if e.seen >= e.n {
		return nil, e.err
	}

	ev, err := e.inner.Recv()
	if err == nil {
		e.seen++
	}

	return ev, err
}

// Close closes the inner stream.
func (e *errorAfterStream) Close() error {
	return e.inner.Close()
}

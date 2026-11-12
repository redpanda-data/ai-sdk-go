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
	"errors"
	"sync/atomic"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
)

type registryEntry struct {
	partition string
	ready     chan struct{}
	session   *sdkmcp.ClientSession
	err       error
	abandoned bool
	leases    int
	lastUsed  time.Time
	retired   bool
	identity  atomic.Pointer[registryCallerContext]

	catalogGate       chan struct{}
	generation        atomic.Uint64
	catalogGeneration uint64
	catalog           []*sdkmcp.Tool
	catalogExpires    time.Time
	fetchTTL          time.Duration
}

func (r *Registry) begin(ctx context.Context) (context.Context, func(), error) {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()

		return nil, nil, ErrClosed
	}

	r.wg.Add(1)
	r.mu.Unlock()

	ctx, cancel := opContext(r.ctx, ctx)

	return ctx, func() { cancel(); r.wg.Done() }, nil
}

func (r *Registry) signalLocked() {
	close(r.changed)
	r.changed = make(chan struct{})
}

func (r *Registry) acquire(ctx context.Context, partition string) (*registryEntry, func(), error) {
	if _, err := r.credentials(ctx, partition); err != nil {
		return nil, nil, err
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}

		r.mu.Lock()
		if r.closed {
			r.mu.Unlock()

			return nil, nil, ErrClosed
		}

		now := r.config.now()
		var evicted []*registryEntry

		for key, entry := range r.entries {
			if entry.leases == 0 && r.config.idleTimeout > 0 && now.Sub(entry.lastUsed) >= r.config.idleTimeout {
				delete(r.entries, key)

				r.closing++

				evicted = append(evicted, entry)
			}
		}

		// Retired entries are never in entries, so a lookup cannot return one.
		entry := r.entries[partition]
		connect := entry == nil

		// Evict only when the sessions already closing cannot satisfy the
		// callers already waiting, so repeated wakeups do not cascade.
		if connect && len(r.entries)+r.closing >= r.config.maxConnections && r.closing-r.draining <= r.waiting {
			var oldest *registryEntry
			for _, entry := range r.entries {
				if entry.leases == 0 && (oldest == nil || entry.lastUsed.Before(oldest.lastUsed)) {
					oldest = entry
				}
			}

			if oldest != nil {
				delete(r.entries, oldest.partition)

				r.closing++

				evicted = append(evicted, oldest)
			}
		}

		if connect && len(r.entries)+r.closing >= r.config.maxConnections {
			changed := r.changed
			r.waiting++
			r.mu.Unlock()

			r.closeDetached(evicted)

			select {
			case <-changed:
			case <-ctx.Done():
			}

			r.mu.Lock()
			r.waiting--
			r.mu.Unlock()

			continue
		}

		if connect {
			// The connector holds its own lease until ready, so a connecting
			// entry is never evicted and keeps its slot if the caller leaves.
			entry = &registryEntry{partition: partition, ready: make(chan struct{}), leases: 1, catalogGate: make(chan struct{}, 1)}
			r.entries[partition] = entry
			r.wg.Add(1)
		}

		entry.rememberIdentity(ctx)
		entry.leases++
		r.mu.Unlock()

		r.closeDetached(evicted)

		if connect {
			go r.connectEntry(ctx, entry)
		}

		select {
		case <-entry.ready:
			if entry.err != nil {
				r.release(entry)

				// Another caller's cancellation must not fail a live waiter.
				if entry.abandoned && ctx.Err() == nil {
					continue
				}

				return nil, nil, entry.err
			}

			return entry, func() { r.release(entry) }, nil
		case <-ctx.Done():
			r.release(entry)

			return nil, nil, ctx.Err()
		}
	}
}

// connectEntry runs the SDK handshake on the first caller's operation context.
// go-sdk may close a half-initialized session synchronously, so the caller
// waits on ready instead and can leave at its own deadline.
func (r *Registry) connectEntry(ctx context.Context, entry *registryEntry) {
	defer r.wg.Done()

	client := r.newSDKClient(entry)
	transport := &registryTransport{
		StreamableClientTransport: &sdkmcp.StreamableClientTransport{
			Endpoint: r.endpoint.String(), HTTPClient: r.httpClient(entry.partition),
		},
		entry: entry,
	}
	session, err := client.Connect(ctx, transport, nil)

	r.mu.Lock()
	entry.session, entry.err = session, err
	entry.abandoned = err != nil && ctx.Err() != nil

	if err != nil && r.entries[entry.partition] == entry {
		delete(r.entries, entry.partition)
		r.signalLocked()
	}

	close(entry.ready)
	r.mu.Unlock()

	r.release(entry)
}

func (r *Registry) release(entry *registryEntry) {
	r.mu.Lock()
	entry.leases--
	entry.lastUsed = r.config.now()

	closeEntry := entry.leases == 0 && entry.retired
	if closeEntry {
		r.draining--
	}

	r.signalLocked()
	r.mu.Unlock()

	if closeEntry {
		r.closeDetached([]*registryEntry{entry})
	}
}

// retireClosed detaches a session the SDK reports closed or missing. New
// operations reconnect; current lease holders unwind on the old session, which
// keeps its capacity until the last release closes it.
func (r *Registry) retireClosed(entry *registryEntry, err error) {
	if !errors.Is(err, sdkmcp.ErrConnectionClosed) && !errors.Is(err, sdkmcp.ErrSessionMissing) {
		return
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if entry.retired {
		return
	}

	entry.retired = true
	if r.entries[entry.partition] == entry {
		delete(r.entries, entry.partition)
	}

	r.closing++
	r.draining++
}

// detachedClosed returns the capacity of n detached sessions whose Close returned.
func (r *Registry) detachedClosed(n int) {
	r.mu.Lock()
	r.closing -= n
	r.signalLocked()
	r.mu.Unlock()
}

// Shutdown cancels operations and capacity waiters, drains leases, and closes all
// SDK sessions. If ctx expires, cleanup continues; a later Shutdown can wait again.
func (r *Registry) Shutdown(ctx context.Context) error {
	r.mu.Lock()
	if !r.closed {
		r.closed = true
		r.cancel()
		r.signalLocked()

		go r.shutdown()
	}
	r.mu.Unlock()

	select {
	case <-r.done:
		return r.closeErr
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (r *Registry) shutdown() {
	r.wg.Wait()

	r.mu.Lock()
	entries := make([]*registryEntry, 0, len(r.entries))

	for _, entry := range r.entries {
		entries = append(entries, entry)
	}

	r.entries = make(map[string]*registryEntry)
	r.mu.Unlock()

	r.closeErr = r.finishCleanup(entries)
	close(r.done)
}

// Close shuts down the registry with a 30-second timeout.
func (r *Registry) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return r.Shutdown(ctx)
}

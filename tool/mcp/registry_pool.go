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
	leases    int
	lastUsed  time.Time
	retired   bool

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

				evicted = append(evicted, entry)
			}
		}

		if entry := r.entries[partition]; entry != nil {
			entry.leases++
			r.mu.Unlock()

			_ = closeRegistryEntries(evicted)

			select {
			case <-entry.ready:
				if entry.err != nil {
					r.release(entry)

					return nil, nil, entry.err
				}

				return entry, func() { r.release(entry) }, nil
			case <-ctx.Done():
				r.release(entry)

				return nil, nil, ctx.Err()
			}
		}

		if len(r.entries) >= r.config.maxConnections {
			var oldest *registryEntry
			for _, entry := range r.entries {
				if entry.leases == 0 && (oldest == nil || entry.lastUsed.Before(oldest.lastUsed)) {
					oldest = entry
				}
			}

			if oldest != nil {
				delete(r.entries, oldest.partition)
				evicted = append(evicted, oldest)
			}
		}

		if len(r.entries) >= r.config.maxConnections {
			changed := r.changed
			r.mu.Unlock()

			_ = closeRegistryEntries(evicted)

			select {
			case <-changed:
				continue
			case <-ctx.Done():
				return nil, nil, ctx.Err()
			}
		}

		entry := &registryEntry{partition: partition, ready: make(chan struct{}), leases: 1, catalogGate: make(chan struct{}, 1)}
		r.entries[partition] = entry
		r.mu.Unlock()

		_ = closeRegistryEntries(evicted)

		client := r.newSDKClient(ctx, entry)
		session, err := client.Connect(ctx, &sdkmcp.StreamableClientTransport{
			Endpoint: r.endpoint.String(), HTTPClient: r.httpClient(partition),
		}, nil)

		entry.session, entry.err = session, err

		r.mu.Lock()
		if err != nil && r.entries[partition] == entry {
			delete(r.entries, partition)
			r.signalLocked()
		}

		close(entry.ready)
		r.mu.Unlock()

		if err != nil {
			r.release(entry)

			return nil, nil, err
		}

		return entry, func() { r.release(entry) }, nil
	}
}

func (r *Registry) release(entry *registryEntry) {
	r.mu.Lock()
	entry.leases--
	entry.lastUsed = r.config.now()

	closeEntry := entry.leases == 0 && entry.retired
	if closeEntry && r.entries[entry.partition] == entry {
		delete(r.entries, entry.partition)
	}

	r.signalLocked()
	r.mu.Unlock()

	if closeEntry {
		_ = closeRegistryEntries([]*registryEntry{entry})
	}
}

func (r *Registry) retireClosed(entry *registryEntry, err error) {
	if errors.Is(err, sdkmcp.ErrConnectionClosed) || errors.Is(err, sdkmcp.ErrSessionMissing) {
		r.mu.Lock()
		entry.retired = true
		r.mu.Unlock()
	}
}

func closeRegistryEntries(entries []*registryEntry) error {
	var errs []error

	for _, entry := range entries {
		if entry.session != nil {
			errs = append(errs, entry.session.Close())
		}
	}

	return errors.Join(errs...)
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

	r.closeErr = closeRegistryEntries(entries)
	close(r.done)
}

// Close shuts down the registry with a 30-second timeout.
func (r *Registry) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return r.Shutdown(ctx)
}

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
	"errors"
	"fmt"
	"sync"
)

// closeDetached closes sessions already detached from entries without blocking
// the caller. go-sdk's Close sends the legacy session-termination DELETE on a
// detached context; its five-second timeout bounds that request only. Close
// itself can take longer, for example while the SDK's reader holds its
// connection lock for its own DELETE or outstanding requests drain. Each
// session holds its capacity until Close returns, so these goroutines never
// exceed maxConnections. Callers must not hold r.mu and must run while r.wg is
// held (an operation or connect), so every Add precedes shutdown's cleanup Wait.
func (r *Registry) closeDetached(entries []*registryEntry) {
	if len(entries) == 0 {
		return
	}

	r.cleanup.Add(len(entries))

	for _, entry := range entries {
		go func() {
			defer r.cleanup.Done()

			if entry.session != nil {
				r.cleanupErrs.record(entry.session.Close())
			}

			r.detachedClosed(1)
		}()
	}
}

// finishCleanup closes the remaining sessions concurrently with detached
// closes still in flight, waits for both, and reports every failure. The wait
// has no time bound of its own: Shutdown's context bounds only its caller, and
// a later Shutdown can wait again.
func (r *Registry) finishCleanup(entries []*registryEntry) error {
	err := closeRegistrySessions(entries)

	r.cleanup.Wait()

	return errors.Join(err, r.cleanupErrs.err())
}

// closeRegistrySessions closes sessions concurrently and joins their errors in
// input order, so shutdown overlaps their termination DELETEs rather than
// serializing them.
func closeRegistrySessions(entries []*registryEntry) error {
	errs := make([]error, len(entries))

	var wg sync.WaitGroup

	for i, entry := range entries {
		if entry.session != nil {
			wg.Go(func() { errs[i] = entry.session.Close() })
		}
	}

	wg.Wait()

	return errors.Join(errs...)
}

// registryCleanupErrors keeps detached close failures for Shutdown in bounded
// memory: the first error and a count, not one error per evicted session.
type registryCleanupErrors struct {
	mu       sync.Mutex
	first    error
	failures int
}

func (e *registryCleanupErrors) record(err error) {
	if err == nil {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if e.first == nil {
		e.first = err
	}

	e.failures++
}

func (e *registryCleanupErrors) err() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.failures == 0 {
		return nil
	}

	return fmt.Errorf("mcp: %d detached session close(s) failed, first: %w", e.failures, e.first)
}

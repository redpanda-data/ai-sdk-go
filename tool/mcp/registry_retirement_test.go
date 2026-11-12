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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/tool"
)

const retirementInitializeMethod = "initialize"

type retirementIdentityKey struct{}

type retirementIdentity struct {
	owner  string
	secret string
	hold   *retirementHold
}

// retirementHold parks one tool call inside its HTTP credential resolution, so
// the caller keeps its pool lease while the session is lost underneath it.
type retirementHold struct {
	reached chan struct{}
	release chan struct{}
	once    sync.Once
}

func newRetirementHold() *retirementHold {
	return &retirementHold{reached: make(chan struct{}), release: make(chan struct{})}
}

func (h *retirementHold) wait(ctx context.Context) error {
	h.once.Do(func() { close(h.reached) })

	select {
	case <-h.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func retirementCaller(ctx context.Context, owner, secret string) context.Context {
	return context.WithValue(ctx, retirementIdentityKey{}, retirementIdentity{owner: owner, secret: secret})
}

func retirementHeldCaller(ctx context.Context, owner, secret string, hold *retirementHold) context.Context {
	return context.WithValue(ctx, retirementIdentityKey{}, retirementIdentity{owner: owner, secret: secret, hold: hold})
}

func retirementCredentials(ctx context.Context) (Credentials, error) {
	identity, ok := ctx.Value(retirementIdentityKey{}).(retirementIdentity)
	if !ok {
		return Credentials{}, errors.New("retirement caller identity missing")
	}

	if _, sending := ctx.Value(registryParameterHeadersKey{}).(http.Header); sending && identity.hold != nil {
		if err := identity.hold.wait(ctx); err != nil {
			return Credentials{}, err
		}
	}

	return Credentials{PartitionKey: identity.owner, Headers: http.Header{"Authorization": {"Bearer " + identity.owner + ":" + identity.secret}}}, nil
}

func retirementOwner(authorization string) string {
	owner, _, _ := strings.Cut(strings.TrimPrefix(authorization, "Bearer "), ":")

	return owner
}

type retirementWire struct {
	httpMethod string
	rpcMethod  string
	owner      string
	session    string
	op         string
}

type retirementExecution struct {
	op      string
	owner   string
	server  string
	bearer  string
	session string
}

type retirementServerSession struct {
	owner string
	open  bool
}

// retirementPeer is a loopback MCP backend built on the real go-sdk HTTP
// handler. Every user gets its own server with an identity-specific tool, and
// the wire wrapper records handshakes, sessions, listen streams and business
// executions as the server observes them. Responses are recorded when their
// header is written, before the client can act on them.
type retirementPeer struct {
	legacy bool
	sse    bool
	url    string

	handler http.Handler
	servers sync.Map

	// fault runs before the SDK handler and may block on the request context;
	// a nonzero status is returned to the client instead of serving it.
	fault atomic.Pointer[func(*http.Request, retirementWire) int]
	gate  atomic.Pointer[func(context.Context, retirementExecution) error]

	mu            sync.Mutex
	methods       map[string]int
	handshakes    map[string]int
	sessions      map[string]*retirementServerSession
	deletes       map[string]int
	executions    map[string][]retirementExecution
	liveSessions  int
	peakSessions  int
	listens       int
	peakListens   int
	executing     int
	peakExecuting int
	injected      int
	kills         int
	leaks         []string

	// snapshotted, if set before use, runs after terminateMatching has read
	// the server's sessions and before it claims any of them.
	snapshotted func()
}

func newRetirementPeer(tb testing.TB, legacy, sse bool) *retirementPeer {
	tb.Helper()

	peer := &retirementPeer{
		legacy: legacy, sse: sse,
		methods: make(map[string]int), handshakes: make(map[string]int), sessions: make(map[string]*retirementServerSession),
		deletes: make(map[string]int), executions: make(map[string][]retirementExecution),
	}
	peer.handler = sdkmcp.NewStreamableHTTPHandler(func(req *http.Request) *sdkmcp.Server {
		owner := retirementOwner(req.Header.Get("Authorization"))
		if owner == "" {
			return nil
		}

		return peer.server(owner)
	}, &sdkmcp.StreamableHTTPOptions{Stateless: !legacy, JSONResponse: true})

	server := httptest.NewServer(peer)
	peer.url = server.URL
	tb.Cleanup(server.Close)

	return peer
}

type retirementResponse struct {
	http.ResponseWriter

	code   int
	header func(code int, header http.Header)
}

func (r *retirementResponse) WriteHeader(code int) {
	r.commit(code)
	r.ResponseWriter.WriteHeader(code)
}

func (r *retirementResponse) Write(b []byte) (int, error) {
	r.commit(http.StatusOK)

	return r.ResponseWriter.Write(b)
}

func (r *retirementResponse) Flush() {
	r.commit(http.StatusOK)

	if flusher, ok := r.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (r *retirementResponse) Unwrap() http.ResponseWriter { return r.ResponseWriter }

func (r *retirementResponse) commit(code int) {
	if r.code != 0 {
		return
	}

	r.code = code
	r.header(code, r.Header())
}

func (p *retirementPeer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	var message struct {
		Method string `json:"method"`
		Params struct {
			Arguments struct {
				Op string `json:"op"`
			} `json:"arguments"`
		} `json:"params"`
	}

	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)

			return
		}

		req.Body = io.NopCloser(bytes.NewReader(body))
		_ = json.Unmarshal(body, &message)
	}

	wire := retirementWire{
		httpMethod: req.Method, rpcMethod: message.Method, owner: retirementOwner(req.Header.Get("Authorization")),
		session: req.Header.Get("Mcp-Session-Id"), op: message.Params.Arguments.Op,
	}

	p.mu.Lock()

	p.methods[wire.httpMethod+" "+wire.rpcMethod]++
	if session := p.sessions[wire.session]; wire.session != "" && session != nil && session.owner != wire.owner {
		p.leaks = append(p.leaks, fmt.Sprintf("%s used %s's session %s", wire.owner, session.owner, wire.session))
	}
	p.mu.Unlock()

	if req.Method == http.MethodGet && !p.sse {
		http.Error(w, "standalone SSE disabled", http.StatusMethodNotAllowed)

		return
	}

	if fault := p.fault.Load(); fault != nil {
		if status := (*fault)(req, wire); status != 0 {
			p.mu.Lock()
			p.injected++
			p.mu.Unlock()
			http.Error(w, "injected fault", status)

			return
		}
	}

	listen := wire.rpcMethod == registrySubscriptionMethod
	if listen {
		p.mu.Lock()
		p.listens++
		p.peakListens = max(p.peakListens, p.listens)
		p.mu.Unlock()
	}

	recorder := &retirementResponse{ResponseWriter: w, header: func(code int, header http.Header) {
		p.mu.Lock()
		defer p.mu.Unlock()

		p.respondedLocked(wire, code, header.Get("Mcp-Session-Id"))
	}}
	p.handler.ServeHTTP(recorder, req)

	if listen {
		p.mu.Lock()
		p.listens--
		p.mu.Unlock()
	}
}

// respondedLocked records a response before the client can receive it, so a
// client that reacts to it (reusing a session ID, or returning from a DELETE)
// always finds it recorded.
func (p *retirementPeer) respondedLocked(wire retirementWire, code int, id string) {
	if code == http.StatusOK && ((p.legacy && wire.rpcMethod == retirementInitializeMethod) || (!p.legacy && wire.rpcMethod == registryBenchmarkDiscoverMethod)) {
		p.handshakes[wire.owner]++
	}

	if _, seen := p.sessions[id]; wire.rpcMethod == retirementInitializeMethod && id != "" && code == http.StatusOK && !seen {
		p.sessions[id] = &retirementServerSession{owner: wire.owner, open: true}
		p.liveSessions++
		p.peakSessions = max(p.peakSessions, p.liveSessions)
	}

	if wire.httpMethod == http.MethodDelete {
		p.deletes[wire.session]++
		p.closeSessionLocked(wire.session)
	}
}

func retirementToolName(owner string) string { return "whoami_" + owner }

func (p *retirementPeer) server(owner string) *sdkmcp.Server {
	if stored, ok := p.servers.Load(owner); ok {
		server, _ := stored.(*sdkmcp.Server)

		return server
	}

	opts := &sdkmcp.ServerOptions{HasTools: true}
	if p.legacy {
		opts.SupportedProtocolVersions = []string{"2025-06-18"}
	}

	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "retire-" + owner, Version: "1"}, opts)
	server.AddTool(&sdkmcp.Tool{
		Name: retirementToolName(owner), Description: "identity of " + owner,
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{
			"op":             map[string]any{"type": "string"},
			"owner_" + owner: map[string]any{"type": "string", "const": owner},
		}},
	}, func(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
		var args struct {
			Op string `json:"op"`
		}

		_ = json.Unmarshal(req.Params.Arguments, &args)
		bearer := req.Extra.Header.Get("Authorization")
		execution := retirementExecution{op: args.Op, owner: retirementOwner(bearer), server: owner, bearer: bearer, session: req.Session.ID()}

		p.mu.Lock()

		p.executions[args.Op] = append(p.executions[args.Op], execution)
		if execution.owner != owner {
			p.leaks = append(p.leaks, fmt.Sprintf("%s executed on %s's server", execution.owner, owner))
		}

		p.executing++
		p.peakExecuting = max(p.peakExecuting, p.executing)
		p.mu.Unlock()

		defer func() {
			p.mu.Lock()
			p.executing--
			p.mu.Unlock()
		}()

		if gate := p.gate.Load(); gate != nil {
			if err := (*gate)(ctx, execution); err != nil {
				return nil, err
			}
		}

		text := owner + "|" + bearer + "|" + args.Op

		return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: text}}, IsError: strings.HasPrefix(args.Op, "is-error")}, nil
	})

	actual, _ := p.servers.LoadOrStore(owner, server)
	server, _ = actual.(*sdkmcp.Server)

	return server
}

func (p *retirementPeer) closeSessionLocked(id string) bool {
	session := p.sessions[id]
	if session == nil || !session.open {
		return false
	}

	session.open = false
	p.liveSessions--

	return true
}

// terminate closes the owner's server sessions, as a server restart or
// session expiry would, without involving the client.
func (p *retirementPeer) terminate(owner string) int {
	return p.terminateMatching(owner, func(string) bool { return true })
}

// terminateSession closes one server session by its wire ID.
func (p *retirementPeer) terminateSession(owner, id string) bool {
	return p.terminateMatching(owner, func(candidate string) bool { return candidate == id }) == 1
}

// terminateMatching kills each matching open session once. A closing go-sdk
// session stays in Sessions until its in-flight requests return, so
// concurrent terminations can all see it; only the one that claims its open
// record counts the kill and closes it.
func (p *retirementPeer) terminateMatching(owner string, match func(string) bool) int {
	var sessions []*sdkmcp.ServerSession

	for session := range p.server(owner).Sessions() {
		if match(session.ID()) {
			sessions = append(sessions, session)
		}
	}

	if p.snapshotted != nil {
		p.snapshotted()
	}

	killed := 0

	for _, session := range sessions {
		p.mu.Lock()

		claimed := p.closeSessionLocked(session.ID())
		if claimed {
			p.kills++
		}

		p.mu.Unlock()

		if claimed {
			_ = session.Close()
			killed++
		}
	}

	return killed
}

func (p *retirementPeer) setFault(fault func(*http.Request, retirementWire) int) {
	p.fault.Store(&fault)
}

func (p *retirementPeer) setGate(gate func(context.Context, retirementExecution) error) {
	p.gate.Store(&gate)
}

func (p *retirementPeer) failCalls(op string, status int) {
	p.setFault(func(_ *http.Request, wire retirementWire) int {
		if wire.rpcMethod == registryAgentCallMethod && wire.op == op {
			return status
		}

		return 0
	})
}

func (p *retirementPeer) executionsOf(op string) []retirementExecution {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]retirementExecution(nil), p.executions[op]...)
}

func (p *retirementPeer) handshakesOf(owner string) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.handshakes[owner]
}

func (p *retirementPeer) leaked() []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]string(nil), p.leaks...)
}

type retirementPeerStats struct {
	LiveSessions  int `json:"live_sessions"`
	PeakSessions  int `json:"peak_sessions"`
	Listens       int `json:"open_listen_streams"`
	PeakListens   int `json:"peak_listen_streams"`
	Executing     int `json:"executing"`
	PeakExecuting int `json:"peak_executing"`
	Handshakes    int `json:"handshakes"`
	Sessions      int `json:"sessions_created"`
	Deletes       int `json:"deletes"`
	Kills         int `json:"server_terminations"`
	Injected      int `json:"injected_http_faults"`
	Executions    int `json:"business_executions"`
	Duplicates    int `json:"duplicate_executions"`
	Leaks         int `json:"identity_leaks"`
}

func (p *retirementPeer) stats() retirementPeerStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	stats := retirementPeerStats{
		LiveSessions: p.liveSessions, PeakSessions: p.peakSessions, Listens: p.listens, PeakListens: p.peakListens,
		Executing: p.executing, PeakExecuting: p.peakExecuting, Sessions: len(p.sessions), Kills: p.kills,
		Injected: p.injected, Leaks: len(p.leaks),
	}

	for _, count := range p.handshakes {
		stats.Handshakes += count
	}

	for _, count := range p.deletes {
		stats.Deletes += count
	}

	for _, executions := range p.executions {
		stats.Executions += len(executions)
		if len(executions) > 1 {
			stats.Duplicates += len(executions) - 1
		}
	}

	return stats
}

func (p *retirementPeer) deletesOf(session string) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.deletes[session]
}

// retirementBarrier blocks gated executions until n of them are waiting at
// the server at once, proving that they overlap; later executions pass.
type retirementBarrier struct {
	n       int
	mu      sync.Mutex
	waiting int
	open    bool
	all     chan struct{}
}

func newRetirementBarrier(n int) *retirementBarrier {
	return &retirementBarrier{n: n, all: make(chan struct{})}
}

func (b *retirementBarrier) wait(ctx context.Context) error {
	b.mu.Lock()
	if b.open {
		b.mu.Unlock()

		return nil
	}

	b.waiting++
	if b.waiting == b.n {
		b.open = true
		close(b.all)
		b.mu.Unlock()

		return nil
	}
	b.mu.Unlock()

	select {
	case <-b.all:
		return nil
	case <-ctx.Done():
		b.mu.Lock()
		if !b.open {
			b.waiting--
		}
		b.mu.Unlock()

		return ctx.Err()
	}
}

// opened reports whether n executions were ever waiting at once.
func (b *retirementBarrier) opened() bool {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.open
}

func newRetirementRegistry(tb testing.TB, peer *retirementPeer, opts ...RegistryOption) *Registry {
	tb.Helper()

	registry, err := NewRegistry("retire", peer.url, append([]RegistryOption{WithCredentials(retirementCredentials)}, opts...)...)
	require.NoError(tb, err)
	tb.Cleanup(func() { assert.NoError(tb, registry.Close()) })

	return registry
}

func retirementArgs(op string) json.RawMessage {
	return json.RawMessage(fmt.Sprintf(`{"op":%q}`, op))
}

func retirementTool(ctx context.Context, tb testing.TB, captured tool.Registry, owner string) tool.Tool {
	tb.Helper()

	found, err := captured.Get(ctx, "retire__"+retirementToolName(owner))
	require.NoError(tb, err)

	return found
}

func retirementReceive[T any](ctx context.Context, tb testing.TB, ch <-chan T, what string) T {
	tb.Helper()

	select {
	case value := <-ch:
		return value
	case <-ctx.Done():
		tb.Fatalf("timed out waiting for %s", what)

		var zero T

		return zero
	}
}

func TestRegistryRetiredSessionIsNotReusedWhileLeased(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)
			registry := newRetirementRegistry(t, peer)

			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
			defer cancel()

			alice := retirementCaller(ctx, "alice", "one")
			captured, err := registry.Snapshot(alice)
			require.NoError(t, err)

			call := retirementTool(alice, t, captured, "alice")

			registry.mu.Lock()
			original := registry.entries["alice"]
			registry.mu.Unlock()
			require.NotNil(t, original)

			hold := newRetirementHold()
			held := make(chan error, 1)

			go func() {
				_, err := call.Execute(retirementHeldCaller(alice, "alice", "one", hold), retirementArgs("held"))
				held <- err
			}()

			retirementReceive(ctx, t, hold.reached, "the held call to take its lease")

			if legacy {
				require.Equal(t, 1, peer.terminate("alice"))

				_, err = call.Execute(alice, retirementArgs("lost"))
				require.ErrorIs(t, err, sdkmcp.ErrSessionMissing)
			} else {
				peer.failCalls("poison", http.StatusBadRequest)

				_, err = call.Execute(alice, retirementArgs("poison"))
				require.Error(t, err)

				_, err = call.Execute(alice, retirementArgs("lost"))
				require.ErrorIs(t, err, sdkmcp.ErrConnectionClosed)
			}

			result, err := call.Execute(alice, retirementArgs("replacement"))
			require.NoError(t, err, "a new caller must reconnect instead of receiving the retired session another call still leases")
			assert.Contains(t, string(result), "alice|Bearer alice:one|replacement")
			assert.Equal(t, 2, peer.handshakesOf("alice"), "the replacement performs exactly one new handshake")

			close(hold.release)
			retirementHeldResult(t, legacy, retirementReceive(ctx, t, held, "the held call to unwind"))

			registry.mu.Lock()
			current := registry.entries["alice"]
			resident := len(registry.entries)
			registry.mu.Unlock()
			require.NotNil(t, current, "releasing the old lease must not remove the replacement")
			assert.NotSame(t, original, current)
			assert.Equal(t, 1, resident)

			again, err := call.Execute(alice, retirementArgs("after"))
			require.NoError(t, err, "the replacement must survive the old entry's last release")
			assert.Contains(t, string(again), "alice|Bearer alice:one|after")
			assert.Equal(t, 2, peer.handshakesOf("alice"))

			assert.Empty(t, peer.executionsOf("lost"), "the failed call never reached the tool")
			assert.Len(t, peer.executionsOf("replacement"), 1)
			assert.Len(t, peer.executionsOf("after"), 1)

			if legacy {
				assert.Empty(t, peer.executionsOf("held"), "the held call must not be replayed on the replacement")
			} else {
				assert.LessOrEqual(t, len(peer.executionsOf("held")), 1, "the held request is sent at most once and never replayed")
				assert.Empty(t, peer.executionsOf("poison"))
			}

			assert.Empty(t, peer.leaked())
		})
	}
}

// retirementHeldResult checks a call that held its lease across the loss. A
// legacy request carries the dead session ID and is refused. A modern request
// was already retired by the failed connection's reader, but its single HTTP
// send may still reach the stateless server once.
func retirementHeldResult(tb testing.TB, legacy bool, err error) {
	tb.Helper()

	if legacy {
		require.Error(tb, err, "the held call belongs to the lost session")
	}
}

// retirementLoseSession makes the server lose owner's current session and
// returns once a probe call observed the loss with a retiring SDK error.
func retirementLoseSession(caller context.Context, tb testing.TB, peer *retirementPeer, call tool.Tool, owner, probe string) {
	tb.Helper()

	if peer.legacy && !peer.sse {
		require.Positive(tb, peer.terminate(owner))

		_, err := call.Execute(caller, retirementArgs(probe))
		require.ErrorIs(tb, err, sdkmcp.ErrSessionMissing)

		return
	}

	if peer.legacy {
		// The standalone SSE stream races the probe: its failed reconnect can
		// fail the connection first, so the probe that observes it may report
		// a non-retiring stream error. The test, not the registry, retries.
		require.Positive(tb, peer.terminate(owner))

		for attempt := range 3 {
			_, err := call.Execute(caller, retirementArgs(fmt.Sprintf("%s-%d", probe, attempt)))
			require.Error(tb, err)

			if errors.Is(err, sdkmcp.ErrSessionMissing) || errors.Is(err, sdkmcp.ErrConnectionClosed) {
				return
			}
		}

		tb.Fatal("no probe observed a retiring error")
	}

	peer.failCalls(probe+"-poison", http.StatusBadRequest)

	_, err := call.Execute(caller, retirementArgs(probe+"-poison"))
	require.Error(tb, err)
	require.NotErrorIs(tb, err, sdkmcp.ErrConnectionClosed, "a stream failure alone is not a retiring error")

	_, err = call.Execute(caller, retirementArgs(probe))
	require.ErrorIs(tb, err, sdkmcp.ErrConnectionClosed)
}

func retirementStartHeld(ctx context.Context, tb testing.TB, call tool.Tool, owner, op string) (*retirementHold, <-chan error) {
	tb.Helper()

	hold := newRetirementHold()
	done := make(chan error, 1)

	go func() {
		_, err := call.Execute(retirementHeldCaller(ctx, owner, "one", hold), retirementArgs(op))
		done <- err
	}()

	retirementReceive(ctx, tb, hold.reached, op+" to take its lease")

	return hold, done
}

func retirementEntry(r *Registry, owner string) (*registryEntry, int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry := r.entries[owner]
	if entry == nil {
		return nil, 0
	}

	return entry, entry.leases
}

func retirementAwait(ctx context.Context, tb testing.TB, what string, done func() bool) {
	tb.Helper()

	ticker := time.NewTicker(time.Millisecond)
	defer ticker.Stop()

	for !done() {
		select {
		case <-ticker.C:
		case <-ctx.Done():
			tb.Fatalf("timed out waiting for %s", what)
		}
	}
}

// TestRegistryRetirementSamePartitionCallers holds sixteen calls on one
// session across its loss, then proves sixteen new callers overlap at the
// server on exactly one replacement while the old calls are still held.
func TestRegistryRetirementSamePartitionCallers(t *testing.T) {
	t.Parallel()

	const held, fresh = 16, 16

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)
			registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()

			alice := retirementCaller(ctx, "alice", "one")
			captured, err := registry.Snapshot(alice)
			require.NoError(t, err)

			call := retirementTool(alice, t, captured, "alice")
			original, _ := retirementEntry(registry, "alice")

			holds := make([]*retirementHold, held)
			heldDone := make([]<-chan error, held)

			for i := range held {
				holds[i], heldDone[i] = retirementStartHeld(ctx, t, call, "alice", fmt.Sprintf("held-%d", i))
			}

			_, leases := retirementEntry(registry, "alice")
			require.Equal(t, held, leases, "every held call leases the original session")

			retirementLoseSession(alice, t, peer, call, "alice", "lost")

			barrier := newRetirementBarrier(fresh)

			peer.setGate(func(ctx context.Context, execution retirementExecution) error {
				if strings.HasPrefix(execution.op, "fresh-") {
					return barrier.wait(ctx)
				}

				return nil
			})

			results := make(chan error, fresh)
			for i := range fresh {
				go func() {
					op := fmt.Sprintf("fresh-%d", i)

					out, err := call.Execute(alice, retirementArgs(op))
					if err == nil && !strings.Contains(string(out), "alice|Bearer alice:one|"+op) {
						err = fmt.Errorf("unexpected result %s", out)
					}

					results <- err
				}()
			}

			for range fresh {
				require.NoError(t, retirementReceive(ctx, t, results, "a fresh call on the replacement"))
			}

			stats := peer.stats()
			assert.Equal(t, fresh, stats.PeakExecuting, "the fresh calls overlapped at the server")
			assert.Equal(t, 2, peer.handshakesOf("alice"), "concurrent callers share one replacement")

			replacement, _ := retirementEntry(registry, "alice")
			require.NotNil(t, replacement)
			assert.NotSame(t, original, replacement)

			for i := range held {
				close(holds[i].release)
				retirementHeldResult(t, legacy, retirementReceive(ctx, t, heldDone[i], "a held call"))
			}

			current, leases := retirementEntry(registry, "alice")
			assert.Same(t, replacement, current, "old leases must not detach the replacement")
			assert.Zero(t, leases)

			for i := range held {
				executions := peer.executionsOf(fmt.Sprintf("held-%d", i))
				if legacy {
					assert.Empty(t, executions)
				} else {
					assert.LessOrEqual(t, len(executions), 1)
				}
			}

			for i := range fresh {
				assert.Len(t, peer.executionsOf(fmt.Sprintf("fresh-%d", i)), 1)
			}

			assert.Zero(t, peer.stats().Duplicates)
			assert.Empty(t, peer.leaked())
		})
	}
}

// TestRegistryRetirementOrdinaryFailuresKeepSession checks that errors which leave the
// SDK connection usable never retire it or trigger a reconnect.
func TestRegistryRetirementOrdinaryFailuresKeepSession(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)
			registry := newRetirementRegistry(t, peer)

			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()

			alice := retirementCaller(ctx, "alice", "one")
			captured, err := registry.Snapshot(alice)
			require.NoError(t, err)

			call := retirementTool(alice, t, captured, "alice")
			original, _ := retirementEntry(registry, "alice")
			sessionID := original.session.ID()

			slow := make(chan struct{})
			release := make(chan struct{})

			t.Cleanup(func() { close(release) })

			peer.setGate(func(ctx context.Context, execution retirementExecution) error {
				switch execution.op {
				case "handler-error":
					return errors.New("tool handler failed")
				case "canceled", "deadline":
					slow <- struct{}{}

					select {
					case <-ctx.Done():
					case <-release:
					}

					return errors.New("abandoned")
				}

				return nil
			})
			peer.failCalls("unavailable", http.StatusServiceUnavailable)

			cases := []struct {
				op    string
				check func(t *testing.T, err error)
				ctx   func() (context.Context, context.CancelFunc)
			}{
				{op: "is-error-result", check: func(t *testing.T, err error) {
					t.Helper()

					var result *tool.ResultError
					require.ErrorAs(t, err, &result)
				}},
				{op: "handler-error", check: func(t *testing.T, err error) {
					t.Helper()
					require.Error(t, err)
				}},
				{op: "unavailable", check: func(t *testing.T, err error) {
					t.Helper()
					require.Error(t, err)
					require.NotErrorIs(t, err, sdkmcp.ErrConnectionClosed)
				}},
				{op: "canceled", ctx: func() (context.Context, context.CancelFunc) {
					ctx, cancel := context.WithCancel(alice)

					go func() { <-slow; cancel() }()

					return ctx, cancel
				}, check: func(t *testing.T, err error) {
					t.Helper()
					require.ErrorIs(t, err, context.Canceled)
				}},
				{op: "deadline", ctx: func() (context.Context, context.CancelFunc) {
					ctx, cancel := context.WithTimeout(alice, 100*time.Millisecond)

					go func() { <-slow }()

					return ctx, cancel
				}, check: func(t *testing.T, err error) {
					t.Helper()
					require.ErrorIs(t, err, context.DeadlineExceeded)
				}},
			}

			for i, tc := range cases {
				callCtx, stop := alice, context.CancelFunc(func() {})
				if tc.ctx != nil {
					callCtx, stop = tc.ctx()
				}

				_, err := call.Execute(callCtx, retirementArgs(tc.op))

				stop()
				tc.check(t, err)
				require.NotErrorIs(t, err, sdkmcp.ErrSessionMissing, tc.op)

				out, err := call.Execute(alice, retirementArgs(fmt.Sprintf("ok-%d", i)))
				require.NoError(t, err, "%s must leave the session usable", tc.op)
				assert.Contains(t, string(out), fmt.Sprintf("alice|Bearer alice:one|ok-%d", i))

				current, _ := retirementEntry(registry, "alice")
				require.Same(t, original, current, "%s must not retire the session", tc.op)
			}

			assert.Equal(t, 1, peer.handshakesOf("alice"))
			assert.Zero(t, peer.deletesOf(sessionID))
			assert.Zero(t, peer.stats().Kills)
			assert.Empty(t, peer.leaked())
		})
	}
}

// TestRegistryRetirementHoldsCapacity uses one connection slot: the
// retired session keeps it until its last lease ends, so other callers wait
// cancellably and proceed once it drains.
func TestRegistryRetirementHoldsCapacity(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)
			registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(1))

			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()

			alice := retirementCaller(ctx, "alice", "one")
			captured, err := registry.Snapshot(alice)
			require.NoError(t, err)

			call := retirementTool(alice, t, captured, "alice")
			hold, held := retirementStartHeld(ctx, t, call, "alice", "held")

			retirementLoseSession(alice, t, peer, call, "alice", "lost")

			short, stop := context.WithTimeout(alice, 100*time.Millisecond)
			_, err = call.Execute(short, retirementArgs("blocked"))

			stop()
			require.ErrorIs(t, err, context.DeadlineExceeded, "the replacement must wait for the retired session's slot")

			bob, stopBob := context.WithTimeout(retirementCaller(ctx, "bob", "one"), 100*time.Millisecond)
			_, err = registry.Snapshot(bob)

			stopBob()
			require.ErrorIs(t, err, context.DeadlineExceeded, "another partition cannot exceed the bound either")

			assert.Equal(t, 1, peer.handshakesOf("alice"))
			assert.Zero(t, peer.handshakesOf("bob"))

			waiter := make(chan error, 1)

			go func() {
				out, err := call.Execute(alice, retirementArgs("waiter"))
				if err == nil && !strings.Contains(string(out), "alice|Bearer alice:one|waiter") {
					err = fmt.Errorf("unexpected result %s", out)
				}

				waiter <- err
			}()

			close(hold.release)
			retirementHeldResult(t, legacy, retirementReceive(ctx, t, held, "the held call"))
			require.NoError(t, retirementReceive(ctx, t, waiter, "the waiting caller"))
			assert.Equal(t, 2, peer.handshakesOf("alice"))

			_, err = registry.Snapshot(retirementCaller(ctx, "bob", "one"))
			require.NoError(t, err, "bob evicts the idle replacement once capacity is free")

			if legacy {
				assert.Equal(t, 1, peer.stats().PeakSessions, "never more than one server session")
			}

			assert.Empty(t, peer.executionsOf("blocked"))
			assert.Zero(t, peer.stats().Duplicates)
			assert.Empty(t, peer.leaked())
		})
	}
}

// TestRegistryRetirementChurn repeatedly loses sessions under a held lease
// with two partitions sharing two slots.
func TestRegistryRetirementChurn(t *testing.T) {
	t.Parallel()

	const rounds = 24

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, true)
			registry := newRetirementRegistry(t, peer, WithRegistryMaxConnections(2))

			ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
			defer cancel()

			owners := []string{"alice", "bob"}
			calls := make(map[string]tool.Tool)

			for _, owner := range owners {
				caller := retirementCaller(ctx, owner, "one")
				captured, err := registry.Snapshot(caller)
				require.NoError(t, err)

				calls[owner] = retirementTool(caller, t, captured, owner)
			}

			for round := range rounds {
				owner := owners[round%2]
				caller := retirementCaller(ctx, owner, "one")
				hold, held := retirementStartHeld(ctx, t, calls[owner], owner, fmt.Sprintf("held-%d", round))

				retirementLoseSession(caller, t, peer, calls[owner], owner, fmt.Sprintf("lost-%d", round))

				// The retired session still holds a slot, so each new session
				// below first evicts the other, idle partition.
				op := fmt.Sprintf("replacement-%d", round)
				out, err := calls[owner].Execute(caller, retirementArgs(op))
				require.NoError(t, err, "round %d", round)
				assert.Contains(t, string(out), owner+"|Bearer "+owner+":one|"+op)

				other := owners[(round+1)%2]
				otherOp := fmt.Sprintf("other-%d", round)
				out, err = calls[other].Execute(retirementCaller(ctx, other, "one"), retirementArgs(otherOp))
				require.NoError(t, err, "round %d", round)
				assert.Contains(t, string(out), other+"|Bearer "+other+":one|"+otherOp)

				close(hold.release)
				retirementHeldResult(t, legacy, retirementReceive(ctx, t, held, "the held call"))
			}

			stats := peer.stats()
			assert.Equal(t, 2+2*rounds, stats.Handshakes, "one replacement and one eviction reconnect per round, nothing more")

			if legacy {
				assert.Equal(t, 2, stats.PeakSessions, "two slots used, never exceeded")
			}

			assert.Zero(t, peer.stats().Duplicates)
			assert.Empty(t, peer.leaked())
		})
	}
}

// TestRegistryRetirementCanceledConnectorDoesNotFailWaiters cancels the caller whose
// handshake other callers joined. A waiter's own cancellation still ends only
// that waiter, but a live waiter must acquire a fresh connection rather than
// inherit another caller's cancellation.
func TestRegistryRetirementCanceledConnectorDoesNotFailWaiters(t *testing.T) {
	t.Parallel()

	for _, legacy := range []bool{true, false} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			t.Parallel()

			peer := newRetirementPeer(t, legacy, false)

			handshake := registryBenchmarkDiscoverMethod
			if legacy {
				handshake = retirementInitializeMethod
			}

			arrived := make(chan struct{})

			var attempts atomic.Int64

			peer.setFault(func(req *http.Request, wire retirementWire) int {
				if wire.rpcMethod != handshake || attempts.Add(1) != 1 {
					return 0
				}

				close(arrived)
				<-req.Context().Done()

				return http.StatusServiceUnavailable
			})

			registry := newRetirementRegistry(t, peer)

			ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
			defer cancel()

			first, cancelFirst := context.WithCancel(retirementCaller(ctx, "alice", "one"))
			firstDone := make(chan error, 1)

			go func() {
				_, err := registry.Snapshot(first)
				firstDone <- err
			}()

			retirementReceive(ctx, t, arrived, "the first handshake")

			_, baseline := retirementEntry(registry, "alice")

			joined := func(caller context.Context) <-chan error {
				done := make(chan error, 1)

				go func() {
					definitions, err := registry.List(caller)
					if err == nil && (len(definitions) != 1 || definitions[0].Name != "retire__whoami_alice") {
						err = fmt.Errorf("unexpected catalog %v", registryNames(definitions))
					}

					done <- err
				}()

				return done
			}

			impatient, cancelImpatient := context.WithCancel(retirementCaller(ctx, "alice", "three"))
			impatientDone := joined(impatient)

			retirementAwait(ctx, t, "the impatient caller to join", func() bool {
				_, leases := retirementEntry(registry, "alice")
				return leases == baseline+1
			})

			cancelImpatient()
			require.ErrorIs(t, retirementReceive(ctx, t, impatientDone, "the impatient caller"), context.Canceled, "a waiter's own cancellation ends it")

			live := joined(retirementCaller(ctx, "alice", "two"))
			retirementAwait(ctx, t, "the live caller to join", func() bool {
				_, leases := retirementEntry(registry, "alice")
				return leases == baseline+1
			})

			cancelFirst()
			require.ErrorIs(t, retirementReceive(ctx, t, firstDone, "the canceled connector"), context.Canceled)
			require.NoError(t, retirementReceive(ctx, t, live, "the live waiter"), "another caller's cancellation must not fail a live waiter")

			assert.EqualValues(t, 2, attempts.Load(), "the live waiter reconnected exactly once")
			assert.Equal(t, 1, peer.handshakesOf("alice"))

			entry, leases := retirementEntry(registry, "alice")
			require.NotNil(t, entry)
			assert.Zero(t, leases)
			assert.Empty(t, peer.leaked())
		})
	}
}

// TestRegistryRetirementPeerCountsEachLostSessionOnce checks the fixture that
// the exact-replacement assertions rely on. Concurrent terminations that all
// observe one session count one loss, and a session and its handshake are
// recorded before the initialize response reaches the client.
func TestRegistryRetirementPeerCountsEachLostSessionOnce(t *testing.T) {
	t.Parallel()

	const terminators = 16

	peer := newRetirementPeer(t, true, false)
	registry := newRetirementRegistry(t, peer)

	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()

	alice := retirementCaller(ctx, "alice", "one")
	captured, err := registry.Snapshot(alice)
	require.NoError(t, err)

	stats := peer.stats()
	require.Equal(t, 1, stats.Handshakes)
	require.Equal(t, 1, stats.Sessions)
	require.Equal(t, 1, stats.LiveSessions)

	entry, _ := retirementEntry(registry, "alice")
	id := entry.session.ID()

	var snapshots sync.WaitGroup

	snapshots.Add(terminators)

	peer.snapshotted = func() {
		snapshots.Done()
		snapshots.Wait()
	}

	claims := make(chan bool, terminators)
	for range terminators {
		go func() { claims <- peer.terminateSession("alice", id) }()
	}

	won := 0

	for range terminators {
		if retirementReceive(ctx, t, claims, "a termination") {
			won++
		}
	}

	peer.snapshotted = nil

	require.Equal(t, 1, won, "every terminator saw the session; exactly one claims it")

	stats = peer.stats()
	assert.Equal(t, 1, stats.Kills)
	assert.Zero(t, stats.LiveSessions)
	assert.Zero(t, peer.terminate("alice"), "a closing session that lingers in Sessions is not lost again")

	call := retirementTool(alice, t, captured, "alice")
	_, err = call.Execute(alice, retirementArgs("lost"))
	require.ErrorIs(t, err, sdkmcp.ErrSessionMissing)

	out, err := call.Execute(alice, retirementArgs("replacement"))
	require.NoError(t, err)
	assert.Contains(t, string(out), "alice|Bearer alice:one|replacement")

	stats = peer.stats()
	assert.Equal(t, 2, stats.Handshakes)
	assert.Equal(t, 2, stats.Sessions)
	assert.Equal(t, 1, stats.Kills)
	assert.Equal(t, 1, stats.LiveSessions)
	assert.Equal(t, stats.Kills+1, stats.Handshakes)
	assert.Zero(t, peer.deletesOf(id), "a session the server lost needs no DELETE")
	assert.Empty(t, peer.leaked())
}

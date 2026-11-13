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
	"maps"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// cleanupBarrier bounds every channel barrier; it is far above any expected wait
// and is not used to decide ordering.
const cleanupBarrier = 15 * time.Second

const cleanupToolName = "remote__whoami"

type cleanupCallerKey struct{}

func cleanupCaller(ctx context.Context, partition string) context.Context {
	return context.WithValue(ctx, cleanupCallerKey{}, partition)
}

func cleanupCredentials(ctx context.Context) (Credentials, error) {
	partition, ok := ctx.Value(cleanupCallerKey{}).(string)
	if !ok {
		return Credentials{}, errors.New("cleanup caller missing")
	}

	return Credentials{PartitionKey: partition, Headers: http.Header{"Authorization": {"Bearer " + partition}}}, nil
}

func cleanupRequest() *llm.ToolRequestPart {
	return &llm.ToolRequestPart{ID: cleanupToolName, Name: cleanupToolName, Arguments: json.RawMessage(`{}`)}
}

type cleanupPeerStats struct {
	Opens             int `json:"opens"`
	Lists             int `json:"lists"`
	Calls             int `json:"calls"`
	CallInflight      int `json:"call_inflight"`
	CallPeak          int `json:"call_peak"`
	CallsDuringDelete int `json:"calls_during_delete"`
	Listens           int `json:"listens"`
	ListenInflight    int `json:"listen_inflight"`
	ListenPeak        int `json:"listen_peak"`
	Cancels           int `json:"cancel_notifications"`
	CancelInflight    int `json:"cancel_inflight"`
	DeleteStarted     int `json:"delete_started"`
	DeleteDone        int `json:"delete_done"`
	DeleteInflight    int `json:"delete_inflight"`
	DeletePeak        int `json:"delete_peak"`
	Live              int `json:"live_sessions"`
	LivePeak          int `json:"live_session_peak"`
	Conns             int `json:"conns"`
	ConnPeak          int `json:"conn_peak"`
	Sessions          int `json:"sessions"`
}

// cleanupPeer is a loopback MCP server built on the SDK's streamable handler.
// It records per-partition opens, legacy session owners and DELETEs, and can
// hold session-termination DELETEs at the peer regardless of client deadlines.
// Live counts legacy sessions from the initialize response that assigned them
// until their DELETE passes the peer's hold and pacing gates. The registry
// frees a slot only after that DELETE's response, so Live never exceeds the
// registry's occupied capacity unless go-sdk gives up on a held DELETE.
type cleanupPeer struct {
	url     string
	legacy  bool
	handler http.Handler

	mu         sync.Mutex
	changed    chan struct{}
	gate       chan struct{}
	pace       chan struct{}
	paceOff    chan struct{}
	holdCancel bool
	failDelete bool
	callGate   chan struct{}
	live       map[string]bool
	used       map[string]int
	stats      cleanupPeerStats
	opens      map[string]int
	owners     map[string]string
	deletes    map[string]int
	breakCalls map[string]int
	failStream map[string]bool
	failOpen   map[string]bool
	mismatches []string
}

func newCleanupPeer(t *testing.T, legacy bool) *cleanupPeer {
	t.Helper()

	opts := &sdkmcp.ServerOptions{HasTools: true}
	if legacy {
		opts.SupportedProtocolVersions = []string{"2025-06-18"}
	}

	peer := &cleanupPeer{
		legacy: legacy, changed: make(chan struct{}), paceOff: make(chan struct{}), live: make(map[string]bool), used: make(map[string]int),
		opens: make(map[string]int), owners: make(map[string]string), deletes: make(map[string]int),
		breakCalls: make(map[string]int), failStream: make(map[string]bool), failOpen: make(map[string]bool),
	}

	server := sdkmcp.NewServer(&sdkmcp.Implementation{Name: "cleanup", Version: "1"}, opts)
	server.AddTool(&sdkmcp.Tool{Name: "whoami", InputSchema: map[string]any{"type": "object"}}, peer.whoami)
	peer.handler = sdkmcp.NewStreamableHTTPHandler(func(*http.Request) *sdkmcp.Server { return server }, &sdkmcp.StreamableHTTPOptions{Stateless: !legacy, JSONResponse: true})

	httpServer := httptest.NewUnstartedServer(peer)
	httpServer.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		peer.mu.Lock()
		defer peer.mu.Unlock()

		switch state { //nolint:exhaustive // Only socket lifetime transitions matter.
		case http.StateNew:
			peer.stats.Conns++
			peer.stats.ConnPeak = max(peer.stats.ConnPeak, peer.stats.Conns)
		case http.StateClosed, http.StateHijacked:
			peer.stats.Conns--
		}

		peer.signalLocked()
	}
	httpServer.Start()
	peer.url = httpServer.URL
	t.Cleanup(httpServer.Close)
	t.Cleanup(peer.release)

	return peer
}

func (p *cleanupPeer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	partition := strings.TrimPrefix(req.Header.Get("Authorization"), "Bearer ")
	session := req.Header.Get("Mcp-Session-Id")

	p.mu.Lock()
	if owner, ok := p.owners[session]; session != "" && ok && owner != partition {
		p.mismatches = append(p.mismatches, fmt.Sprintf("%s %s used by %s", req.Method, owner, partition))
	}

	if session != "" {
		p.used[session]++
	}
	p.mu.Unlock()

	var method string

	switch req.Method {
	case http.MethodDelete:
		p.serveDelete(w, req, session)

		return
	case http.MethodGet:
		if p.serveStream(w, partition) {
			return
		}
	case http.MethodPost:
		var ok bool
		if method, ok = p.recordPost(w, req, partition); !ok {
			return
		}
	}

	if method == "subscriptions/listen" {
		p.mu.Lock()
		p.stats.Listens++
		p.stats.ListenInflight++
		p.stats.ListenPeak = max(p.stats.ListenPeak, p.stats.ListenInflight)
		p.signalLocked()
		p.mu.Unlock()

		defer func() {
			p.mu.Lock()
			p.stats.ListenInflight--
			p.signalLocked()
			p.mu.Unlock()
		}()
	}

	p.handler.ServeHTTP(&cleanupResponseWriter{ResponseWriter: w, header: func(header http.Header) {
		if id := header.Get("Mcp-Session-Id"); id != "" && session == "" && req.Method == http.MethodPost {
			p.mu.Lock()
			if _, ok := p.owners[id]; !ok {
				p.owners[id] = partition
				p.live[id] = true
				p.stats.Live++
				p.stats.LivePeak = max(p.stats.LivePeak, p.stats.Live)
			}

			p.signalLocked()
			p.mu.Unlock()
		}
	}}, req)
}

func (p *cleanupPeer) signalLocked() {
	close(p.changed)
	p.changed = make(chan struct{})
}

// hold makes subsequent DELETEs block at the peer until release.
func (p *cleanupPeer) hold() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.gate == nil {
		p.gate = make(chan struct{})
	}
}

// release opens every gate; test cleanup always calls it before closing the
// registry or the server, so no handler stays blocked.
func (p *cleanupPeer) release() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.gate != nil {
		close(p.gate)
		p.gate = nil
	}

	if p.callGate != nil {
		close(p.callGate)
		p.callGate = nil
	}

	if p.pace != nil {
		close(p.paceOff)
		p.pace = nil
	}
}

// throttle makes each later DELETE wait for one token from the returned
// channel, or for release.
func (p *cleanupPeer) throttle() chan<- struct{} {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.pace = make(chan struct{})
	p.paceOff = make(chan struct{})

	return p.pace
}

// holdCalls makes subsequent tool calls block in the server handler until
// releaseCalls or release.
func (p *cleanupPeer) holdCalls() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.callGate == nil {
		p.callGate = make(chan struct{})
	}
}

func (p *cleanupPeer) releaseCalls() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.callGate != nil {
		close(p.callGate)
		p.callGate = nil
	}
}

func (p *cleanupPeer) whoami(ctx context.Context, req *sdkmcp.CallToolRequest) (*sdkmcp.CallToolResult, error) {
	p.mu.Lock()
	gate := p.callGate
	p.stats.CallInflight++
	p.stats.CallPeak = max(p.stats.CallPeak, p.stats.CallInflight)

	if p.stats.DeleteInflight > 0 {
		p.stats.CallsDuringDelete++
	}

	p.signalLocked()
	p.mu.Unlock()

	if gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
		}
	}

	p.mu.Lock()
	p.stats.CallInflight--
	p.signalLocked()
	p.mu.Unlock()

	return &sdkmcp.CallToolResult{Content: []sdkmcp.Content{&sdkmcp.TextContent{Text: "caller=" + req.Extra.Header.Get("Authorization")}}}, nil
}

func (p *cleanupPeer) snapshot() cleanupPeerStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	stats := p.stats
	stats.Sessions = len(p.owners)

	return stats
}

func (p *cleanupPeer) openCount(partition string) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return p.opens[partition]
}

func (p *cleanupPeer) sessionsOf(partition string) []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var sessions []string

	for session, owner := range p.owners {
		if owner == partition {
			sessions = append(sessions, session)
		}
	}

	return sessions
}

func (p *cleanupPeer) deleteCounts() map[string]int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return maps.Clone(p.deletes)
}

// orphans lists sessions the peer assigned that the client never referenced
// in any request, not even a DELETE: the client never learned their ids.
func (p *cleanupPeer) orphans() []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var orphans []string

	for session := range p.owners {
		if p.used[session] == 0 {
			orphans = append(orphans, session)
		}
	}

	return orphans
}

func (p *cleanupPeer) identityMismatches() []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]string(nil), p.mismatches...)
}

// await is a channel barrier on peer state, bounded by cleanupBarrier.
func (p *cleanupPeer) await(t *testing.T, what string, cond func(cleanupPeerStats) bool) cleanupPeerStats {
	t.Helper()

	timer := time.NewTimer(cleanupBarrier)
	defer timer.Stop()

	for {
		p.mu.Lock()
		stats := p.stats
		stats.Sessions = len(p.owners)
		changed := p.changed
		p.mu.Unlock()

		if cond(stats) {
			return stats
		}

		select {
		case <-changed:
		case <-timer.C:
			t.Fatalf("peer never reached %s: %+v", what, stats)
		}
	}
}

// breakCall answers partition's next tools/call with a malformed SSE event,
// which the SDK treats as a permanent connection failure. The session id is
// still live, so the SDK's Close must send a termination DELETE.
func (p *cleanupPeer) breakCall(partition string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	p.breakCalls[partition]++
}

func (p *cleanupPeer) recordPost(w http.ResponseWriter, req *http.Request, partition string) (string, bool) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)

		return "", false
	}

	req.Body = io.NopCloser(bytes.NewReader(body))

	var message struct {
		Method string `json:"method"`
	}

	_ = json.Unmarshal(body, &message)

	p.mu.Lock()
	defer p.mu.Unlock()

	// go-sdk opens a modern session with server/discover. A legacy open probes
	// server/discover first, then sends initialize, the only other request
	// without an Mcp-Session-Id.
	discover := message.Method == registryBenchmarkDiscoverMethod
	open := message.Method != "" && discover != (p.legacy && req.Header.Get("Mcp-Session-Id") == "")

	switch {
	case open:
		p.stats.Opens++
		p.opens[partition]++

		if p.failOpen[partition] {
			p.signalLocked()
			http.Error(w, "open refused", http.StatusInternalServerError)

			return "", false
		}
	case message.Method == registryParameterListMethod:
		p.stats.Lists++
	case message.Method == registryAgentCallMethod:
		p.stats.Calls++

		if p.breakCalls[partition] > 0 {
			p.breakCalls[partition]--
			p.signalLocked()
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w, "malformed\n\n")

			return "", false
		}
	case message.Method == "notifications/cancelled":
		p.stats.Cancels++

		if gate := p.gate; p.holdCancel && gate != nil {
			p.stats.CancelInflight++
			p.signalLocked()
			p.mu.Unlock()
			<-gate
			p.mu.Lock()
			p.stats.CancelInflight--
		}
	}

	p.signalLocked()

	return message.Method, true
}

func (p *cleanupPeer) serveDelete(w http.ResponseWriter, req *http.Request, session string) {
	p.mu.Lock()
	gate := p.gate
	p.deletes[session]++
	p.stats.DeleteStarted++
	p.stats.DeleteInflight++
	p.stats.DeletePeak = max(p.stats.DeletePeak, p.stats.DeleteInflight)
	p.signalLocked()
	p.mu.Unlock()

	if gate != nil {
		// Deliberately ignore the request context: the peer stays blocked even
		// after the SDK gives up, until the test releases it.
		<-gate
	}

	p.mu.Lock()
	pace, paceOff := p.pace, p.paceOff
	p.mu.Unlock()

	if pace != nil {
		select {
		case <-pace:
		case <-paceOff:
		}
	}

	p.mu.Lock()
	if p.live[session] {
		delete(p.live, session)
		p.stats.Live--
	}

	fail := p.failDelete
	p.signalLocked()
	p.mu.Unlock()

	if fail {
		// Drop the connection without a response: the SDK's Close returns the
		// transport error.
		if conn, _, err := http.NewResponseController(w).Hijack(); err == nil {
			_ = conn.Close()
		}
	} else {
		p.handler.ServeHTTP(w, req)
	}

	p.mu.Lock()
	p.stats.DeleteInflight--
	p.stats.DeleteDone++
	p.signalLocked()
	p.mu.Unlock()
}

func (p *cleanupPeer) serveStream(w http.ResponseWriter, partition string) bool {
	p.mu.Lock()
	fail := p.failStream[partition]
	p.mu.Unlock()

	if !fail {
		return false
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusInternalServerError)

	return true
}

type cleanupResponseWriter struct {
	http.ResponseWriter

	header func(http.Header)
	wrote  bool
}

func (w *cleanupResponseWriter) WriteHeader(code int) {
	if !w.wrote {
		w.wrote = true
		w.header(w.Header())
	}

	w.ResponseWriter.WriteHeader(code)
}

func (w *cleanupResponseWriter) Write(b []byte) (int, error) {
	if !w.wrote {
		w.WriteHeader(http.StatusOK)
	}

	return w.ResponseWriter.Write(b)
}

func (w *cleanupResponseWriter) Flush() {
	if !w.wrote {
		w.WriteHeader(http.StatusOK)
	}

	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *cleanupResponseWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

type cleanupClientStats struct {
	DeleteReturned int `json:"delete_returned"`
	CancelReturned int `json:"cancel_returned"`
	Dials          int `json:"dials"`
	Sockets        int `json:"sockets"`
	SocketPeak     int `json:"socket_peak"`
}

// cleanupClient is the registry's base round tripper. It observes when the
// SDK's DELETE round trip returns and counts client-side sockets.
type cleanupClient struct {
	transport *http.Transport

	mu      sync.Mutex
	changed chan struct{}
	stats   cleanupClientStats
}

func newCleanupClient(t *testing.T) *cleanupClient {
	t.Helper()

	client := &cleanupClient{changed: make(chan struct{})}
	dialer := &net.Dialer{}
	client.transport = &http.Transport{
		MaxIdleConnsPerHost: 256,
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			conn, err := dialer.DialContext(ctx, network, address)
			if err != nil {
				return nil, err
			}

			client.mu.Lock()
			client.stats.Dials++
			client.stats.Sockets++
			client.stats.SocketPeak = max(client.stats.SocketPeak, client.stats.Sockets)
			client.mu.Unlock()

			return &cleanupConn{Conn: conn, client: client}, nil
		},
	}
	t.Cleanup(client.transport.CloseIdleConnections)

	return client
}

func (c *cleanupClient) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := c.transport.RoundTrip(req)

	switch {
	case req.Method == http.MethodDelete:
		c.mu.Lock()
		c.stats.DeleteReturned++
		close(c.changed)
		c.changed = make(chan struct{})
		c.mu.Unlock()
	case req.Header.Get("Mcp-Method") == "notifications/cancelled":
		c.mu.Lock()
		c.stats.CancelReturned++
		c.mu.Unlock()
	}

	return resp, err
}

func (c *cleanupClient) snapshot() cleanupClientStats {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.stats
}

// awaitDeletes is a channel barrier on SDK DELETE round trips returning.
func (c *cleanupClient) awaitDeletes(t *testing.T, n int) {
	t.Helper()

	timer := time.NewTimer(cleanupBarrier)
	defer timer.Stop()

	for {
		c.mu.Lock()
		returned, changed := c.stats.DeleteReturned, c.changed
		c.mu.Unlock()

		if returned >= n {
			return
		}

		select {
		case <-changed:
		case <-timer.C:
			t.Fatalf("only %d of %d DELETE round trips returned", returned, n)
		}
	}
}

type cleanupConn struct {
	net.Conn

	client *cleanupClient
	once   sync.Once
}

func (c *cleanupConn) Close() error {
	c.once.Do(func() {
		c.client.mu.Lock()
		c.client.stats.Sockets--
		c.client.mu.Unlock()
	})

	return c.Conn.Close()
}

type cleanupFixture struct {
	peer     *cleanupPeer
	client   *cleanupClient
	registry *Registry
}

// newCleanupFixture registers peer release before registry Close in cleanup
// order, so a failing test never leaves a peer DELETE blocked forever.
func newCleanupFixture(t *testing.T, legacy bool, opts ...RegistryOption) *cleanupFixture {
	t.Helper()

	peer := newCleanupPeer(t, legacy)
	client := newCleanupClient(t)
	registry, err := NewRegistry("remote", peer.url, append([]RegistryOption{
		WithCredentials(cleanupCredentials), WithRegistryHTTPClient(&http.Client{Transport: client}),
	}, opts...)...)
	require.NoError(t, err)
	t.Cleanup(func() { _ = registry.Close() })
	t.Cleanup(peer.release)

	return &cleanupFixture{peer: peer, client: client, registry: registry}
}

func (f *cleanupFixture) list(t *testing.T, partition string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), partition), cleanupBarrier)
	defer cancel()

	definitions, err := f.registry.List(ctx)
	require.NoError(t, err, partition)
	require.Equal(t, []string{cleanupToolName}, registryNames(definitions))
}

func (f *cleanupFixture) whoami(t *testing.T, partition string) {
	t.Helper()

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), partition), cleanupBarrier)
	defer cancel()

	response, err := f.registry.Execute(ctx, cleanupRequest())
	require.NoError(t, err, partition)
	require.False(t, response.IsError, string(response.Result))
	require.Contains(t, string(response.Result), "caller=Bearer "+partition+`"`)
}

func cleanupRun(ctx context.Context, op func(context.Context) error) <-chan error {
	done := make(chan error, 1)

	go func() { done <- op(ctx) }()

	return done
}

func cleanupResult(t *testing.T, what string, done <-chan error) error {
	t.Helper()

	select {
	case err := <-done:
		return err
	case <-time.After(cleanupBarrier):
		t.Fatalf("%s never returned", what)

		return nil
	}
}

func cleanupEndName(deadline bool) string {
	if deadline {
		return "deadline"
	}

	return "cancel"
}

type cleanupOperation struct {
	name string
	run  func(ctx context.Context, f *cleanupFixture, captured tool.Registry) error
}

func cleanupOperations() []cleanupOperation {
	return []cleanupOperation{
		{"Snapshot", func(ctx context.Context, f *cleanupFixture, _ tool.Registry) error {
			_, err := f.registry.Snapshot(ctx)
			return err
		}},
		{"List", func(ctx context.Context, f *cleanupFixture, _ tool.Registry) error {
			_, err := f.registry.List(ctx)
			return err
		}},
		{"Get", func(ctx context.Context, f *cleanupFixture, _ tool.Registry) error {
			_, err := f.registry.Get(ctx, cleanupToolName)
			return err
		}},
		{"Execute", func(ctx context.Context, f *cleanupFixture, _ tool.Registry) error {
			_, err := f.registry.Execute(ctx, cleanupRequest())
			return err
		}},
		{"CapturedExecute", func(ctx context.Context, _ *cleanupFixture, captured tool.Registry) error {
			captive, err := captured.Get(ctx, cleanupToolName)
			if err != nil {
				return err
			}

			_, err = captive.Execute(ctx, json.RawMessage(`{}`))

			return err
		}},
	}
}

// TestRegistryEvictionReturnsWhilePeerDeleteBlocked is the regression for
// synchronous eviction cleanup: with one connection slot, Bob evicts Alice's
// legacy session whose session-termination DELETE is held at the peer. Bob
// must observe his own deadline or cancellation while that DELETE is still
// blocked at the peer and its SDK round trip has not returned, and the slot
// must not be reused for Bob until Alice's SDK Close has returned.
func TestRegistryEvictionReturnsWhilePeerDeleteBlocked(t *testing.T) {
	t.Parallel()

	for _, op := range cleanupOperations() {
		for _, deadline := range []bool{true, false} {
			t.Run(op.name+"/"+cleanupEndName(deadline), func(t *testing.T) {
				t.Parallel()

				f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))

				setup, stop := context.WithTimeout(cleanupCaller(t.Context(), "bob"), cleanupBarrier)
				defer stop()

				captured, err := f.registry.Snapshot(setup)
				require.NoError(t, err)
				f.list(t, "alice")
				aliceSessions, bobSessions := f.peer.sessionsOf("alice"), f.peer.sessionsOf("bob")
				require.Len(t, aliceSessions, 1, "Alice must hold a legacy stateful session")
				require.Len(t, bobSessions, 1)
				f.client.awaitDeletes(t, 1)
				require.Equal(t, map[string]int{bobSessions[0]: 1}, f.peer.deleteCounts(), "Alice's open evicted Bob's first session")

				f.peer.hold()

				var (
					ctx    context.Context
					cancel context.CancelFunc
				)
				if deadline {
					ctx, cancel = context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
				} else {
					ctx, cancel = context.WithCancel(cleanupCaller(t.Context(), "bob"))
				}
				defer cancel()

				done := cleanupRun(ctx, func(ctx context.Context) error { return op.run(ctx, f, captured) })

				f.peer.await(t, "Alice's DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })

				if !deadline {
					cancel()
				}

				err = cleanupResult(t, "Bob's "+op.name, done)
				peer, client := f.peer.snapshot(), f.client.snapshot()

				if deadline {
					require.ErrorIs(t, err, context.DeadlineExceeded)
				} else {
					require.ErrorIs(t, err, context.Canceled)
				}

				require.Equal(t, 1, peer.DeleteInflight, "Alice's DELETE must still be blocked at the peer")
				require.Equal(t, 1, client.DeleteReturned, "Bob returned only after the SDK's DELETE round trip for Alice ended")
				require.Equal(t, 1, f.peer.openCount("bob"), "one slot must not open Bob's session while Alice's close is in flight")

				f.peer.release()
				f.client.awaitDeletes(t, 2)
				f.whoami(t, "bob")
				assert.Equal(t, map[string]int{bobSessions[0]: 1, aliceSessions[0]: 1}, f.peer.deleteCounts())
				assert.Empty(t, f.peer.identityMismatches())
			})
		}
	}
}

// TestRegistryModernEvictionReturnsWhileCancelBlocked is the modern-protocol
// variant. Stateless sessions have no Mcp-Session-Id, so go-sdk sends no
// DELETE; instead Close cancels the subscriptions/listen stream and waits for
// its notifications/cancelled, whose send go-sdk times out after five seconds.
// Holding that notification at the peer must not hold Bob, and the slot is not
// reused until Close ends.
func TestRegistryModernEvictionReturnsWhileCancelBlocked(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, false, WithRegistryMaxConnections(1))
	f.whoami(t, "alice")
	f.peer.await(t, "Alice's listen stream", func(s cleanupPeerStats) bool { return s.ListenInflight == 1 })

	f.peer.mu.Lock()
	f.peer.holdCancel = true
	f.peer.mu.Unlock()
	f.peer.hold()

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
	defer cancel()

	done := cleanupRun(ctx, func(ctx context.Context) error {
		_, err := f.registry.Execute(ctx, cleanupRequest())
		return err
	})

	f.peer.await(t, "Alice's held listen cancellation", func(s cleanupPeerStats) bool { return s.CancelInflight == 1 })
	require.ErrorIs(t, cleanupResult(t, "Bob's Execute", done), context.DeadlineExceeded)
	require.Equal(t, 1, f.peer.snapshot().CancelInflight, "Alice's cancellation must still be blocked at the peer")
	require.Zero(t, f.client.snapshot().CancelReturned, "Bob returned only after the SDK's cancellation round trip ended")
	require.Zero(t, f.peer.openCount("bob"))

	f.peer.release()
	f.whoami(t, "bob")

	stats := f.peer.await(t, "one listen stream per resident session", func(s cleanupPeerStats) bool { return s.ListenInflight == 1 })
	assert.Equal(t, 2, stats.Listens)
	assert.Zero(t, stats.DeleteStarted, "modern sessions are never DELETEd")
	assert.Zero(t, stats.Sessions)

	require.NoError(t, f.registry.Close())
	f.peer.await(t, "listen streams closed", func(s cleanupPeerStats) bool { return s.ListenInflight == 0 })
	assert.Zero(t, f.peer.snapshot().DeleteStarted)
}

// TestRegistryIdleSweepClosesInBackground expires several idle legacy sessions
// at once. With spare capacity the sweeping caller connects while every
// termination DELETE is still blocked; with exactly the swept capacity it must
// wait for real capacity and still honor its deadline.
func TestRegistryIdleSweepClosesInBackground(t *testing.T) {
	t.Parallel()

	for _, spare := range []bool{true, false} {
		t.Run(fmt.Sprintf("spare=%t", spare), func(t *testing.T) {
			t.Parallel()

			limit := 3
			if spare {
				limit = 4
			}

			clock := newRegistryTestClock()
			f := newCleanupFixture(t, true, WithRegistryMaxConnections(limit), WithRegistryIdleTimeout(time.Minute), func(cfg *registryConfig) { cfg.now = clock.Now })

			for _, partition := range []string{"a", "b", "c"} {
				f.list(t, partition)
			}

			f.peer.hold()
			clock.Advance(time.Minute)

			ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "d"), 2*time.Second)
			defer cancel()

			if !spare {
				cancel()

				ctx, cancel = context.WithTimeout(cleanupCaller(t.Context(), "d"), 100*time.Millisecond)
				defer cancel()
			}

			done := cleanupRun(ctx, func(ctx context.Context) error {
				_, err := f.registry.Execute(ctx, cleanupRequest())
				return err
			})

			f.peer.await(t, "three concurrent DELETEs", func(s cleanupPeerStats) bool { return s.DeleteInflight == 3 })

			err := cleanupResult(t, "sweeping caller", done)
			require.Equal(t, 3, f.peer.snapshot().DeleteInflight, "swept DELETEs must still be blocked")
			require.Zero(t, f.client.snapshot().DeleteReturned)

			if spare {
				require.NoError(t, err)
				require.Equal(t, 1, f.peer.openCount("d"))
			} else {
				require.ErrorIs(t, err, context.DeadlineExceeded)
				require.Zero(t, f.peer.openCount("d"), "swept sessions still occupy capacity until closed")
			}

			f.peer.release()
			f.peer.await(t, "swept DELETEs", func(s cleanupPeerStats) bool { return s.DeleteDone == 3 })
			f.whoami(t, "d")
			assert.Len(t, f.peer.deleteCounts(), 3)

			for session, n := range f.peer.deleteCounts() {
				assert.Equal(t, 1, n, session)
			}
		})
	}
}

// TestRegistryCanceledCallerDoesNotEvict checks an already-canceled caller
// returns without sweeping, evicting or closing anyone's session.
func TestRegistryCanceledCallerDoesNotEvict(t *testing.T) {
	t.Parallel()

	clock := newRegistryTestClock()
	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1), WithRegistryIdleTimeout(time.Minute), func(cfg *registryConfig) { cfg.now = clock.Now })
	f.list(t, "alice")
	clock.Advance(time.Hour)
	f.peer.hold()

	ctx, cancel := context.WithCancel(cleanupCaller(t.Context(), "bob"))
	cancel()

	for _, op := range cleanupOperations()[:4] {
		require.ErrorIs(t, op.run(ctx, f, nil), context.Canceled, op.name)
	}

	assert.Zero(t, f.peer.snapshot().DeleteStarted)
	assert.Zero(t, f.peer.openCount("bob"))

	f.registry.mu.Lock()
	_, resident := f.registry.entries["alice"]
	f.registry.mu.Unlock()
	assert.True(t, resident, "a canceled caller must not evict")
}

// TestRegistryEvictsFailedSessionWhileSDKDeleteBlocked fails Alice's legacy
// connection with a malformed SSE event. go-sdk's jsonrpc2 reader then sends
// the termination DELETE itself while holding the connection state lock, so
// that session's Close blocks until the DELETE returns. Evicting it for Bob
// must still honor Bob's deadline and keep the slot until Close returns.
func TestRegistryEvictsFailedSessionWhileSDKDeleteBlocked(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
	f.whoami(t, "alice")
	aliceSessions := f.peer.sessionsOf("alice")
	require.Len(t, aliceSessions, 1)

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "alice"), cleanupBarrier)
	defer cancel()

	captured, err := f.registry.Snapshot(ctx)
	require.NoError(t, err)
	captive, err := captured.Get(ctx, cleanupToolName)
	require.NoError(t, err)

	f.peer.hold()
	f.peer.breakCall("alice")

	_, err = captive.Execute(ctx, json.RawMessage(`{}`))
	require.Error(t, err, "the malformed stream must fail the call that received it")
	require.NotErrorIs(t, err, sdkmcp.ErrConnectionClosed, "a stream error alone does not retire the session")
	f.peer.await(t, "the SDK reader's own DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })

	bob, stop := context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
	defer stop()

	err = cleanupResult(t, "Bob's List", cleanupRun(bob, func(ctx context.Context) error {
		_, err := f.registry.List(ctx)
		return err
	}))
	require.ErrorIs(t, err, context.DeadlineExceeded)
	require.Equal(t, 1, f.peer.snapshot().DeleteInflight, "Alice's DELETE must still be blocked at the peer")
	require.Zero(t, f.client.snapshot().DeleteReturned, "Bob returned only after the SDK's DELETE round trip ended")
	require.Zero(t, f.peer.openCount("bob"), "the failed session keeps its slot until its Close returns")

	f.peer.release()
	f.client.awaitDeletes(t, 1)
	f.whoami(t, "bob")
	assert.Equal(t, map[string]int{aliceSessions[0]: 1}, f.peer.deleteCounts())
}

// TestRegistryRetiredReleaseClosesInBackground drives the pool directly: two
// leases share Alice's live legacy session, which is then retired as the SDK
// would report ErrConnectionClosed. While leased it keeps the only slot and is
// not closed; the last release must return while the session's termination
// DELETE is held at the peer, and the slot stays occupied until Close returns.
func TestRegistryRetiredReleaseClosesInBackground(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
	f.whoami(t, "alice")
	aliceSessions := f.peer.sessionsOf("alice")
	require.Len(t, aliceSessions, 1)

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "alice"), cleanupBarrier)
	defer cancel()

	ctx, end, err := f.registry.begin(ctx)
	require.NoError(t, err)

	defer end()

	first, releaseFirst, err := f.registry.acquire(ctx, "alice")
	require.NoError(t, err)
	second, releaseSecond, err := f.registry.acquire(ctx, "alice")
	require.NoError(t, err)
	require.Same(t, first, second)

	f.peer.hold()
	f.registry.retireClosed(first, sdkmcp.ErrConnectionClosed)

	bobCannotOpen := func(what string) {
		short, stop := context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
		defer stop()

		err := cleanupResult(t, "Bob's List "+what, cleanupRun(short, func(ctx context.Context) error {
			_, err := f.registry.List(ctx)
			return err
		}))
		require.ErrorIs(t, err, context.DeadlineExceeded, what)
		require.Zero(t, f.peer.openCount("bob"), what)
	}

	bobCannotOpen("while Alice's retired session is leased")
	releaseFirst()
	bobCannotOpen("while one retired lease remains")
	require.Zero(t, f.peer.snapshot().DeleteStarted, "a leased session must not be closed")

	done := cleanupRun(ctx, func(context.Context) error {
		releaseSecond()
		return nil
	})

	f.peer.await(t, "Alice's DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })
	require.NoError(t, cleanupResult(t, "the last release", done))
	require.Zero(t, f.client.snapshot().DeleteReturned, "the last release returned only after the SDK's DELETE round trip ended")
	bobCannotOpen("while Alice's session is closing")

	f.peer.release()
	f.client.awaitDeletes(t, 1)
	f.whoami(t, "bob")
	f.whoami(t, "alice")
	assert.Equal(t, 2, f.peer.openCount("alice"))
	assert.Equal(t, 1, f.peer.deleteCounts()[aliceSessions[0]])
	assert.Empty(t, f.peer.identityMismatches())
}

// TestRegistryDetachedCloseErrorsReachShutdown makes termination DELETEs fail
// at the transport. Background eviction closes must not drop their errors:
// Shutdown joins them, bounded to a count and the first error, with the final
// close errors, and repeated Shutdown and Close calls return that same result.
func TestRegistryDetachedCloseErrorsReachShutdown(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
	f.list(t, "alice")
	f.peer.mu.Lock()
	f.peer.failDelete = true
	f.peer.mu.Unlock()

	f.list(t, "bob")
	f.client.awaitDeletes(t, 1)
	f.list(t, "carol")
	f.client.awaitDeletes(t, 2)
	f.peer.hold()

	canceled, cancel := context.WithCancel(t.Context())
	cancel()
	require.ErrorIs(t, f.registry.Shutdown(canceled), context.Canceled)
	f.peer.await(t, "Carol's final DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })

	for range 2 {
		expired, stop := context.WithTimeout(t.Context(), 20*time.Millisecond)
		require.ErrorIs(t, f.registry.Shutdown(expired), context.DeadlineExceeded)
		stop()
	}

	f.peer.release()

	err := f.registry.Close()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "mcp: 2 detached session close(s) failed, first: ")
	assert.Equal(t, 2, strings.Count(err.Error(), `Delete "`+f.peer.url), "Carol's final close and the first detached failure: %v", err)

	again := f.registry.Close()
	require.Error(t, again)
	assert.Equal(t, err.Error(), again.Error())

	ctx, stop := context.WithTimeout(t.Context(), cleanupBarrier)
	defer stop()

	assert.Equal(t, err.Error(), f.registry.Shutdown(ctx).Error())
}

// TestRegistryShutdownDuringBlockedCleanup interleaves canceled and expired
// Shutdown calls with an eviction DELETE and resident sessions held at the
// peer. New admissions fail immediately, and a later Shutdown drains every
// session exactly once after the peer releases.
func TestRegistryShutdownDuringBlockedCleanup(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(2))
	for _, partition := range []string{"alice", "bob"} {
		f.list(t, partition)
	}

	f.peer.hold()

	ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "carol"), 100*time.Millisecond)
	defer cancel()

	done := cleanupRun(ctx, func(ctx context.Context) error {
		_, err := f.registry.List(ctx)
		return err
	})

	f.peer.await(t, "eviction DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })
	require.ErrorIs(t, cleanupResult(t, "Carol's List", done), context.DeadlineExceeded)

	canceled, cancelShutdown := context.WithCancel(t.Context())
	cancelShutdown()
	require.ErrorIs(t, f.registry.Shutdown(canceled), context.Canceled)

	for range 3 {
		expired, stop := context.WithTimeout(t.Context(), 20*time.Millisecond)
		require.ErrorIs(t, f.registry.Shutdown(expired), context.DeadlineExceeded)
		stop()
	}

	admitted, stop := context.WithTimeout(cleanupCaller(t.Context(), "dave"), cleanupBarrier)
	defer stop()

	_, err := f.registry.List(admitted)
	require.ErrorIs(t, err, ErrClosed)
	require.Zero(t, f.peer.openCount("dave"))
	f.peer.await(t, "shutdown DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 2 })
	require.Zero(t, f.client.snapshot().DeleteReturned)

	f.peer.release()

	drain, stopDrain := context.WithTimeout(t.Context(), cleanupBarrier)
	defer stopDrain()

	require.NoError(t, f.registry.Shutdown(drain))
	require.NoError(t, f.registry.Close())
	assert.Equal(t, 2, f.peer.snapshot().DeleteDone)
	assert.Len(t, f.peer.deleteCounts(), 2)

	for session, n := range f.peer.deleteCounts() {
		assert.Equal(t, 1, n, session)
	}
}

// TestRegistryConcurrentCloseWhileCleanupBlocked closes the registry from many
// goroutines while termination DELETEs are blocked. Every caller observes the
// same drained result once the peer releases.
func TestRegistryConcurrentCloseWhileCleanupBlocked(t *testing.T) {
	t.Parallel()

	f := newCleanupFixture(t, true, WithRegistryMaxConnections(3))
	for _, partition := range []string{"a", "b", "c"} {
		f.list(t, partition)
	}

	f.peer.hold()

	const closers = 16

	results := make(chan error, closers)

	for i := range closers {
		go func() {
			if i%2 == 0 {
				results <- f.registry.Close()

				return
			}

			ctx, cancel := context.WithTimeout(t.Context(), cleanupBarrier)
			defer cancel()

			results <- f.registry.Shutdown(ctx)
		}()
	}

	f.peer.await(t, "shutdown DELETEs", func(s cleanupPeerStats) bool { return s.DeleteStarted == 3 })
	f.peer.release()

	for range closers {
		select {
		case err := <-results:
			require.NoError(t, err)
		case <-time.After(cleanupBarrier):
			t.Fatal("concurrent Close never returned")
		}
	}

	assert.Equal(t, 3, f.peer.snapshot().DeleteDone)

	for session, n := range f.peer.deleteCounts() {
		assert.Equal(t, 1, n, session)
	}
}

// TestRegistryInitializationFailureReleasesCapacity covers a refused open,
// which has no session to terminate, and a legacy stream failure after the
// session id was assigned, where the SDK's own Connect sends the DELETE.
func TestRegistryInitializationFailureReleasesCapacity(t *testing.T) {
	t.Parallel()

	t.Run("refused", func(t *testing.T) {
		t.Parallel()

		f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
		f.peer.mu.Lock()
		f.peer.failOpen["alice"] = true
		f.peer.mu.Unlock()

		ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "alice"), cleanupBarrier)
		defer cancel()

		_, err := f.registry.List(ctx)
		require.Error(t, err)
		f.whoami(t, "bob")
		assert.Zero(t, f.peer.snapshot().DeleteStarted)
	})

	t.Run("stream", func(t *testing.T) {
		t.Parallel()

		f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
		f.peer.mu.Lock()
		f.peer.failStream["alice"] = true
		f.peer.mu.Unlock()

		ctx, cancel := context.WithTimeout(cleanupCaller(t.Context(), "alice"), cleanupBarrier)
		defer cancel()

		_, err := f.registry.List(ctx)
		require.Error(t, err)
		require.NotErrorIs(t, err, context.DeadlineExceeded)
		f.peer.await(t, "SDK init-failure DELETE", func(s cleanupPeerStats) bool { return s.DeleteDone == 1 })
		f.whoami(t, "bob")
		assert.Len(t, f.peer.sessionsOf("alice"), 1)
	})

	// go-sdk's Connect closes a half-initialized legacy session itself, sending
	// the DELETE before Connect returns. The caller must still leave on its own
	// cancellation, while the connecting entry keeps the only slot.
	t.Run("stream/blocked", func(t *testing.T) {
		t.Parallel()

		f := newCleanupFixture(t, true, WithRegistryMaxConnections(1))
		f.peer.mu.Lock()
		f.peer.failStream["alice"] = true
		f.peer.mu.Unlock()
		f.peer.hold()

		ctx, cancel := context.WithCancel(cleanupCaller(t.Context(), "alice"))
		defer cancel()

		done := cleanupRun(ctx, func(ctx context.Context) error {
			_, err := f.registry.List(ctx)
			return err
		})

		f.peer.await(t, "SDK init-failure DELETE", func(s cleanupPeerStats) bool { return s.DeleteInflight == 1 })
		cancel()
		require.ErrorIs(t, cleanupResult(t, "Alice's List", done), context.Canceled)
		require.Zero(t, f.client.snapshot().DeleteReturned, "Alice returned only after the SDK's DELETE round trip ended")

		bob, stop := context.WithTimeout(cleanupCaller(t.Context(), "bob"), 100*time.Millisecond)
		defer stop()

		_, err := f.registry.List(bob)
		require.ErrorIs(t, err, context.DeadlineExceeded)
		require.Zero(t, f.peer.openCount("bob"), "a connecting entry keeps its slot until Connect returns")
		require.Equal(t, 1, f.peer.snapshot().DeleteInflight)

		f.peer.release()
		f.client.awaitDeletes(t, 1)
		f.whoami(t, "bob")
		assert.Equal(t, 1, f.peer.deleteCounts()[f.peer.sessionsOf("alice")[0]])
	})
}

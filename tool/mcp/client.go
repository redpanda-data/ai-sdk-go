package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	sdkmcp "github.com/modelcontextprotocol/go-sdk/mcp"
	"golang.org/x/sync/singleflight"

	"github.com/redpanda-data/ai-sdk-go/llm"
	"github.com/redpanda-data/ai-sdk-go/tool"
)

// Sentinel errors for client state management.
var (
	// ErrNotStarted indicates the client has not been started yet.
	ErrNotStarted = errors.New("mcp client not started")

	// ErrClosed indicates the client has been closed.
	ErrClosed = errors.New("mcp client closed")
)

const (
	reconnectInitialDelay = 200 * time.Millisecond
	reconnectMaxDelay     = 30 * time.Second
)

// Client connects to an MCP server to discover and execute tools.
// Tools are accessible via ListTools() and ExecuteTool(), or can be
// automatically registered with a tool.Registry for LLM consumption.
type Client interface {
	// Start connects to the MCP server and performs initial tool sync.
	// The provided context is used only for the initial connection and sync operations.
	// After Start returns, the client runs independently with its own background context.
	//
	// Automatically registers tools if a registry is configured.
	// Launches background sync goroutine if auto-sync is enabled.
	//
	// Note: This is a single-shot connection attempt with no automatic retry.
	// If connection or initial sync fails, the client remains stopped and the
	// caller must handle retry logic if desired.
	Start(ctx context.Context) error

	// Shutdown gracefully stops the client, respecting the provided context deadline.
	// It waits for in-flight operations to complete or until ctx expires.
	// Returns ctx.Err() if shutdown times out, nil on clean shutdown.
	// Idempotent and safe to call multiple times.
	Shutdown(ctx context.Context) error

	// Close gracefully shuts down the client using a background context with default timeout.
	// Satisfies io.Closer. Equivalent to Shutdown(context.Background()) with 30s timeout.
	// Idempotent and safe to call multiple times.
	io.Closer

	// ServerID returns the unique identifier for this MCP server connection.
	ServerID() string

	// Started returns true if the client is currently connected and running.
	Started() bool

	// SyncTools fetches tools from the MCP server and reconciles with the registry
	// (if configured). Concurrent calls are collapsed via singleflight.
	SyncTools(ctx context.Context) error

	// ListTools returns tool definitions for all available MCP server tools.
	// Tool names are always namespaced with serverID (e.g., "github/create-issue").
	ListTools(ctx context.Context) ([]llm.ToolDefinition, error)

	// ExecuteTool executes a tool using the namespaced name from ListTools().
	// The client automatically strips the namespace prefix before calling the MCP server.
	ExecuteTool(ctx context.Context, toolName string, args json.RawMessage) (json.RawMessage, error)
}

// Ensure clientImpl implements Client at compile time.
var _ Client = (*clientImpl)(nil)

// clientImpl is the internal implementation of the Client interface.
//
// Lifecycle:
//   - Start(ctx) uses the caller's context ONLY for initial connection and sync.
//     Creates a background-rooted context (bgCtx) that governs the client's lifetime,
//     ensuring the client doesn't die if the caller's Start context expires.
//     Uses sync.Once to ensure initialization runs exactly once. If Start fails,
//     subsequent calls return the same error - the client instance cannot be retried.
//   - Shutdown(ctx) gracefully terminates: cancels bgCtx and waits for in-flight
//     operations (tracked by wg) to complete or ctx to expire. Uses sync.Once for
//     idempotent shutdown.
//   - Close() is a convenience wrapper that calls Shutdown with a background context
//     and default timeout (satisfies io.Closer).
//   - Operations (ExecuteTool, SyncTools) use opContext() to combine the client's bgCtx
//     with the caller's ctx, honoring both the client shutdown signal and caller deadlines.
//
// Concurrency:
//   - A read-write mutex (mu) protects session/tools state.
//   - startOnce and stopOnce ensure initialization and shutdown execute exactly once.
//   - Network I/O is performed outside locks to avoid blocking other operations.
//   - singleflight collapses concurrent sync requests to reduce server load.
//   - beginOp() prevents WaitGroup races during shutdown by checking state before Add(1).
//
// Design Invariants:
//   - Only one sessionManagerLoop runs per client instance (guaranteed by startOnce).
//   - The sessionManagerLoop naturally re-fetches session state at the top of each iteration.
//   - Session changes are signaled via sessionChanged channel (closed and replaced pattern).
//   - All operations respect both client lifetime (bgCtx) and caller deadlines (via opContext).
type clientImpl struct {
	// Configuration (immutable after construction)
	serverID         string
	transportFactory TransportFactory
	registry         tool.Registry
	autoSyncInterval time.Duration
	shutdownTimeout  time.Duration
	toolTimeout      time.Duration
	logger           *slog.Logger
	toolFilter       ToolFilterFunc

	// MCP SDK components
	mcpClient *sdkmcp.Client
	session   *sdkmcp.ClientSession

	// Lifecycle management
	mu        sync.RWMutex
	startOnce sync.Once          // Ensures Start executes exactly once
	startErr  error              // Error from Start, if any
	stopOnce  sync.Once          // Ensures shutdown executes exactly once
	closing   atomic.Bool        // Set at start of Shutdown to block new operations
	done      chan struct{}      // Closed when fully shut down
	bgCtx     context.Context    // Background-rooted context for client lifetime
	cancel    context.CancelFunc // Cancels bgCtx to signal shutdown
	wg        sync.WaitGroup     // Tracks background goroutines and in-flight operations
	wgDone    func() <-chan struct{}

	// Tool synchronization
	tools     map[string]*toolWrapper
	syncGroup singleflight.Group
	notifyCh  chan struct{}
	// sessionChanged is closed and replaced whenever the session pointer is updated,
	// allowing waitForSession to wake up and check for a new session.
	sessionChanged chan struct{}
}

// NewClient creates a new MCP client. The client must be started with Start()
// before it can be used.
//
// Parameters:
//   - serverID: Unique identifier used as namespace prefix for tools
//   - transportFactory: Creates the transport (stdio, HTTP, SSE)
//   - opts: Optional configuration (registry, auto-sync, logger, filters)
//
// Example:
//
//	client, err := NewClient("github", transportFactory,
//	    WithRegistry(registry),
//	    WithAutoSync(5*time.Minute),
//	)
func NewClient(serverID string, transportFactory TransportFactory, opts ...ClientOption) (Client, error) {
	if serverID == "" {
		return nil, errors.New("serverID cannot be empty")
	}

	if transportFactory == nil {
		return nil, errors.New("transportFactory cannot be nil")
	}

	c := &clientImpl{
		serverID:         serverID,
		transportFactory: transportFactory,
		logger:           slog.Default(),
		tools:            make(map[string]*toolWrapper),
		notifyCh:         make(chan struct{}, 1),
		sessionChanged:   make(chan struct{}),
		done:             make(chan struct{}),
	}
	c.wgDone = sync.OnceValue(func() <-chan struct{} {
		ch := make(chan struct{})

		go func() {
			c.wg.Wait()
			close(ch)
		}()

		return ch
	})

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// Start connects to the MCP server and performs initial tool sync.
// The provided context is used only for initial connection and sync; the client
// runs independently afterward with a background context.
//
// Start executes exactly once using sync.Once. Concurrent calls will block until
// the first call completes, then return the same result. If Start fails, the client
// is permanently in a failed state - create a new client to retry.
func (c *clientImpl) Start(ctx context.Context) error {
	c.startOnce.Do(func() {
		if c.isShutdown() {
			c.mu.Lock()
			c.startErr = ErrClosed
			c.mu.Unlock()

			return
		}

		// Create background context and cancel for client lifetime
		bgCtx, cancel := context.WithCancel(context.Background())

		c.mu.Lock()
		c.bgCtx = bgCtx
		c.cancel = cancel
		c.mu.Unlock()

		// Connect to MCP server
		session, mcpClient, err := c.connect(ctx)
		if err != nil {
			c.mu.Lock()
			c.startErr = err
			c.mu.Unlock()
			c.cleanupAfterFailedStart(nil)

			return
		}

		// Update client state with connection details
		c.mu.Lock()
		c.replaceSessionLocked(session, mcpClient)
		c.mu.Unlock()

		c.logger.Info("connected to MCP server", "serverID", c.serverID)

		// Perform initial tool sync
		if err := c.SyncTools(ctx); err != nil {
			c.mu.Lock()
			c.startErr = fmt.Errorf("initial tool sync failed: %w", err)
			c.mu.Unlock()
			c.cleanupAfterFailedStart(session)

			return
		}

		// Start auto-sync if enabled
		if c.autoSyncInterval > 0 && c.registry != nil {
			c.wg.Add(1)
			//nolint:contextcheck // autoSyncLoop intentionally uses c.bgCtx for client lifetime, not caller context
			go c.autoSyncLoop()

			c.logger.Info("auto-sync enabled",
				"serverID", c.serverID,
				"interval", c.autoSyncInterval)
		}

		// Start unified session manager loop to monitor connection and handle reconnection
		c.wg.Add(1)
		//nolint:contextcheck // sessionManagerLoop intentionally uses c.bgCtx for client lifetime, not caller context
		go c.sessionManagerLoop()

		c.logger.Info("MCP client started", "serverID", c.serverID)
	})

	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.startErr
}

// Shutdown gracefully stops the client, waiting for in-flight operations to complete.
// Respects the provided context deadline. Returns ctx.Err() if shutdown times out.
// Idempotent and safe to call multiple times.
func (c *clientImpl) Shutdown(ctx context.Context) error {
	var retErr error

	c.stopOnce.Do(func() {
		// 1) Block new operations immediately
		c.closing.Store(true)

		// 2) Snapshot state under lock
		c.mu.RLock()
		cancel := c.cancel
		session := c.session

		var toolNames []string
		if c.registry != nil {
			toolNames = make([]string, 0, len(c.tools))
			for name := range c.tools {
				toolNames = append(toolNames, name)
			}
		}

		c.mu.RUnlock()

		// 3) Cancel background context to signal shutdown
		if cancel != nil {
			cancel()
		}

		var errs []error

		// 4) Close session BEFORE waiting for goroutines to prevent deadlock.
		// The sessionManagerLoop is blocked in session.Wait() and can only exit
		// after the session is closed. Closing here allows it to unblock and complete.
		if session != nil {
			err := session.Close()
			if err != nil {
				errs = append(errs, fmt.Errorf("close MCP session: %w", err))
			}
		}

		// 5) Wait for background goroutines and in-flight operations
		select {
		case <-c.wgDone():
			// All operations completed within deadline
		case <-ctx.Done():
			c.logger.Warn("graceful shutdown timed out, forcing closure",
				"err", ctx.Err(),
				"serverID", c.serverID)
			retErr = ctx.Err()
		}

		// Unregister tools
		if c.registry != nil {
			for _, name := range toolNames {
				err := c.registry.Unregister(name)
				if err != nil {
					errs = append(errs, fmt.Errorf("unregister tool %q: %w", name, err))
					c.logger.Warn("failed to unregister tool", "tool", name, "err", err)
				}
			}
		}

		// Clear internal state
		c.mu.Lock()
		c.tools = make(map[string]*toolWrapper)
		c.resetClientStateLocked()
		c.mu.Unlock()

		// Mark as fully shut down
		close(c.done)
		c.logger.Info("MCP client stopped", "serverID", c.serverID)

		// Combine all errors
		if len(errs) > 0 {
			if retErr != nil {
				errs = append(errs, retErr)
			}

			retErr = errors.Join(errs...)
		}
	})

	return retErr
}

// Close gracefully shuts down the client using a background context with default timeout.
// Satisfies io.Closer. Idempotent and safe to call multiple times.
func (c *clientImpl) Close() error {
	timeout := c.shutdownTimeout
	if timeout == 0 {
		timeout = 30 * time.Second // Default timeout
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return c.Shutdown(ctx)
}

// ServerID returns the unique server identifier.
func (c *clientImpl) ServerID() string {
	return c.serverID
}

// Started returns true if the client is currently running.
func (c *clientImpl) Started() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.startErr == nil && c.bgCtx != nil
}

// Done returns a channel that's closed when the client is fully shut down.
// Useful for tests and coordination.
func (c *clientImpl) Done() <-chan struct{} {
	return c.done
}

// isShutdown performs a non-blocking check on the done channel.
func (c *clientImpl) isShutdown() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

// connect creates and connects the MCP client.
func (c *clientImpl) connect(ctx context.Context) (*sdkmcp.ClientSession, *sdkmcp.Client, error) {
	mcpClient := sdkmcp.NewClient(&sdkmcp.Implementation{
		Name:    "redpanda-ai-agent-sdk",
		Title:   "Redpanda AI Agent SDK",
		Version: "v1.0.0",
	}, &sdkmcp.ClientOptions{
		KeepAlive: 30 * time.Second,
		ToolListChangedHandler: func(_ context.Context, _ *sdkmcp.ToolListChangedRequest) {
			select {
			case c.notifyCh <- struct{}{}:
			default:
			}
		},
	})

	transport, err := c.transportFactory()
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create transport: %w", err)
	}

	session, err := mcpClient.Connect(ctx, transport, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to connect to MCP server: %w", err)
	}

	return session, mcpClient, nil
}

// cleanupAfterFailedStart centralizes cleanup logic for failed Start attempts.
func (c *clientImpl) cleanupAfterFailedStart(session *sdkmcp.ClientSession) {
	var errs []error

	c.mu.RLock()

	if c.startErr != nil {
		errs = append(errs, c.startErr)
	}

	c.mu.RUnlock()

	if session != nil {
		closeErr := session.Close()
		if closeErr != nil {
			c.logger.Warn("failed to close session during startup cleanup", "err", closeErr)
			errs = append(errs, fmt.Errorf("session cleanup failed: %w", closeErr))
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.cancel != nil {
		c.cancel()
	}

	c.resetClientStateLocked()
	c.startErr = errors.Join(errs...)
}

// resetClientStateLocked resets connection-specific state.
// Caller must hold c.mu.
func (c *clientImpl) resetClientStateLocked() {
	c.replaceSessionLocked(nil, nil)
	c.bgCtx = nil
	c.cancel = nil
}

// replaceSessionLocked atomically swaps the session and signals waiting operations.
//
// Session Signaling Pattern:
// The channel close pattern ensures thread-safe notification: Capture old channel,
// install new session and channel, then close old channel. This ordering is critical:
//
//  1. Waiting goroutines hold a reference to the old channel (captured before unlock)
//  2. When the old channel closes, they wake up exactly once
//  3. They re-acquire the lock and see the new session state
//  4. They capture the new channel reference for the next wait cycle
//
// Reversing the order would cause a race where goroutines might capture and block on
// an already-closed channel, leading to spurious wakeups or missed signals.
//
// Caller must hold c.mu write lock.
func (c *clientImpl) replaceSessionLocked(session *sdkmcp.ClientSession, client *sdkmcp.Client) {
	old := c.sessionChanged

	c.session = session
	c.mcpClient = client
	c.sessionChanged = make(chan struct{})

	if old != nil {
		close(old)
	}
}

// waitForSession blocks until a live MCP session is available or the provided
// context (or client) is cancelled. Returns ErrClosed if the client is shutting
// down.
func (c *clientImpl) waitForSession(ctx context.Context) (*sdkmcp.ClientSession, context.Context, error) {
	for {
		ctxErr := ctx.Err()
		if ctxErr != nil {
			return nil, nil, ctxErr
		}

		c.mu.RLock()
		session := c.session
		ch := c.sessionChanged
		bgCtx := c.bgCtx
		closing := c.closing.Load()
		c.mu.RUnlock()

		if closing || bgCtx == nil {
			return nil, nil, ErrClosed
		}

		if session != nil {
			return session, bgCtx, nil
		}

		select {
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		case <-bgCtx.Done():
			return nil, nil, ErrClosed
		case <-ch:
			// Session state changed; re-check.
		}
	}
}

// sessionManagerLoop monitors the active session and automatically reconnects when lost.
// Runs for the client's lifetime, exiting only on shutdown.
//
// Design invariant: Only one sessionManagerLoop runs per client instance, guaranteed by
// wg.Add(1) in Start() under startOnce. This loop naturally re-fetches the session at
// the top of each iteration, so there's no need for defensive session-mismatch checks.
func (c *clientImpl) sessionManagerLoop() {
	defer c.wg.Done()

	for {
		// Get current session under lock
		c.mu.RLock()
		session := c.session
		bgCtx := c.bgCtx
		c.mu.RUnlock()

		// Exit if no session or shutting down
		if session == nil || bgCtx == nil {
			c.logger.Debug("sessionManagerLoop exiting: no session or context")
			return
		}

		// Wait for the current session to close (blocks until disconnect)
		waitErr := session.Wait()

		// Check if shutting down
		if c.closing.Load() || bgCtx.Err() != nil {
			return
		}

		if waitErr != nil {
			c.logger.Warn("MCP session closed with error", "serverID", c.serverID, "err", waitErr)
		} else {
			c.logger.Info("MCP session closed", "serverID", c.serverID)
		}

		// Clear disconnected session
		c.mu.Lock()
		c.replaceSessionLocked(nil, nil)
		c.mu.Unlock()

		// Perform blocking reconnect with exponential backoff
		newSession, newClient, err := c.performReconnect(bgCtx)
		if err != nil {
			// Context canceled during reconnect (client shutting down)
			c.logger.Info("reconnect terminated", "serverID", c.serverID, "err", err)
			return
		}

		// Install new session
		c.mu.Lock()

		if c.closing.Load() {
			// Shutdown initiated during reconnect
			c.mu.Unlock()

			_ = newSession.Close()

			return
		}

		c.replaceSessionLocked(newSession, newClient)
		c.mu.Unlock()

		c.logger.Info("reconnected to MCP server", "serverID", c.serverID)

		// Trigger background tool sync
		go func() {
			syncCtx, cancel := context.WithTimeout(bgCtx, 30*time.Second)
			defer cancel()

			err := c.SyncTools(syncCtx)
			if err != nil {
				c.logger.Warn("sync after reconnect failed", "serverID", c.serverID, "err", err)
			}
		}()
	}
}

// performReconnect attempts reconnection with exponential backoff until success or context cancellation.
func (c *clientImpl) performReconnect(ctx context.Context) (*sdkmcp.ClientSession, *sdkmcp.Client, error) {
	delay := reconnectInitialDelay
	attempt := 0

	for {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}

		attempt++
		c.logger.Info("attempting reconnect", "serverID", c.serverID, "attempt", attempt)

		// Don't use a timeout context - the SDK retains the passed context for the
		// connection's entire lifetime, using it for all HTTP requests including the
		// long-lived "hanging GET" for SSE streaming. A timeout would kill the connection.
		session, mcpClient, err := c.connect(ctx)
		if err == nil {
			return session, mcpClient, nil
		}

		c.logger.Warn("reconnect attempt failed", "serverID", c.serverID, "attempt", attempt, "err", err)

		// Apply exponential backoff with cap
		wait := min(delay, reconnectMaxDelay)

		select {
		case <-time.After(wait):
			// Continue to next attempt
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}

		// Double delay for next iteration
		if delay < reconnectMaxDelay {
			delay *= 2
			if delay > reconnectMaxDelay {
				delay = reconnectMaxDelay
			}
		}
	}
}

// beginOp safely starts a new operation, preventing WaitGroup race conditions.
// Returns an end function to call when the operation completes, or an error if
// the client is not started or is closing.
//
// This prevents the classic WaitGroup race where Add(1) is called concurrently
// with Wait() during shutdown, which would cause a panic.
func (c *clientImpl) beginOp() (func(), error) {
	// Fast checks without lock
	if c.closing.Load() {
		return nil, ErrClosed
	}

	c.mu.RLock()
	bgCtx := c.bgCtx
	c.mu.RUnlock()

	if bgCtx == nil {
		return nil, ErrNotStarted
	}

	// Safe to add now that we've verified we're not closing
	c.wg.Add(1)

	return func() { c.wg.Done() }, nil
}

// opContext creates a context that respects both the client's lifetime (bgCtx) and
// the caller's deadline (callerCtx). The returned context is cancelled when either
// the client shuts down or the caller's context is cancelled.
//
// The caller must call the returned cancel function to release resources.
//
// This pattern ensures operations can be cancelled by either:
//   - Client shutdown (bgCtx cancelled) - prevents operations after Close()
//   - Caller timeout/cancellation (callerCtx cancelled) - respects caller's deadline
//
// IMPORTANT: The context is derived from callerCtx to preserve trace context and
// other values, while still respecting bgCtx cancellation for client shutdown.
func opContext(bgCtx, callerCtx context.Context) (context.Context, context.CancelFunc) {
	// Derive from callerCtx to preserve trace context and other values
	ctx, cancel := context.WithCancel(callerCtx)

	// Also cancel when bgCtx is cancelled (client shutdown)
	stopBg := context.AfterFunc(bgCtx, cancel)

	// Wrap cancel to clean up both cancel and stopBg
	cleanup := func() {
		cancel()
		stopBg()
	}

	return ctx, cleanup
}

// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/twmb/franz-go/pkg/kgo"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// ErrRunNotFound is returned by Describe when the engine does not know the run.
var ErrRunNotFound = errors.New("durable: run not found")

// ErrNoEngineURL is returned by read operations when Config.EngineURL is empty.
var ErrNoEngineURL = errors.New("durable: Config.EngineURL is required for reads")

// Client starts, continues, signals and cancels runs. All writes go through the
// command topic; reads go through the engine's HTTP API.
type Client struct {
	cfg    Config
	topics Topics
	prod   producer
	cl     *kgo.Client
	http   *http.Client
}

// NewClient connects to the brokers in cfg.
func NewClient(cfg Config) (*Client, error) {
	cl, err := kgo.NewClient(cfg.kafkaOpts(kgo.ClientID("durable-client"))...)
	if err != nil {
		return nil, fmt.Errorf("durable: kafka client: %w", err)
	}

	return &Client{
		cfg:    cfg,
		topics: cfg.topics(),
		prod:   &kgoProducer{cl: cl},
		cl:     cl,
		http:   &http.Client{Timeout: 10 * time.Second},
	}, nil
}

// Close releases the Kafka client.
func (c *Client) Close() {
	if c.cl != nil {
		c.cl.Close()
	}
}

// StartOptions configures Client.Start.
type StartOptions struct {
	// RunID is caller-chosen and idempotent: starting an open run again is a
	// no-op, so upstream producers may retry freely.
	RunID string
	// Agent is the agent name registered on the worker.
	Agent string
	// Version pins the agent version. Empty selects one via the agent's rollout
	// record; if none exists the engine uses "v1".
	Version string
	// TaskQueue selects the worker pool. Empty selects via rollout, else Agent.
	TaskQueue string
	// Message is the first user message. Text is a shortcut for a text message.
	Message *llm.Message
	Text    string
	// Metadata seeds the session metadata.
	Metadata map[string]any

	RetryPolicy      *RetryPolicy
	ExecutionTimeout time.Duration
	// ResultTopic overrides the default results topic for this run.
	ResultTopic string
}

// Start starts a run.
func (c *Client) Start(ctx context.Context, opts StartOptions) error {
	if opts.RunID == "" || opts.Agent == "" {
		return errors.New("durable: StartOptions.RunID and Agent are required")
	}

	msg := opts.Message
	if msg == nil {
		m := llm.NewMessage(llm.RoleUser, llm.NewTextPart(opts.Text))
		msg = &m
	}

	cmd := NewCommand(CommandStartRun, opts.RunID)
	cmd.Agent = opts.Agent
	cmd.Version = opts.Version
	cmd.TaskQueue = opts.TaskQueue
	cmd.Message = msg
	cmd.Metadata = opts.Metadata
	cmd.RetryPolicy = opts.RetryPolicy
	cmd.ExecutionTimeout = opts.ExecutionTimeout
	cmd.ResultTopic = opts.ResultTopic

	return c.prod.Produce(ctx, c.topics.Commands(), opts.RunID, cmd)
}

// Continue appends a user message to a closed run and re-opens it, giving a
// durable multi-turn conversation whose whole history lives in the journal.
func (c *Client) Continue(ctx context.Context, runID string, msg llm.Message) error {
	cmd := NewCommand(CommandContinueRun, runID)
	cmd.Message = &msg

	return c.prod.Produce(ctx, c.topics.Commands(), runID, cmd)
}

// SendInput delivers external input to a run. If the run is waiting on a tool
// call that suspended with WaitForInput(name), the payload becomes that call's
// result and the run resumes; otherwise the input is buffered until it does.
func (c *Client) SendInput(ctx context.Context, runID, name string, payload any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("durable: marshal input payload: %w", err)
	}

	cmd := NewCommand(CommandSendInput, runID)
	cmd.Name = name
	cmd.Payload = raw

	return c.prod.Produce(ctx, c.topics.Commands(), runID, cmd)
}

// Cancel closes an open run with status cancelled.
func (c *Client) Cancel(ctx context.Context, runID, reason string) error {
	cmd := NewCommand(CommandCancelRun, runID)
	cmd.Reason = reason

	return c.prod.Produce(ctx, c.topics.Commands(), runID, cmd)
}

// SetRollout writes the rollout record for an agent to the config topic.
func (c *Client) SetRollout(ctx context.Context, agentName string, targets []RolloutTarget) error {
	rec := Rollout{Kind: "rollout", Agent: agentName, Targets: targets}

	return c.prod.Produce(ctx, c.topics.Config(), RolloutKey(agentName), rec)
}

// Describe fetches the run's current state from the engine.
func (c *Client) Describe(ctx context.Context, runID string) (*RunState, error) {
	if c.cfg.EngineURL == "" {
		return nil, ErrNoEngineURL
	}

	u := c.cfg.EngineURL + "/v1/runs/" + url.PathEscape(runID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
	if err != nil {
		return nil, fmt.Errorf("durable: build request: %w", err)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("durable: describe run: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return nil, ErrRunNotFound
	default:
		return nil, fmt.Errorf("durable: describe run: unexpected status %d", resp.StatusCode)
	}

	var st RunState
	if err := json.NewDecoder(resp.Body).Decode(&st); err != nil {
		return nil, fmt.Errorf("durable: decode run state: %w", err)
	}

	return &st, nil
}

// WaitResult polls Describe until the run closes or ctx ends.
func (c *Client) WaitResult(ctx context.Context, runID string, poll time.Duration) (*RunState, error) {
	if poll <= 0 {
		poll = 500 * time.Millisecond
	}

	t := time.NewTicker(poll)
	defer t.Stop()

	for {
		st, err := c.Describe(ctx, runID)
		if err != nil && !errors.Is(err, ErrRunNotFound) {
			return nil, err
		}

		if st != nil && st.Closed() {
			return st, nil
		}

		select {
		case <-ctx.Done():
			return st, ctx.Err()
		case <-t.C:
		}
	}
}

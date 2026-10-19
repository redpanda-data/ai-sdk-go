// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

package durable

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/redpanda-data/ai-sdk-go/llm"
)

// Command types written to the command topic. Client-originated commands
// address a run by id; worker-originated commands additionally carry the task
// token of the attempt that produced them.
const (
	CommandStartRun         = "start_run"
	CommandContinueRun      = "continue_run"
	CommandSendInput        = "send_input"
	CommandCancelRun        = "cancel_run"
	CommandAppendMessage    = "append_message"
	CommandResetMessages    = "reset_messages"
	CommandRecordToolResult = "record_tool_result"
	CommandRunSuspended     = "run_suspended"
	CommandAttemptFailed    = "attempt_failed"
	CommandRunCompleted     = "run_completed"
)

// Awaiting kinds.
const (
	AwaitInput = "input"
	AwaitTimer = "timer"
)

// Run statuses.
const (
	StatusRunning   = "running"
	StatusSuspended = "suspended"
	StatusRetrying  = "retrying"
	StatusCompleted = "completed"
	StatusFailed    = "failed"
	StatusCancelled = "cancelled"
)

// RetryPolicy governs how the engine retries a failed attempt.
type RetryPolicy struct {
	InitialInterval    time.Duration `json:"initial_interval"`
	BackoffCoefficient float64       `json:"backoff_coefficient"`
	MaxInterval        time.Duration `json:"max_interval"`
	// MaxAttempts counts failed attempts before the run fails. 0 means unlimited.
	MaxAttempts int `json:"max_attempts"`
}

// DefaultRetryPolicy is applied when a run does not specify one.
func DefaultRetryPolicy() RetryPolicy {
	return RetryPolicy{
		InitialInterval:    2 * time.Second,
		BackoffCoefficient: 2,
		MaxInterval:        5 * time.Minute,
		MaxAttempts:        5,
	}
}

// Backoff returns the delay before retrying after failures failed attempts.
func (p RetryPolicy) Backoff(failures int) time.Duration {
	if failures <= 0 {
		return 0
	}

	d := float64(p.InitialInterval)
	for range failures - 1 {
		d *= p.BackoffCoefficient
		if d >= float64(p.MaxInterval) {
			return p.MaxInterval
		}
	}

	if d > float64(p.MaxInterval) {
		return p.MaxInterval
	}

	return time.Duration(d)
}

// Exhausted reports whether failures failed attempts exhaust the policy.
func (p RetryPolicy) Exhausted(failures int) bool {
	return p.MaxAttempts > 0 && failures >= p.MaxAttempts
}

// Awaiting describes what a suspended run is waiting for.
type Awaiting struct {
	Kind string `json:"kind"`
	// ToolCallID identifies the tool call that suspended the run. The delivered
	// payload becomes that call's result.
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name,omitempty"`
	// Name is the input name (kind=input). Client.SendInput must use the same name.
	Name string `json:"name,omitempty"`
	// FireAt is when the timer fires (kind=timer).
	FireAt *time.Time `json:"fire_at,omitempty"`
}

// Command is the wire form of every record on the command topic. Fields not
// relevant to a command type are omitted.
type Command struct {
	Type      string    `json:"type"`
	CommandID string    `json:"command_id"`
	RunID     string    `json:"run_id"`
	SentAt    time.Time `json:"sent_at"`

	// start_run / continue_run
	Agent            string         `json:"agent,omitempty"`
	Version          string         `json:"version,omitempty"`
	TaskQueue        string         `json:"task_queue,omitempty"`
	Message          *llm.Message   `json:"message,omitempty"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	RetryPolicy      *RetryPolicy   `json:"retry_policy,omitempty"`
	ExecutionTimeout time.Duration  `json:"execution_timeout,omitempty"`
	ResultTopic      string         `json:"result_topic,omitempty"`

	// send_input
	Name    string          `json:"name,omitempty"`
	Payload json.RawMessage `json:"payload,omitempty"`

	// cancel_run
	Reason string `json:"reason,omitempty"`

	// worker commands
	TaskToken    string                `json:"task_token,omitempty"`
	MessageIndex int                   `json:"message_index,omitempty"`
	Messages     []llm.Message         `json:"messages,omitempty"`
	ToolResult   *llm.ToolResponsePart `json:"tool_result,omitempty"`
	Awaiting     *Awaiting             `json:"awaiting,omitempty"`
	Error        string                `json:"error,omitempty"`
	Retryable    *bool                 `json:"retryable,omitempty"`
	FinishReason string                `json:"finish_reason,omitempty"`
	Usage        *llm.TokenUsage       `json:"usage,omitempty"`
}

// Journal record types. The engine's state is a pure fold over these.
const (
	RecordRunStarted         = "run_started"
	RecordRunContinued       = "run_continued"
	RecordMessageAppended    = "message_appended"
	RecordMessagesReset      = "messages_reset"
	RecordTaskDispatched     = "task_dispatched"
	RecordLeaseExpired       = "lease_expired"
	RecordToolResultRecorded = "tool_result_recorded"
	RecordRunSuspended       = "run_suspended"
	RecordInputReceived      = "input_received"
	RecordInputDelivered     = "input_delivered"
	RecordTimerFired         = "timer_fired"
	RecordAttemptFailed      = "attempt_failed"
	RecordRunCompleted       = "run_completed"
	RecordRunFailed          = "run_failed"
	RecordRunCancelled       = "run_cancelled"
)

// Record is one journal entry. Seq is monotonic per run starting at 1.
type Record struct {
	Seq       uint64    `json:"seq"`
	Type      string    `json:"type"`
	RunID     string    `json:"run_id"`
	At        time.Time `json:"at"`
	CommandID string    `json:"command_id,omitempty"`

	Agent            string         `json:"agent,omitempty"`
	Version          string         `json:"version,omitempty"`
	TaskQueue        string         `json:"task_queue,omitempty"`
	Metadata         map[string]any `json:"metadata,omitempty"`
	RetryPolicy      *RetryPolicy   `json:"retry_policy,omitempty"`
	ExecutionTimeout time.Duration  `json:"execution_timeout,omitempty"`
	ResultTopic      string         `json:"result_topic,omitempty"`

	Message      *llm.Message  `json:"message,omitempty"`
	MessageIndex int           `json:"message_index,omitempty"`
	Messages     []llm.Message `json:"messages,omitempty"`

	TaskToken      string     `json:"task_token,omitempty"`
	Attempt        int        `json:"attempt,omitempty"`
	LeaseExpiresAt *time.Time `json:"lease_expires_at,omitempty"`

	ToolResult *llm.ToolResponsePart `json:"tool_result,omitempty"`
	Awaiting   *Awaiting             `json:"awaiting,omitempty"`
	Name       string                `json:"name,omitempty"`
	ToolCallID string                `json:"tool_call_id,omitempty"`
	Payload    json.RawMessage       `json:"payload,omitempty"`

	Error     string     `json:"error,omitempty"`
	Failures  int        `json:"failures,omitempty"`
	RetryAt   *time.Time `json:"retry_at,omitempty"`
	Exhausted bool       `json:"exhausted,omitempty"`

	FinishReason string          `json:"finish_reason,omitempty"`
	Usage        *llm.TokenUsage `json:"usage,omitempty"`
	Reason       string          `json:"reason,omitempty"`
}

// Task is what the engine writes to a task-queue topic. It carries everything a
// worker needs to resume the run without reading any other topic.
type Task struct {
	TaskToken      string         `json:"task_token"`
	RunID          string         `json:"run_id"`
	Agent          string         `json:"agent"`
	Version        string         `json:"version"`
	TaskQueue      string         `json:"task_queue"`
	Attempt        int            `json:"attempt"`
	DispatchedAt   time.Time      `json:"dispatched_at"`
	LeaseExpiresAt time.Time      `json:"lease_expires_at"`
	Messages       []llm.Message  `json:"messages"`
	Metadata       map[string]any `json:"metadata,omitempty"`
	// ToolResults holds results journaled by a previous attempt for tool calls
	// whose tool message was never appended. Keyed by tool call id.
	ToolResults map[string]*llm.ToolResponsePart `json:"tool_results,omitempty"`
	// Delivered holds external input and fired timers, keyed by the tool call
	// id that was waiting for them. The interceptor returns the payload as the
	// tool result instead of executing the tool.
	Delivered map[string]json.RawMessage `json:"delivered,omitempty"`
}

// PendingInput is an input that arrived before any tool waited for it.
type PendingInput struct {
	Name    string          `json:"name"`
	Payload json.RawMessage `json:"payload"`
}

// RunState is the engine's view of a run, also written to the compacted state
// topic after every transition.
type RunState struct {
	RunID            string          `json:"run_id"`
	Agent            string          `json:"agent"`
	Version          string          `json:"version"`
	TaskQueue        string          `json:"task_queue"`
	Status           string          `json:"status"`
	Awaiting         *Awaiting       `json:"awaiting,omitempty"`
	CurrentTaskToken string          `json:"current_task_token,omitempty"`
	Attempt          int             `json:"attempt"`
	Failures         int             `json:"failures"`
	RetryAt          *time.Time      `json:"retry_at,omitempty"`
	LeaseExpiresAt   *time.Time      `json:"lease_expires_at,omitempty"`
	Messages         []llm.Message   `json:"messages"`
	Metadata         map[string]any  `json:"metadata,omitempty"`
	PendingInputs    []PendingInput  `json:"pending_inputs,omitempty"`
	RetryPolicy      RetryPolicy     `json:"retry_policy"`
	ExecutionTimeout time.Duration   `json:"execution_timeout,omitempty"`
	ResultTopic      string          `json:"result_topic,omitempty"`
	FinishReason     string          `json:"finish_reason,omitempty"`
	Error            string          `json:"error,omitempty"`
	Usage            *llm.TokenUsage `json:"usage,omitempty"`
	JournalLen       uint64          `json:"journal_len"`
	StartedAt        time.Time       `json:"started_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
	ClosedAt         *time.Time      `json:"closed_at,omitempty"`

	// ToolResults and Delivered are carried on tasks; they are kept here so
	// the fold is self-contained.
	ToolResults map[string]*llm.ToolResponsePart `json:"tool_results,omitempty"`
	Delivered   map[string]json.RawMessage       `json:"delivered,omitempty"`
}

// Closed reports whether the run reached a terminal status.
func (s *RunState) Closed() bool {
	switch s.Status {
	case StatusCompleted, StatusFailed, StatusCancelled:
		return true
	default:
		return false
	}
}

// Result is published to the results topic when a run closes.
type Result struct {
	RunID        string          `json:"run_id"`
	Agent        string          `json:"agent"`
	Version      string          `json:"version"`
	Status       string          `json:"status"`
	FinishReason string          `json:"finish_reason,omitempty"`
	FinalMessage *llm.Message    `json:"final_message,omitempty"`
	Error        string          `json:"error,omitempty"`
	Usage        *llm.TokenUsage `json:"usage,omitempty"`
	Metadata     map[string]any  `json:"metadata,omitempty"`
	StartedAt    time.Time       `json:"started_at"`
	ClosedAt     time.Time       `json:"closed_at"`
}

// RolloutTarget is one weighted (version, task queue) pair for an agent.
type RolloutTarget struct {
	Version   string `json:"version"`
	TaskQueue string `json:"task_queue"`
	Weight    uint32 `json:"weight"`
}

// Rollout is the config-topic record under key "rollout/<agent>". Starts that
// do not pin a version pick a target deterministically by hashing the run id,
// so shifting weights from 5 to 100 percent is a single compacted write.
type Rollout struct {
	Kind    string          `json:"kind"`
	Agent   string          `json:"agent"`
	Targets []RolloutTarget `json:"targets"`
}

// RolloutKey is the config-topic key for an agent's rollout record.
func RolloutKey(agentName string) string {
	return "rollout/" + agentName
}

// TaskToken builds the token for an attempt.
func TaskToken(runID string, attempt int) string {
	return runID + ":" + itoa(attempt)
}

func itoa(i int) string {
	return strconv.Itoa(i)
}

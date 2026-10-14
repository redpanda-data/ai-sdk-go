// Copyright 2026 Redpanda Data, Inc.
//
// Licensed as a Redpanda Enterprise feature, governed by the Redpanda
// Community License Agreement included in the file licenses/RCL.md
//
// Use of this software requires a Redpanda Enterprise license.

// Package durable adds durable execution to agents built with this SDK, using
// Redpanda topics as the only durable log.
//
// A run is one agent invocation that must reach a terminal state even if the
// process executing it dies, is redeployed, or waits days for a human. The
// pieces:
//
//   - Client starts runs, continues conversations, delivers external input
//     (approvals, human answers, webhook payloads) and cancels runs by writing
//     commands to the command topic.
//   - Worker hosts one or more agent.Agent implementations, consumes tasks from
//     a task-queue topic and executes them. It journals every message the
//     agent appends and every tool result the agent produces, so a replacement
//     worker resumes from the last recorded step rather than from the start.
//   - engine.Engine (package durable/engine) is the controller. It folds the
//     journal into per-run state, dispatches tasks, enforces leases, retries
//     failed attempts with backoff, fires durable timers, matches external
//     input to waiting tool calls, and publishes results to a topic for
//     downstream consumers.
//
// Tools opt into durability by returning a suspension from Execute:
//
//	func (t approvalTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
//	    return nil, durable.WaitForInput("approval") // run parks until Client.SendInput(runID, "approval", payload)
//	}
//
// The run holds no worker while suspended. When input arrives (or the timer
// fires) the engine re-dispatches the run and the tool call completes with
// the delivered payload as its result. Add durable.NewInterceptor() to the
// agent's interceptors for this to work; without it tools run exactly as
// before and suspensions surface as ordinary tool errors.
//
// See docs/durable-execution.md for the design and the wire protocol.
package durable

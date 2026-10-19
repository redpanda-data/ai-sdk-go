# Durable execution for agents on Redpanda

An agent run is a multi-step process that can last minutes, hours, or weeks: the
model calls tools, tools call slow systems, and sometimes a human has to answer
before the run can continue. If the process hosting that run dies, the work is
normally lost. Retrying from scratch re-charges the model, re-executes tools that
already had side effects, and loses everything the run had established.

The `durable` package makes an agent run survive that. Redpanda topics are the
only stateful dependency: no database, no coordination service, no second
control plane. Every durable fact about a run is an append-only record keyed by
run id, so the durability, replication, retention, and multi-consumer fan-out
properties already relied on for event transport apply to agent execution too.

## Components

```
                 ┌────────────────┐  start / continue / input / cancel
   your service ─┤ durable.Client ├──────────────┐
                 └────────────────┘              │
                                                 ▼
  ┌─────────────────┐  fold        ┌──────────────────────┐
  │ durable.journal │◀─────────────┤    engine.Engine     │◀─ commands topic
  │  (append-only)  │─────────────▶│ state · leases ·     │
  └─────────────────┘  replay      │ retries · timers     │
                                   └──────┬─────┬─────────┘
                    dispatch task         │     │  snapshot         result
                                          ▼     ▼                     │
                             ┌──────────────────────┐   ┌─────────────▼──────┐
                             │ tasks.<queue> topic  │   │ state · results    │
                             └──────────┬───────────┘   └────────────────────┘
                                        ▼
                              ┌──────────────────┐
                              │  durable.Worker  │ hosts agent.Agent
                              └──────────────────┘
```

| Topic | Key | Cleanup | Holds |
|---|---|---|---|
| `<prefix>.commands` | run id | delete | every intent, from clients and workers |
| `<prefix>.journal` | run id | delete | the ordered record of what actually happened |
| `<prefix>.tasks.<queue>` | run id | delete | work handed to a worker pool |
| `<prefix>.state` | run id | compact | latest `RunState` per run, for dashboards |
| `<prefix>.results` | run id | delete | terminal results for downstream consumers |
| `<prefix>.config` | config key | compact | rollout records |

The command and journal topics must have the same partition count. The engine
consumes commands in a consumer group and, on assignment of partition *p*,
replays journal partition *p* to rebuild the state of the runs that live there.
Sharding and failover therefore come from consumer-group semantics rather than
custom leader election.

## What is durable

The engine's in-memory state is a pure fold over journal records. Restarting
every engine instance loses nothing. Three things get journaled:

- **Each message the agent appends** to the session, one record per message, in
  order. A resumed run replays the conversation exactly as it was.
- **Each tool result**, before the agent sees it. A crash after a tool ran but
  before its message landed does not re-run the tool: the next attempt is
  handed the recorded result.
- **Each suspension and each delivered input**, so a run waiting on a human
  holds no worker and cannot lose the answer.

Journaling is synchronous with `acks=all`: a returned nil error means the record
is durable. The engine writes the journal before any side effect that depends on
it, so if it dies between journaling a dispatch and writing the task, the lease
recorded in that journal entry expires and the task is re-dispatched.

## Programming model

Nothing about writing an agent changes. Register the interceptor and tools can
suspend:

```go
reg := tool.NewRegistry(tool.RegistryConfig{})
reg.Register(durable.InputTool("request_approval",
    "Ask a human operator to approve a refund before issuing it.", "approval"))
reg.Register(durable.SleepTool())

ag, _ := llmagent.New("support", systemPrompt, model,
    llmagent.WithTools(reg),
    llmagent.WithInterceptors(durable.NewInterceptor()),
)

w, _ := durable.NewWorker(cfg, "support", durable.WorkerOptions{MaxConcurrent: 8})
w.Register("support", "v1", ag)
w.Run(ctx)
```

Any tool can suspend its run by returning a suspension as its result:

```go
func (t refundTool) Execute(ctx context.Context, args json.RawMessage) (json.RawMessage, error) {
    if large(args) {
        return durable.WaitForInput("approval") // parks the run, frees the worker
    }
    return refund(ctx, args)
}
```

When `Client.SendInput(ctx, runID, "approval", payload)` arrives, the engine
re-dispatches the run and the tool call completes with that payload as its
result. Input that arrives before the run waits for it is buffered, not dropped,
so there is no race between an approval and the run reaching the approval step.
`durable.Sleep(d)` and `durable.SleepUntil(t)` work the same way through a
durable timer.

Without the interceptor those helpers degrade gracefully: the marker object is
just a tool result, and the run is not durable at the tool level.

## Failure handling

| Failure | What happens |
|---|---|
| Worker process dies mid-run | The task lease expires and another worker resumes from the last journaled message and tool result. Only unfinished tools re-execute. |
| Worker is shutting down | It journals nothing further and reports nothing, so the attempt is re-dispatched rather than recorded as a failure. |
| Model or tool infrastructure error | The attempt fails; the engine retries with exponential backoff per the run's `RetryPolicy`. |
| Retries exhausted, or a non-retryable failure | The run fails, a result is published, and the error is in `RunState.Error`. |
| Every engine instance dies | Each replays its journal partitions on restart and re-arms timers and leases. Suspended runs stay suspended; runnable runs are re-dispatched. |
| Agent or tool panics | Recovered and treated as a retryable attempt failure; the worker stays up. |
| Duplicate or redelivered command | Deduplicated by `command_id`, which is journaled and therefore survives replay. |
| A stale worker reporting after its lease expired | Rejected: only commands carrying the current task token are accepted. |

Tool execution is at-least-once, as in any activity system. A tool with side
effects should be idempotent or should key its effect on the tool call id.

## Versioning and gradual rollout

A run pins `(version, task_queue)` at start and the pin is journaled, so
deploying new worker code never changes a run already in flight. Publish a
rollout record and new runs split by weight:

```go
client.SetRollout(ctx, "support", []durable.RolloutTarget{
    {Version: "v1", TaskQueue: "support-v1", Weight: 95},
    {Version: "v2", TaskQueue: "support-v2", Weight: 5},
})
```

Selection hashes the run id, so a retried start lands on the same version.
Moving 5% to 100% is one compacted-topic write, and rollback is the same write
with the old weights.

## The streaming handoff

`engine.Trigger` consumes a business topic and starts one run per record,
deriving the run id from the record key. Starts are idempotent on run id, so
replaying or redelivering the source topic cannot start duplicate runs:

```go
tr := &engine.Trigger{Config: cfg, Source: "orders.placed", Agent: "fulfillment", TaskQueue: "fulfillment"}
tr.Run(ctx)
```

When a run closes, its result is published to the results topic, so any number
of downstream consumers react without knowing the engine exists. That pair of
topics is the entire contract between event streaming and agent execution: the
engine never subscribes to business topics itself.

The task topic is also the flow-control boundary. A worker bounds in-flight
tasks with `MaxConcurrent`; a burst beyond that accumulates as consumer lag
rather than hammering a rate-limited downstream API.

## Session journal store

`store/session/journal` is a `session.Store` backed by an append-only topic,
for services that want durable transcripts without the full engine. Unlike the
compacted `kvstore` store it appends one small record per new message instead of
rewriting the whole session, and `History` returns every record ever written for
a session, including messages that compaction later pruned from the working
context.

## Operating it

The engine serves a read-only HTTP API: `/healthz`, `/metrics` in Prometheus
text format, `GET /v1/runs/{id}`, and `GET /v1/runs?status=&limit=`. All
mutations go through the command topic; there is no write API. `Engine.Metrics`
exposes the same counters to an embedding service that has its own registry.

A run's full history is readable with ordinary tooling:

```bash
rpk topic consume durable.journal | jq 'select(.key == "run-42")'
```

## Try it

```bash
docker compose up -d                 # single-node Redpanda on :19092
cd examples/durable_agent
go run . engine &                    # controller + read API on :8080
go run . worker &                    # hosts the support agent
go run . start refund-1 "refund order 42"
go run . describe refund-1           # suspended, awaiting approval
go run . input refund-1 approval '{"approved":true,"by":"ops"}'
go run . wait refund-1
```

Kill the worker between `start` and `input` and start it again: the run picks up
where it left off.

## Licensing

Durable execution is a Redpanda Enterprise feature. The `durable`,
`durable/engine`, and `store/session/journal` packages are governed by the
Redpanda Community License Agreement in `licenses/RCL.md`, not the Apache 2.0
license that covers the rest of the SDK. Using them requires a Redpanda
Enterprise license. See `licenses/README.md` for the boundary; no Apache-licensed
package in this repository imports them.

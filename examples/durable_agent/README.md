# Durable agent execution

An agent run that survives worker crashes, engine restarts, and a human taking a
week to answer. Redpanda topics are the only state. See
[docs/durable-execution.md](../../docs/durable-execution.md) for the design.

The agent is a support agent with one durable tool: `request_approval` suspends
the run until an operator answers. With `OPENAI_API_KEY` set it uses a real
model; otherwise a scripted fake asks for approval whenever the request mentions
a refund.

## Run it

```bash
docker compose up -d          # from the repo root: Redpanda on localhost:19092

go run . engine               # terminal 1: controller + read API on :8080
go run . worker               # terminal 2: hosts the agent

go run . start refund-1 "refund order 42 for $120"
go run . describe refund-1    # status: suspended, awaiting input "approval"
go run . input refund-1 approval '{"approved":true,"by":"ops"}'
go run . wait refund-1        # prints the final assistant message
go run . continue refund-1 "thanks!"   # same run, durable multi-turn
```

## Prove it is durable

Kill the worker (Ctrl-C in terminal 2) while a run is in flight, then start it
again. The task lease expires, the run is re-dispatched, and it resumes from the
last journaled message: completed tool calls are not re-executed.

Kill the engine instead. On restart it replays the journal, so suspended runs
are still suspended and runnable runs are dispatched again.

Watch what is durable:

```bash
rpk topic consume durable.journal -o start | jq -c '{seq,type,run_id}'
rpk topic consume durable.results -o start | jq
```

## Start runs from a topic

```bash
go run . trigger orders.placed
rpk topic produce orders.placed -k order-9 <<< "refund order 9"
```

One run per record, with the run id derived from the record key, so replaying
the source topic does not start duplicate runs.

## Commands

| Command | Purpose |
|---|---|
| `engine` | run the controller (`DX_HTTP_ADDR`, default `:8080`) |
| `worker` | host the agent on task queue `support` |
| `start <id> <text>` | start a run |
| `continue <id> <text>` | add a turn to a closed run |
| `input <id> <name> <json>` | deliver external input |
| `cancel <id>` | cancel a run |
| `describe <id>` | print the run state |
| `wait <id>` | block until the run closes |
| `trigger <topic>` | start one run per record on a topic |

Environment: `DX_BROKERS` (`localhost:19092`), `DX_PREFIX` (`durable`),
`DX_ENGINE_URL` (`http://localhost:8080`).

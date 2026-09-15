# Licensing

The SDK is Apache 2.0 (see [`LICENSE`](../LICENSE)). Source files carry the
Apache header from [`header.txt`](../header.txt).

Durable execution is a Redpanda Enterprise feature and is licensed separately
under the Redpanda Community License Agreement in [`RCL.md`](RCL.md). Files
governed by it say so in their header and are confined to:

- `durable/` and `durable/engine/` — the client, worker, suspendable tools, and
  the execution controller
- `store/session/journal/` — the append-only-topic session store
- `examples/durable_agent/` — the runnable demo

Everything else in the repository, including the `agent`, `llm`, `tool`,
`runner`, `providers`, and `adapter` packages, stays Apache 2.0. Importing an
Apache-licensed package from enterprise code is fine; the reverse is not, so no
Apache-licensed package in this repository imports `durable`.

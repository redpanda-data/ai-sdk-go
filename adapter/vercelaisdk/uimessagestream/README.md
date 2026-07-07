# uimessagestream

Server half of the Vercel AI SDK [UI Message Stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) (v1) — the wire format `useChat` from `@ai-sdk/react` speaks — with **server-side sessions**. `Handler` exposes an `agent.Agent` (system prompt, tools, interceptors, agentic loop) as a chat resource backed by a `session.Store`, following the AI SDK's canonical [persistence pattern](https://ai-sdk.dev/docs/ai-sdk-ui/chatbot-message-persistence): the server loads the chat by id, appends the one posted message, runs the agent, and saves. Verified against `ai@7.0.6`.

## Quickstart

Server:

```go
provider, _ := openai.NewProvider(os.Getenv("OPENAI_API_KEY"))
model, _ := provider.NewModel(openai.ModelGPT5Mini)

reg := tool.NewRegistry(tool.RegistryConfig{})
_ = reg.Register(myTool)

ag, _ := llmagent.New("assistant", "You are a helpful assistant.", model,
    llmagent.WithTools(reg))

chat := uimessagestream.Handler(ag, session.NewInMemoryStore())

// Two-line mount: the exact path serves POST (run) and GET (list);
// the trailing-slash pattern serves /{id} (history, delete).
mux.Handle("/api/chat", http.StripPrefix("/api/chat", chat))
mux.Handle("/api/chat/", http.StripPrefix("/api/chat", chat))
```

Client (React) — send only the last message; the server owns the history:

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

const { messages, sendMessage, regenerate } = useChat({
  id: chatId, // stable chat id, e.g. from your router
  messages: initialMessages, // resume: fetched from GET /api/chat/{id}
  transport: new DefaultChatTransport({
    api: '/api/chat',
    prepareSendMessagesRequest: ({ id, messages, trigger, messageId }) => ({
      body:
        trigger === 'regenerate-message'
          ? { id, trigger, messageId }
          : { id, trigger, messageId, message: messages[messages.length - 1] },
    }),
  }),
});
```

The trimmed transport is an optimization, not a requirement: the default transport (full message list) also works — the server takes the last message and ignores the rest. Posted history is never trusted; the store is authoritative.

No client-side tool registry is needed: agent tools are runtime-discovered (MCP, subagents), so they stream as `dynamic-tool` parts, not statically-typed `tool-<name>` parts.

## Routes

Relative to the mount point:

| Route | Purpose |
|---|---|
| `POST /` | Run a turn. Body `{id, trigger, messageId, message}` (or default full body). Responds with the UI Message Stream (SSE). |
| `GET /{id}` | Chat history as UI messages (`{id, updatedAt, messages}`) — feed `messages` to `useChat` to resume after a reload. |
| `DELETE /{id}` | Delete the chat (204, idempotent). |
| `GET /` | List chats (`{chats, nextPageToken}`, `pageSize`/`pageToken` query params). Disabled (501) with `WithSessionKey`, see below. |

## Session semantics

- **Submit** (`trigger: "submit-message"`, the default): load the session by chat id — creating it on first use — append the posted user message, run the agent, save. Saves happen before the run (the user message is never lost), after every completed assistant message, and when the run ends, on a context that survives client disconnects: closing the tab mid-answer does not lose the turn.
- **Regenerate** (`trigger: "regenerate-message"`): truncate the stored history to the last user message and re-run without appending. `messageId` is accepted but unused — sessions persist model messages, which carry no UI ids, so v1 always regenerates the last answer (the `regenerate()` default). Editing a historical message is not supported for the same reason.
- **Storage shape**: sessions persist `llm.Message` — the SDK's canonical conversation shape, shared with the A2A adapter and the runner — and are projected to UI messages on read. This deliberately diverges from the AI SDK's persist-UIMessages advice; the cost is that UI message ids and custom data parts do not round-trip.
- **Concurrency**: concurrent POSTs to the same chat are serialized by an in-process keyed lock. Multi-replica deployments must serialize per session themselves (sticky routing, or a store-level guard).
- An interrupted run can leave a trailing assistant tool call in the session; `llmagent` heals it on the next submit (executing the tools before consulting the model), and the history projection closes it as `output-error` so the UI never shows an eternal spinner.

## Multi-tenancy

The client chooses its chat id, so by default the id is the storage key — fine for single-tenant or dev setups, not for shared deployments. `WithSessionKey` derives the storage key from the authenticated request:

```go
uimessagestream.Handler(ag, store, uimessagestream.WithSessionKey(
    func(r *http.Request, chatID string) (string, error) {
        user, err := authn(r) // your auth middleware/session
        if err != nil {
            return "", err // -> 403
        }
        return user.ID + "/" + chatID, nil
    }))
```

Every route resolves through it, so one user cannot read, run, or delete another's chat. Configuring it disables `GET /` (501): `session.Store.List` enumerates all storage keys and cannot be tenant-scoped here — expose your own list API (this is app-level in the AI SDK world too).

## Errors

Terminal and tool errors are sanitized to `"An error occurred."` by default so server-side detail never reaches the browser. Use `WithOnError` to surface specific text:

```go
uimessagestream.Handler(ag, store,
    uimessagestream.WithOnError(func(err error) string { return err.Error() }))
```

Failures before the stream starts (store errors, validation) are plain HTTP statuses; once streaming, the grammar guarantees the client never hangs — every terminal path (finish, error, abort, cancellation) closes open text spans, steps, and unresolved tool calls before the terminator, and at most one `error` chunk is emitted per stream.

## Security notes

- Posted history is ignored; only the last user message is appended. A client cannot forge prior assistant turns or tool results.
- Inbound `system` messages are rejected (only user messages are accepted); the agent owns its system prompt.

## Conformance

`task test:conformance` (requires node + bun) runs the suite in `conformance/` against the real `ai` TypeScript client (`DefaultChatTransport` + `readUIMessageStream`), driving `Handler` backed by `llmagent` over scripted fake models — including multi-turn accumulation over the trimmed transport, regenerate, history resume, and delete/list.

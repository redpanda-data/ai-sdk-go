# uimessagestream

Server half of the Vercel AI SDK [UI Message Stream protocol](https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol) (v1) — the wire format `useChat` from `@ai-sdk/react` speaks. `AgentHandler` exposes an `agent.Agent` (system prompt, tools, interceptors, agentic loop) over that protocol. Verified against `ai@7.0.6`.

## Quickstart

Server:

```go
provider, _ := openai.NewProvider(os.Getenv("OPENAI_API_KEY"))
model, _ := provider.NewModel(openai.ModelGPT5Mini)

reg := tool.NewRegistry(tool.RegistryConfig{})
_ = reg.Register(myTool)

ag, _ := llmagent.New("assistant", "You are a helpful assistant.", model,
    llmagent.WithTools(reg))

http.Handle("POST /api/chat", uimessagestream.AgentHandler(ag))
```

Client (React):

```tsx
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';

const { messages, sendMessage } = useChat({
  transport: new DefaultChatTransport({ api: '/api/chat' }),
});
```

That is the whole integration. No client-side tool registry is needed: agent tools are runtime-discovered (MCP, subagents), so they stream as `dynamic-tool` parts, not statically-typed `tool-<name>` parts.

## History model: client-authoritative

`useChat` re-sends the **full message list** on every request. The handler rebuilds the whole conversation from the posted messages, runs it against a fresh, non-persisted session, and throws the session away. Do not configure the client to trim history (`prepareSendMessagesRequest` sending only the last message) — the server would see a one-message conversation. The `useChat` default is correct as-is.

This matches the Vercel AI SDK's own model: the SDK has no server-side session or chat-list concept; history lives in the client and persistence is an application concern.

Consequences:

- Regenerate and edit-message work naturally — the client simply posts the rewritten list.
- Nothing is stored server-side. There is deliberately no session keyed on the client-supplied chat id: an unauthenticated, client-chosen key into a server-side store is a tenant-isolation hazard, and it fights `useChat`'s regenerate/edit semantics.
- The client does not persist anything either. `useChat` state is in-memory React state; it is gone on reload unless your application saves it. There is nothing to switch off.

## Chat id and session id

The client sends its chat id in the request body (`{"id": "...", "messages": [...]}` — set via `useChat({ id })`, otherwise generated). The handler uses it as the session ID of the throwaway session, purely for telemetry: transcripts and OTel spans from the agent's interceptor chain correlate across turns of the same chat. It is never a lookup key. When absent, a random id is used.

## Persisted chats

If you need server-listed, server-persisted conversations (chat history UI across devices/reloads), that is an application-level API, exactly as it would be with a Node/Vercel backend. Two options:

1. **Keep this handler, persist alongside.** Store chats in your own backend keyed by *authenticated* user + chat id. To resume, load the messages and seed `useChat({ messages: initialMessages })`; the conversation then continues client-authoritatively against `AgentHandler`.

2. **Own the history server-side.** Build a custom HTTP handler: authenticate, load the session from a `session.Store`, append the posted message, and stream with the exported `StreamAgent`. Only then does trimming the request make sense:

   ```ts
   new DefaultChatTransport({
     api: '/api/chat',
     prepareSendMessagesRequest: ({ id, messages }) => ({
       body: { id, message: messages[messages.length - 1] },
     }),
   });
   ```

   Your handler owns authorization of `id` against the caller. `session.Store.List` provides paginated session summaries for a chat-list endpoint; the Vercel AI SDK itself has no such API — listing is always yours.

## Errors

Terminal and tool errors are sanitized to `"An error occurred."` by default so server-side detail never reaches the browser. Use `WithOnError` to surface specific text:

```go
uimessagestream.AgentHandler(ag,
    uimessagestream.WithOnError(func(err error) string { return err.Error() }))
```

The stream grammar guarantees the client never hangs: every terminal path (finish, error, abort, cancellation) closes open text spans, steps, and unresolved tool calls before the terminator, and at most one `error` chunk is emitted per stream.

## Security notes

- Inbound `system` messages are dropped; the agent owns its system prompt.
- Incomplete tool calls in re-sent history (state `input-streaming`/`input-available`) are dropped: a browser must not be able to forge an unexecuted tool call that the agent's recovery path would run.

## Conformance

`task test:conformance` (requires node + bun) runs the suite in `conformance/` against the real `ai` TypeScript client (`DefaultChatTransport` + `readUIMessageStream`), driving `AgentHandler` backed by `llmagent` over scripted fake models.

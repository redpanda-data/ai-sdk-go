# vercelaisdk_chat

Full-stack example: an ai-sdk-go agent (OpenAI + one tool) served over the Vercel AI SDK UI Message Stream protocol with server-side sessions, consumed by a vanilla `useChat` React client.

What it demonstrates end to end:

- streaming answers and dynamic tool calls rendered live,
- server-owned history: the client sends only the last message per turn (`prepareSendMessagesRequest`), yet the model remembers earlier turns,
- resume: reload the page or switch chats — history comes back from `GET /api/chat/{id}`,
- chat list, delete, and regenerate.

## Run

Terminal 1 — the Go server (port 8080):

```bash
OPENAI_API_KEY=sk-... go run .
```

Terminal 2 — the web client (Vite dev server, proxies `/api` to the Go server):

```bash
cd web
bun install   # or: npm install
bun run dev   # or: npm run dev
```

Open http://localhost:5173, ask "what's the weather in Berlin?", watch the tool call stream in. Then say "and in Paris?" — the server remembered the context. Reload the page and click the chat in the sidebar: the conversation resumes from the server.

## Notes

- The client is intentionally vanilla Vercel AI SDK: `useChat` + `DefaultChatTransport` from `ai`/`@ai-sdk/react`, no wrappers. `web/src/App.tsx` is the whole integration.
- Sessions live in `session.NewInMemoryStore()` — they survive reloads, not server restarts. Swap in a persistent `session.Store` for real use.
- See `adapter/vercelaisdk/uimessagestream/README.md` for the protocol surface, multi-tenancy (`WithSessionKey`), and error-sanitization semantics.

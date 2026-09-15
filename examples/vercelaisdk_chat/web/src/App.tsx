import { useCallback, useEffect, useState } from 'react';
import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport, type UIMessage } from 'ai';

// The canonical server-session transport from the adapter README: only the
// last message goes over the wire; the Go server owns the history.
const transport = new DefaultChatTransport({
  api: '/api/chat',
  prepareSendMessagesRequest: ({ id, messages, trigger, messageId }) => ({
    body:
      trigger === 'regenerate-message'
        ? { id, trigger, messageId }
        : { id, trigger, messageId, message: messages[messages.length - 1] },
  }),
});

interface ChatSummary {
  id: string;
  updatedAt?: string;
}

export function App() {
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [selected, setSelected] = useState<string>(() => crypto.randomUUID());
  const [initialMessages, setInitialMessages] = useState<UIMessage[] | null>([]);

  const refreshChats = useCallback(async () => {
    const res = await fetch('/api/chat');
    if (res.ok) {
      const body = await res.json();
      setChats(body.chats ?? []);
    }
  }, []);

  useEffect(() => {
    void refreshChats();
  }, [refreshChats]);

  // Resume: load the server-side history and seed useChat with it.
  const openChat = useCallback(async (id: string) => {
    setInitialMessages(null); // loading
    const res = await fetch(`/api/chat/${id}`);
    const messages = res.ok ? (await res.json()).messages : [];
    setInitialMessages(messages);
    setSelected(id);
  }, []);

  const newChat = useCallback(() => {
    setInitialMessages([]);
    setSelected(crypto.randomUUID());
  }, []);

  const deleteChat = useCallback(
    async (id: string) => {
      await fetch(`/api/chat/${id}`, { method: 'DELETE' });
      await refreshChats();
      if (id === selected) newChat();
    },
    [refreshChats, selected, newChat],
  );

  return (
    <div className="layout">
      <aside>
        <button className="primary" onClick={newChat}>
          New chat
        </button>
        <ul>
          {chats.map((c) => (
            <li key={c.id} className={c.id === selected ? 'active' : ''}>
              <button className="chat-link" onClick={() => void openChat(c.id)}>
                {c.id.slice(0, 8)}…
              </button>
              <button className="delete" title="Delete chat" onClick={() => void deleteChat(c.id)}>
                ×
              </button>
            </li>
          ))}
        </ul>
      </aside>
      {initialMessages === null ? (
        <main className="loading">loading…</main>
      ) : (
        // key remounts useChat when switching chats.
        <Chat key={selected} id={selected} initialMessages={initialMessages} onTurnDone={refreshChats} />
      )}
    </div>
  );
}

function Chat({
  id,
  initialMessages,
  onTurnDone,
}: {
  id: string;
  initialMessages: UIMessage[];
  onTurnDone: () => Promise<void>;
}) {
  const [input, setInput] = useState('');

  const { messages, sendMessage, regenerate, stop, status, error } = useChat({
    id,
    messages: initialMessages,
    transport,
    onFinish: () => void onTurnDone(),
  });

  const busy = status === 'submitted' || status === 'streaming';

  return (
    <main>
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.role}`}>
            <span className="role">{m.role}</span>
            {m.parts.map((part, i) => {
              switch (part.type) {
                case 'text':
                  return <p key={i}>{part.text}</p>;
                case 'reasoning':
                  return (
                    <p key={i} className="reasoning">
                      {part.text}
                    </p>
                  );
                case 'dynamic-tool':
                  return (
                    <div key={i} className="tool">
                      <code>
                        {part.toolName}({JSON.stringify(part.input)}) → {part.state}
                      </code>
                      {part.state === 'output-available' && <pre>{JSON.stringify(part.output, null, 2)}</pre>}
                      {part.state === 'output-error' && <pre className="error">{part.errorText}</pre>}
                    </div>
                  );
                default:
                  return null;
              }
            })}
          </div>
        ))}
        {error && <div className="error">error: {error.message}</div>}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (!input.trim() || busy) return;
          void sendMessage({ text: input });
          setInput('');
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the weather somewhere…"
          autoFocus
        />
        <button className="primary" type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
        <button type="button" onClick={() => (busy ? void stop() : void regenerate())} disabled={messages.length === 0}>
          {busy ? 'Stop' : 'Regenerate'}
        </button>
        <span className="status">{status}</span>
      </form>
    </main>
  );
}

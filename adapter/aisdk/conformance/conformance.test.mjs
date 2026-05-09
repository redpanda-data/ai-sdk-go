// Conformance tests: verify that the Go aisdk adapter produces SSE streams
// that the real Vercel AI SDK v6 TypeScript client parses correctly.
//
// This uses DefaultChatTransport + readUIMessageStream, which is the exact
// same parsing path that useChat/AbstractChat uses internally.

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import {
  DefaultChatTransport,
  readUIMessageStream,
} from 'ai';

const SERVER_BIN = process.env.TEST_SERVER_BIN;
const SERVER_PORT = process.env.TEST_SERVER_PORT;

let serverProcess;
let baseUrl;

/**
 * Send a chat request via DefaultChatTransport and collect the final
 * assembled UIMessage from the stream.
 */
async function sendChat(endpoint, userMessages) {
  const transport = new DefaultChatTransport({
    api: `${baseUrl}${endpoint}`,
  });

  // Build messages in AI SDK v6 format
  const messages = userMessages.map((msg, i) => ({
    id: `msg-${i}`,
    role: msg.role || 'user',
    parts: [{ type: 'text', text: msg.text }],
  }));

  const chunkStream = await transport.sendMessages({
    chatId: 'test-chat',
    messages,
    abortSignal: AbortSignal.timeout(10000),
    trigger: 'submit-message',
  });

  // Use readUIMessageStream to assemble the message exactly as the real
  // client does. Collect all intermediate states; the last one is final.
  const errors = [];
  const messageStream = readUIMessageStream({
    stream: chunkStream,
    onError: (err) => errors.push(err),
  });

  let finalMessage;
  for await (const msg of messageStream) {
    finalMessage = msg;
  }

  return { message: finalMessage, errors };
}

// ── Test lifecycle ──────────────────────────────────────────────────

before(async () => {
  if (SERVER_PORT) {
    // Server already running externally
    baseUrl = `http://127.0.0.1:${SERVER_PORT}`;
  } else if (SERVER_BIN) {
    // Start the Go test server
    baseUrl = await new Promise((resolve, reject) => {
      serverProcess = spawn(SERVER_BIN, ['-port', '0'], {
        stdio: ['ignore', 'pipe', 'pipe'],
      });

      let stdout = '';
      serverProcess.stdout.on('data', (data) => {
        stdout += data.toString();
        const match = stdout.match(/LISTEN_ADDR=(.+)/);
        if (match) {
          resolve(`http://${match[1]}`);
        }
      });

      serverProcess.stderr.on('data', (data) => {
        process.stderr.write(data);
      });

      serverProcess.on('error', reject);
      serverProcess.on('exit', (code) => {
        if (!baseUrl) reject(new Error(`server exited with code ${code}`));
      });

      setTimeout(() => reject(new Error('server startup timeout')), 10000);
    });

    // Wait for health check
    for (let i = 0; i < 50; i++) {
      try {
        const resp = await fetch(`${baseUrl}/health`);
        if (resp.ok) break;
      } catch {
        // not ready yet
      }
      await new Promise((r) => setTimeout(r, 100));
    }
  } else {
    throw new Error(
      'Set TEST_SERVER_BIN or TEST_SERVER_PORT to run conformance tests'
    );
  }
});

after(() => {
  if (serverProcess) {
    serverProcess.kill('SIGTERM');
  }
});

// ── Tests ───────────────────────────────────────────────────────────

describe('AI SDK conformance', () => {
  it('Test 1: simple text response', async () => {
    const { message, errors } = await sendChat('/api/simple', [
      { text: 'hi' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');
    assert.equal(message.role, 'assistant');

    // Find text parts
    const textParts = message.parts.filter((p) => p.type === 'text');
    assert.ok(textParts.length > 0, 'should have at least one text part');

    // The final assembled text should be "Hello, world!"
    const fullText = textParts.map((p) => p.text).join('');
    assert.equal(fullText, 'Hello, world!');

    // Each text part should be in "done" state (stream finished)
    for (const tp of textParts) {
      assert.equal(tp.state, 'done', 'text part state should be done');
    }
  });

  it('Test 2: streaming chunks', async () => {
    const { message, errors } = await sendChat('/api/streaming', [
      { text: 'test' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');
    assert.equal(message.role, 'assistant');

    const textParts = message.parts.filter((p) => p.type === 'text');
    const fullText = textParts.map((p) => p.text).join('');
    assert.equal(fullText, 'Hello streaming world');

    // Verify all text parts are done
    for (const tp of textParts) {
      assert.equal(tp.state, 'done');
    }
  });

  it('Test 3: error handling', async () => {
    const { message, errors } = await sendChat('/api/error', [
      { text: 'fail' },
    ]);

    // The SSE error chunk should trigger the onError callback
    assert.ok(errors.length > 0, 'should have received at least one error');

    // Verify the error message contains rate limit text
    const errorTexts = errors.map((e) => e.message || String(e));
    const hasRateLimit = errorTexts.some((t) => t.includes('rate limit'));
    assert.ok(
      hasRateLimit,
      `expected rate limit error, got: ${errorTexts.join(', ')}`
    );
  });

  it('Test 4: multi-turn context with v6 parts format', async () => {
    const { message, errors } = await sendChat('/api/echo-context', [
      { text: 'My name is Alice' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');

    const textParts = message.parts.filter((p) => p.type === 'text');
    const fullText = textParts.map((p) => p.text).join('');

    // The echo-context endpoint returns JSON with the messages it received.
    const received = JSON.parse(fullText);
    assert.ok(Array.isArray(received), 'echoed messages should be an array');
    assert.ok(received.length >= 1, 'should have at least 1 message');

    // Find the user message
    const userMsg = received.find((m) => m.role === 'user');
    assert.ok(userMsg, 'should have a user message');
    assert.equal(userMsg.text, 'My name is Alice');
  });

  it('Test 5: system prompt', async () => {
    const { message, errors } = await sendChat('/api/system', [
      { text: 'say something' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');

    const textParts = message.parts.filter((p) => p.type === 'text');
    const fullText = textParts.map((p) => p.text).join('');

    // The system endpoint echoes back "System: <system prompt>"
    assert.ok(
      fullText.includes('You are a pirate'),
      `expected system prompt text, got: "${fullText}"`
    );
  });

  it('Test 6: multi-turn conversation context', async () => {
    // Send multiple messages including assistant turn
    const transport = new DefaultChatTransport({
      api: `${baseUrl}/api/echo-context`,
    });

    const messages = [
      {
        id: 'msg-0',
        role: 'user',
        parts: [{ type: 'text', text: 'hello' }],
      },
      {
        id: 'msg-1',
        role: 'assistant',
        parts: [
          { type: 'step-start' },
          { type: 'text', text: 'hi there', state: 'done' },
        ],
      },
      {
        id: 'msg-2',
        role: 'user',
        parts: [{ type: 'text', text: 'how are you?' }],
      },
    ];

    const chunkStream = await transport.sendMessages({
      chatId: 'test-multi',
      messages,
      abortSignal: AbortSignal.timeout(10000),
      trigger: 'submit-message',
    });

    const errors = [];
    const msgStream = readUIMessageStream({
      stream: chunkStream,
      onError: (err) => errors.push(err),
    });

    let finalMessage;
    for await (const msg of msgStream) {
      finalMessage = msg;
    }

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);

    const textParts = finalMessage.parts.filter((p) => p.type === 'text');
    const fullText = textParts.map((p) => p.text).join('');
    const received = JSON.parse(fullText);

    // Should have 3 messages: user, assistant, user
    assert.equal(received.length, 3, `expected 3 messages, got ${received.length}`);
    assert.equal(received[0].role, 'user');
    assert.equal(received[0].text, 'hello');
    assert.equal(received[1].role, 'assistant');
    assert.equal(received[1].text, 'hi there');
    assert.equal(received[2].role, 'user');
    assert.equal(received[2].text, 'how are you?');
  });
});

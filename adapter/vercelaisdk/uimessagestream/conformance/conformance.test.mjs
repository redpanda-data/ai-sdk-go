// Conformance tests: verify that the Go aisdk adapter produces SSE streams
// that the real Vercel AI SDK v7 TypeScript client parses correctly.
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

  // Build messages in AI SDK v7 format
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

  it('Test 4: multi-turn context with v7 parts format', async () => {
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

  it('Test 7: reasoning then text', async () => {
    const { message, errors } = await sendChat('/api/reasoning', [
      { text: 'think' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');

    const reasoningParts = message.parts.filter((p) => p.type === 'reasoning');
    assert.ok(reasoningParts.length > 0, 'should have a reasoning part');
    assert.equal(reasoningParts[0].text, 'Thinking about the question.');
    for (const rp of reasoningParts) {
      assert.equal(rp.state, 'done', 'reasoning part should be done');
    }

    const textParts = message.parts.filter((p) => p.type === 'text');
    const fullText = textParts.map((p) => p.text).join('');
    assert.equal(fullText, 'The answer is 42.');
    for (const tp of textParts) {
      assert.equal(tp.state, 'done', 'text part should be done');
    }

    // Reasoning must precede text in the assembled message.
    const firstReasoning = message.parts.findIndex((p) => p.type === 'reasoning');
    const firstText = message.parts.findIndex((p) => p.type === 'text');
    assert.ok(
      firstReasoning < firstText,
      'reasoning part should come before text part'
    );
  });

  it('Test 8: tool call then final answer', async () => {
    const { message, errors } = await sendChat('/api/tools', [
      { text: 'weather in SF?' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);
    assert.ok(message, 'should have received a message');

    // The real client assembles the tool call into a tool-<name> part that
    // transitions to output-available with the executor's output.
    const toolParts = message.parts.filter((p) =>
      p.type?.startsWith('tool-')
    );
    assert.equal(toolParts.length, 1, 'should have exactly one tool part');
    assert.equal(toolParts[0].type, 'tool-getWeather');
    assert.equal(
      toolParts[0].state,
      'output-available',
      'tool part should reach output-available'
    );
    assert.deepEqual(toolParts[0].output, {
      temperature: '72F',
      conditions: 'sunny',
    });

    const fullText = message.parts
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('');
    assert.equal(fullText, 'It is sunny and 72F in San Francisco.');
  });

  it('Test 9: tool executor error then recovery', async () => {
    const { message, errors } = await sendChat('/api/tool-error', [
      { text: 'weather in SF?' },
    ]);

    // tool-output-error is a normal part-state transition, not a stream error.
    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);

    const toolParts = message.parts.filter((p) =>
      p.type?.startsWith('tool-')
    );
    assert.equal(toolParts.length, 1, 'should have one tool part');
    assert.equal(
      toolParts[0].state,
      'output-error',
      'tool part should reach output-error'
    );
    assert.equal(toolParts[0].errorText, 'weather service unavailable');

    const fullText = message.parts
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('');
    assert.equal(fullText, 'I could not fetch the weather.');
  });

  it('Test 10: multi-step tool calls reset text ids across steps', async () => {
    const { message, errors } = await sendChat('/api/multistep', [
      { text: 'run both steps' },
    ]);

    assert.equal(errors.length, 0, `unexpected errors: ${errors}`);

    const toolParts = message.parts.filter((p) =>
      p.type?.startsWith('tool-')
    );
    assert.equal(toolParts.length, 2, 'should have two tool parts');
    for (const tp of toolParts) {
      assert.equal(tp.state, 'output-available');
    }

    // Final text streamed in a later step (after multiple finish-step resets)
    // must still assemble without "missing text part" errors.
    const fullText = message.parts
      .filter((p) => p.type === 'text')
      .map((p) => p.text)
      .join('');
    assert.equal(fullText, 'Both steps are complete.');
  });

  it('Test 11: mid-stream error closes the open text part', async () => {
    const { message, errors } = await sendChat('/api/midstream-error', [
      { text: 'go' },
    ]);

    // The mid-stream failure surfaces exactly one error to onError.
    assert.ok(errors.length >= 1, 'should surface the mid-stream error');
    const errorTexts = errors.map((e) => e.message || String(e));
    assert.ok(
      errorTexts.some((t) => t.includes('server error')),
      `expected server error, got: ${errorTexts.join(', ')}`
    );

    // The partial text emitted before the failure must be closed (state done):
    // the adapter emits text-end before the error chunk, so the client does not
    // throw "missing text part".
    const textParts = message.parts.filter((p) => p.type === 'text');
    assert.ok(textParts.length > 0, 'should have partial text');
    for (const tp of textParts) {
      assert.equal(tp.state, 'done', 'partial text part should be closed');
    }
  });
});

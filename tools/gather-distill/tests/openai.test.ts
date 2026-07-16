import { expect, test } from "bun:test";
import {
  fromOpenAiMessage,
  sendOpenAiMessage,
  stripThinkBlocks,
  toOpenAiMessages,
  toOpenAiTools,
} from "../src/openai.ts";
import { GATHER_TOOLS } from "../src/tools.ts";
import type { TrajectoryMessage } from "../src/types.ts";

test("translates canonical tool turns to OpenAI tool_calls and tool messages", () => {
  const trajectory: TrajectoryMessage[] = [
    { role: "user", content: "Find the implementation." },
    {
      role: "assistant",
      content: [
        { type: "text", text: "I will inspect the file." },
        { type: "tool_use", id: "call-read", name: "read", input: { filePath: "src/lib.ts", startLine: 1, endLine: 3 } },
      ],
    },
    {
      role: "user",
      content: [{ type: "tool_result", tool_use_id: "call-read", content: "1: export const answer = 42;\n" }],
    },
  ];

  const messages = toOpenAiMessages("same system bytes", trajectory);
  expect(messages).toEqual([
    { role: "system", content: "same system bytes" },
    { role: "user", content: "Find the implementation." },
    {
      role: "assistant",
      content: "I will inspect the file.",
      tool_calls: [
        {
          id: "call-read",
          type: "function",
          function: { name: "read", arguments: '{"filePath":"src/lib.ts","startLine":1,"endLine":3}' },
        },
      ],
    },
    { role: "tool", tool_call_id: "call-read", content: "1: export const answer = 42;\n" },
  ]);
  expect(fromOpenAiMessage(messages[2])).toEqual({
    content: [
      { type: "text", text: "I will inspect the file." },
      { type: "tool_use", id: "call-read", name: "read", input: { filePath: "src/lib.ts", startLine: 1, endLine: 3 } },
    ],
    thinkingTokens: 0,
  });
  expect(toOpenAiTools(GATHER_TOOLS)[0]).toEqual({
    type: "function",
    function: {
      name: GATHER_TOOLS[0]!.name,
      description: GATHER_TOOLS[0]!.description,
      parameters: GATHER_TOOLS[0]!.input_schema,
    },
  });
});

test("strips think blocks and records their fallback token count", () => {
  expect(stripThinkBlocks('<think>first hidden thought\nsecond thought</think>\n{"answer":true}')).toEqual({
    text: '\n{"answer":true}',
    thinkingTokens: 5,
  });
  expect(fromOpenAiMessage({ content: "<think>one two</think>{\"answer\":true}" })).toEqual({
    content: [{ type: "text", text: '{"answer":true}' }],
    thinkingTokens: 2,
  });
});

test("sends a local OpenAI request without authentication and preserves tool_choice none", async () => {
  const originalFetch = globalThis.fetch;
  const requests: Array<{ input: RequestInfo | URL; init: RequestInit | undefined }> = [];
  globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
    requests.push({ input, init });
    return Response.json({
      choices: [
        {
          finish_reason: "stop",
          message: { content: '<think>private reasoning</think>{"interpretation":"ok","scope":[],"snippets":[],"omissions":[]}' },
        },
      ],
      usage: { prompt_tokens: 12, completion_tokens: 8 },
    });
  }) as typeof fetch;

  try {
    const response = await sendOpenAiMessage(
      {
        model: "MiniCPM5-1B-Q4_K_M",
        max_tokens: 128,
        system: "byte-identical system",
        messages: [{ role: "user", content: "Answer with JSON." }],
        tools: GATHER_TOOLS,
        tool_choice: { type: "none" },
      },
      { baseUrl: "http://127.0.0.1:8080/v1", transientRetries: 0 },
    );

    expect(requests).toHaveLength(1);
    expect(String(requests[0]!.input)).toBe("http://127.0.0.1:8080/v1/chat/completions");
    expect(new Headers(requests[0]!.init?.headers).has("authorization")).toBeFalse();
    const body = JSON.parse(String(requests[0]!.init?.body));
    expect(body).toMatchObject({
      model: "MiniCPM5-1B-Q4_K_M",
      messages: [
        { role: "system", content: "byte-identical system" },
        { role: "user", content: "Answer with JSON." },
      ],
      tool_choice: "none",
    });
    expect(body.tools).toHaveLength(9);
    expect(response.content).toEqual([
      { type: "text", text: '{"interpretation":"ok","scope":[],"snippets":[],"omissions":[]}' },
    ]);
    expect(response.usage).toMatchObject({ input_tokens: 12, output_tokens: 8, thinking_tokens: 2 });
  } finally {
    globalThis.fetch = originalFetch;
  }
});


test("supports explicit hosted-endpoint authentication and temperature pinning", async () => {
  const originalFetch = globalThis.fetch;
  let request: RequestInit | undefined;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    request = init;
    return Response.json({
      choices: [{ finish_reason: "stop", message: { content: '{"ok":true}' } }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
  }) as typeof fetch;
  try {
    await sendOpenAiMessage(
      {
        model: "gpt-5.6",
        max_tokens: 32,
        temperature: 0,
        system: "judge",
        messages: [{ role: "user", content: "test" }],
      },
      { baseUrl: "https://judge.example.invalid/v1", apiKey: "secret", transientRetries: 0 },
    );
    expect(new Headers(request?.headers).get("authorization")).toBe("Bearer secret");
    expect(JSON.parse(String(request?.body))).toMatchObject({ model: "gpt-5.6", temperature: 0 });
  } finally {
    globalThis.fetch = originalFetch;
  }
});

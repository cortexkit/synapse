import { expect, test } from "bun:test";
import { GATHER_SYSTEM_PROMPT_V10 } from "../prompts/gather-system-v10.ts";
import { convertRow, validateOpenAiExample } from "../train/convert.ts";
import type { BankedRow } from "../src/types.ts";

function fixtureRow(): BankedRow {
  return {
    request: "Find both symbols.",
    repo_full: "example/repo",
    repo_sha: "0123456789abcdef0123456789abcdef01234567",
    tags: { request_class: "cross_module_trace", expected_difficulty: 2, specificity: "high" },
    full_trajectory: [
      { role: "user", content: "Request:\nFind both symbols.\n\nScope:\n(caller did not restrict scope)" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "Checking both." },
          { type: "tool_use", id: "call-a", name: "read", input: { filePath: "a.ts", startLine: 1 } },
          { type: "tool_use", id: "call-b", name: "read", input: { filePath: "b.ts", startLine: 2 } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call-a", content: "1: export const a = 1;\n" },
          { type: "tool_result", tool_use_id: "call-b", content: "2: export const b = 2;\n" },
        ],
      },
      { role: "user", content: "finish now", synthetic: "budget_finalize" },
      {
        role: "assistant",
        content: [{ type: "text", text: "analysis kept verbatim\n```json\n{\"interpretation\":\"done\",\"scope\":[],\"snippets\":[],\"omissions\":[]}\n```" }],
      },
    ],
    final_json: { interpretation: "done", scope: [], snippets: [], omissions: [] },
    budget_outcome: "budget_finalize",
    input_tokens: 1,
    output_tokens: 1,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 0,
    model: "fixture",
    account: "fixture",
    ts: "2026-01-01T00:00:00Z",
    valid: true,
  };
}

test("converts parallel Anthropic tools and preserves the emitted final text", () => {
  const example = convertRow(fixtureRow());
  expect(example.messages[0]).toEqual({ role: "system", content: GATHER_SYSTEM_PROMPT_V10 });
  expect(example.messages[2]).toMatchObject({
    role: "assistant",
    tool_calls: [
      { id: "call-a", function: { arguments: '{"filePath":"a.ts","startLine":1}' } },
      { id: "call-b", function: { arguments: '{"filePath":"b.ts","startLine":2}' } },
    ],
  });
  expect(example.messages.slice(3, 5).map((message) => message.tool_call_id)).toEqual(["call-a", "call-b"]);
  expect(example.messages.at(-1)?.content).toStartWith("analysis kept verbatim\n```json");
  expect(example.tools).toHaveLength(9);
  expect(validateOpenAiExample(example)).toMatchObject({ toolCalls: 2, toolResults: 2 });
});

test("rejects orphaned OpenAI tool messages", () => {
  const example = convertRow(fixtureRow());
  example.messages.splice(3, 1);
  expect(() => validateOpenAiExample(example)).toThrow("before tool results");
});

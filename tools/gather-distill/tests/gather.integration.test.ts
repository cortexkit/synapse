import { expect, test } from "bun:test";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { AccountLease, AccountPool } from "../src/auth.ts";
import { runGatherJob } from "../src/gather.ts";
import { AftClient } from "../src/tools.ts";
import { recordedProcessFactory } from "./support/aft-process.ts";

async function git(repo: string, ...args: string[]): Promise<string> {
  const process = Bun.spawn(["git", "-C", repo, ...args], { stdout: "pipe", stderr: "pipe" });
  const [status, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  if (status !== 0) throw new Error(stderr);
  return stdout.trim();
}

test("banks and validates a forced-finalize trajectory with an unchanged toolset", async () => {
  const repo = await mkdtemp(join(tmpdir(), "gather-integration-"));
  await mkdir(join(repo, "src"));
  await writeFile(join(repo, "src", "lib.ts"), "export const answer = 42;\n");
  await git(repo, "init", "-q");
  await git(repo, "config", "user.name", "Gather Test");
  await git(repo, "config", "user.email", "gather@example.invalid");
  await git(repo, "add", "src/lib.ts");
  await git(repo, "commit", "-qm", "fixture");
  const sha = await git(repo, "rev-parse", "HEAD");
  await writeFile(
    join(repo, ".gather-corpus-manifest.json"),
    JSON.stringify({ fullName: "owner/repo", sha, language: "TypeScript", size_mb: 0.001 }),
  );

  const lease: AccountLease = {
    credential: { name: "dry-key", secret: "not-a-real-key", kind: "api_key" },
    released: false,
  };
  const pool = {
    acquire: async () => lease,
    release: (value: AccountLease) => {
      value.released = true;
    },
    coolDown: async () => {},
  } as unknown as AccountPool;

  const { factory } = recordedProcessFactory((toolRequest, process) => {
    if (toolRequest.command === "configure") {
      process.respond({ id: toolRequest.id, success: true });
      return;
    }
    process.respond({ id: toolRequest.id, success: true, text: "1: export const answer = 42;\n[AFT E0 W0 | D0 U0 C0 | T0]" });
  });
  const aftClient = new AftClient({ processFactory: factory });
  const originalFetch = globalThis.fetch;
  const bodies: Array<Record<string, unknown>> = [];
  let request = 0;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    request += 1;
    if (request === 1) {
      return Response.json({
        content: Array.from({ length: 4 }, (_, index) => ({
          type: "tool_use",
          id: `tool-${index}`,
          name: "read",
          input: { filePath: "src/lib.ts", startLine: 1, endLine: 1 },
        })),
        stop_reason: "tool_use",
        usage: { input_tokens: 100, output_tokens: 20 },
      });
    }
    return Response.json({
      content: [
        {
          type: "text",
          text: JSON.stringify({
            interpretation: "Locate the answer constant.",
            scope: ["src/lib.ts"],
            snippets: [{ path: "src/lib.ts", startLine: 1, endLine: 1, why: "Defines the answer." }],
            omissions: [],
          }),
        },
      ],
      stop_reason: "end_turn",
      usage: { input_tokens: 120, output_tokens: 30 },
    });
  }) as typeof fetch;

  try {
    const row = await runGatherJob(
      {
        dir: repo,
        request: "Where is the answer constant defined?",
        tags: { request_class: "feature_orientation", expected_difficulty: 1, specificity: "high" },
      },
      { pool, model: "cheap-dry-run-model", maxSteps: 5, inlineValidate: true, aftClient },
    );

    expect(row.valid).toBeTrue();
    expect(row.budget_outcome).toBe("budget_finalize");
    expect(row.full_trajectory.some((message) => message.synthetic === "budget_finalize")).toBeTrue();
    const toolResultTurn = row.full_trajectory.find((message) => message.role === "user" && Array.isArray(message.content));
    const toolResults = Array.isArray(toolResultTurn?.content) ? toolResultTurn.content : [];
    expect(toolResults[0]).toMatchObject({ content: "1: export const answer = 42;\n[AFT E0 W0 | D0 U0 C0 | T0]" });
    expect(bodies).toHaveLength(2);
    expect(bodies[1].tools).toEqual(bodies[0].tools);
    expect(bodies[1].tool_choice).toEqual({ type: "none" });
  } finally {
    globalThis.fetch = originalFetch;
    await aftClient.close();
  }
});


test("OpenAI finalization stops when a local server ignores tool_choice none", async () => {
  const repo = await mkdtemp(join(tmpdir(), "gather-openai-finalize-"));
  await mkdir(join(repo, "src"));
  await writeFile(join(repo, "src", "lib.ts"), "export const answer = 42;\n");
  await git(repo, "init", "-q");
  await git(repo, "config", "user.name", "Gather Test");
  await git(repo, "config", "user.email", "gather@example.invalid");
  await git(repo, "add", "src/lib.ts");
  await git(repo, "commit", "-qm", "fixture");
  const sha = await git(repo, "rev-parse", "HEAD");
  await writeFile(
    join(repo, ".gather-corpus-manifest.json"),
    JSON.stringify({ fullName: "owner/repo", sha, language: "TypeScript", size_mb: 0.001 }),
  );

  const { factory, processes } = recordedProcessFactory((toolRequest, process) => {
    if (toolRequest.command === "configure") {
      process.respond({ id: toolRequest.id, success: true });
      return;
    }
    process.respond({ id: toolRequest.id, success: true, text: "1: export const answer = 42;\n[AFT E0 W0 | D0 U0 C0 | T0]" });
  });
  const aftClient = new AftClient({ processFactory: factory });
  const originalFetch = globalThis.fetch;
  const bodies: Array<Record<string, unknown>> = [];
  let request = 0;
  globalThis.fetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    bodies.push(JSON.parse(String(init?.body)));
    request += 1;
    if (request === 1) {
      const toolCalls = Array.from({ length: 3 }, (_, index) => ({
        id: `call-${request}-${index}`,
        type: "function",
        function: { name: "read", arguments: '{"filePath":"src/lib.ts","startLine":1,"endLine":1}' },
      }));
      return Response.json({
        choices: [{ finish_reason: "tool_calls", message: { content: null, tool_calls: toolCalls } }],
        usage: { prompt_tokens: 100, completion_tokens: 20 },
      });
    }
    return Response.json({
      choices: [{ finish_reason: "stop", message: { content: '<function name="read"><param name="filePath">src/lib.ts</param></function>' } }],
      usage: { prompt_tokens: 100, completion_tokens: 20 },
    });
  }) as typeof fetch;

  try {
    const row = await runGatherJob(
      {
        dir: repo,
        request: "Where is the answer constant defined?",
        tags: { request_class: "feature_orientation", expected_difficulty: 1, specificity: "high" },
      },
      {
        backend: "openai",
        baseUrl: "http://127.0.0.1:8080/v1",
        model: "local-test-model",
        maxSteps: 1,
        aftClient,
      },
    );

    expect(row.budget_outcome).toBe("budget_finalize");
    expect(row.final_json).toBeNull();
    expect(row.reason).toContain("tool_choice:none");
    expect(row.account).toBe("local");
    expect(bodies).toHaveLength(2);
    expect(bodies[1]!.tools).toEqual(bodies[0]!.tools);
    expect(bodies[1]!.tool_choice).toBe("none");
    expect(processes[0]!.requests.filter((entry) => entry.command === "tool_call")).toHaveLength(3);
  } finally {
    globalThis.fetch = originalFetch;
    await aftClient.close();
  }
});

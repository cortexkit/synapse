import { createHash } from "node:crypto";
import { CLAUDE_CODE_IDENTITY } from "@cortexkit/anthropic-auth-core";
import { expect, test } from "bun:test";
import { sendMessage } from "../src/anthropic.ts";
import { loadGatherSystemPrompt } from "../prompts/gather-system.ts";
import { GATHER_TOOLS } from "../src/tools.ts";

test("OAuth messages impersonate Claude Code and preserve the gather prompt", async () => {
  const prompt = loadGatherSystemPrompt();
  const originalFetch = globalThis.fetch;
  const requests: RequestInit[] = [];
  globalThis.fetch = (async (_input: RequestInfo | URL, requestInit?: RequestInit) => {
    if (requestInit) requests.push(requestInit);
    return Response.json({
      content: [{ type: "text", text: "ok" }],
      stop_reason: "end_turn",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
  }) as typeof fetch;

  try {
    await sendMessage(
      { name: "oauth-test", secret: "test-oauth-token", kind: "oauth" },
      {
        model: "claude-sonnet-5",
        max_tokens: 16,
        system: prompt,
        messages: [{ role: "user", content: "Reply with ok." }],
        tools: GATHER_TOOLS,
        tool_choice: { type: "none" },
      },
    );

    expect(requests).toHaveLength(1);
    const oauthInit = requests[0]!;
    const headers = new Headers(oauthInit.headers);
    expect(headers.get("authorization")).toBe("Bearer test-oauth-token");
    expect(headers.get("anthropic-beta")).toContain("oauth-2025-04-20");
    expect(headers.has("x-api-key")).toBeFalse();

    const bodyText = String(oauthInit.body);
    const body = JSON.parse(bodyText);
    // system layout: [billing, identity, gatherPrompt] with a 5m cache
    // breakpoint on the stable trailing prompt block.
    expect(body.system).toEqual([
      { type: "text", text: expect.stringContaining("x-anthropic-billing-header: ") },
      { type: "text", text: CLAUDE_CODE_IDENTITY },
      { type: "text", text: prompt, cache_control: { type: "ephemeral" } },
    ]);
    expect(body.system[0].text).toMatch(/\bcch=[0-9a-f]{5};/);
    expect(bodyText).not.toContain("cch=00000;");
    // tools preserved with a 5m cache breakpoint on the last (stable) tool.
    expect(body.tools).toEqual(
      GATHER_TOOLS.map((tool, index) =>
        index === GATHER_TOOLS.length - 1 ? { ...tool, cache_control: { type: "ephemeral" } } : tool,
      ),
    );
    // the growing message prefix is anchored on the last user block.
    const lastMessage = body.messages[body.messages.length - 1];
    const lastBlock = lastMessage.content[lastMessage.content.length - 1];
    expect(lastBlock.cache_control).toEqual({ type: "ephemeral" });
    expect(body.tool_choice).toEqual({ type: "none" });
    expect(createHash("sha256").update(prompt).digest("hex")).toBe(
      "c87b1aa778e1bbb742f8bb076bb11944d26ed19ddcdb1a1241e07cfbff2707b1",
    );

    await sendMessage(
      { name: "api-test", secret: "test-api-key", kind: "api_key" },
      {
        model: "claude-sonnet-5",
        max_tokens: 16,
        system: prompt,
        messages: [{ role: "user", content: "Reply with ok." }],
      },
    );
    const apiInit = requests[1]!;
    const apiHeaders = new Headers(apiInit.headers);
    expect(apiHeaders.get("x-api-key")).toBe("test-api-key");
    expect(apiHeaders.has("authorization")).toBeFalse();
    // the api-key path also caches: the system string becomes a single cached block.
    expect(JSON.parse(String(apiInit.body)).system).toEqual([
      { type: "text", text: prompt, cache_control: { type: "ephemeral" } },
    ]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

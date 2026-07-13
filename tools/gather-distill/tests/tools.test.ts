import { expect, test } from "bun:test";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AftClient, AftClientPool, canonicalRepoRoot } from "../src/tools.ts";
import { recordedProcessFactory, type RecordedRequest } from "./support/aft-process.ts";

const LEXICAL_SEARCH_DISCLOSURE = "Semantic search unavailable; returning lexical-only fallback results.";

async function fixtureRepo(prefix = "gather-aft-tools-"): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix));
}

function respondToConfigure(request: RecordedRequest, process: { respond(value: Record<string, unknown>): void }): boolean {
  if (request.command !== "configure") return false;
  process.respond({ id: request.id, success: true });
  return true;
}

test("lexical-only canary forwards recorded AFT response.text verbatim and sends a bare search wire name", async () => {
  const repo = await fixtureRepo();
  const responses = (await readFile(join(import.meta.dir, "fixtures/aft-search-lexical.ndjson"), "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
  const { factory, processes } = recordedProcessFactory((request, process) => {
    process.respond(responses[process.requests.length - 1]);
  });
  const client = new AftClient({ processFactory: factory });

  try {
    const text = await client.call(repo, "search", { query: "lexicalCanary" });
    expect(text).toBe(responses[1].text);
    expect(text).toContain(LEXICAL_SEARCH_DISCLOSURE);
    expect(processes[0].requests).toMatchObject([
      { id: "1", command: "configure", harness: "opencode" },
      { id: "2", command: "tool_call", name: "search", arguments: { query: "lexicalCanary" } },
    ]);
  } finally {
    await client.close();
  }
});

test("correlates concurrent tool results by NDJSON id", async () => {
  const repo = await fixtureRepo();
  const queued: RecordedRequest[] = [];
  const { factory } = recordedProcessFactory((request, process) => {
    if (respondToConfigure(request, process)) return;
    queued.push(request);
    if (queued.length !== 2) return;
    process.respond({ id: queued[1].id, success: true, text: "second result" });
    process.respond({ id: queued[0].id, success: true, text: "first result" });
  });
  const client = new AftClient({ processFactory: factory });

  try {
    const [first, second] = await Promise.all([
      client.call(repo, "search", { query: "first" }),
      client.call(repo, "search", { query: "second" }),
    ]);
    expect(first).toBe("first result");
    expect(second).toBe("second result");
  } finally {
    await client.close();
  }
});

test("respawns once after a process dies and retries the interrupted call", async () => {
  const repo = await fixtureRepo();
  const { factory, processes } = recordedProcessFactory(
    (request, process) => {
      if (respondToConfigure(request, process)) return;
      process.exit(1);
    },
    (request, process) => {
      if (respondToConfigure(request, process)) return;
      process.respond({ id: request.id, success: true, text: "recovered result" });
    },
  );
  const client = new AftClient({ processFactory: factory });

  try {
    await expect(client.call(repo, "read", { filePath: "README.md" })).resolves.toBe("recovered result");
    expect(processes).toHaveLength(2);
    expect(processes[1].requests.map((request) => request.command)).toEqual(["configure", "tool_call"]);
  } finally {
    await client.close();
  }
});

test("reuses configure for one repo and swaps it when a pooled process moves repos", async () => {
  const repoA = await fixtureRepo("gather-aft-a-");
  const repoB = await fixtureRepo("gather-aft-b-");
  const { factory, processes } = recordedProcessFactory((request, process) => {
    if (respondToConfigure(request, process)) return;
    process.respond({ id: request.id, success: true, text: String(request.name) });
  });
  const client = new AftClient({ processFactory: factory });
  const pool = new AftClientPool(1, { clientFactory: () => client });

  try {
    await pool.withClient(repoA, (leased) => leased.call(repoA, "glob", { pattern: "README*" }));
    await pool.withClient(repoA, (leased) => leased.call(repoA, "read", { filePath: "README.md" }));
    await pool.withClient(repoB, (leased) => leased.call(repoB, "glob", { pattern: "README*" }));
    await pool.withClient(repoA, (leased) => leased.call(repoA, "glob", { pattern: "README*" }));
    const configuredRoots = processes[0].requests
      .filter((request) => request.command === "configure")
      .map((request) => request.project_root);
    expect(processes).toHaveLength(1);
    expect(configuredRoots).toEqual([await canonicalRepoRoot(repoA), await canonicalRepoRoot(repoB), await canonicalRepoRoot(repoA)]);
  } finally {
    await pool.close();
  }
});

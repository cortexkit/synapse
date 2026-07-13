import { expect, test } from "bun:test";
import { mkdtemp, readFile, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  AFT_STORAGE_DIR,
  AftClient,
  AftClientPool,
  AftWarmupCoordinator,
  canonicalRepoRoot,
  executeTool,
} from "../src/tools.ts";
import { recordedProcessFactory, type RecordedRequest } from "./support/aft-process.ts";

const LEXICAL_SEARCH_DISCLOSURE = "Semantic search unavailable; returning lexical-only fallback results.";
const DEGRADED_SEARCH =
  "Semantic search is not enabled.\nSemantic search unavailable; lexical-only fallback returned 0 result(s). [semantic: unavailable]\nSearch status: fully degraded; partial/incomplete.";
const WARM_LEXICAL_SEARCH = `Semantic search is not enabled.\n${LEXICAL_SEARCH_DISCLOSURE}\n\nsrc/tools.ts\n1: export function ready() {}`;

async function fixtureRepo(prefix = "gather-aft-tools-"): Promise<string> {
  const repo = await mkdtemp(join(tmpdir(), prefix));
  await writeFile(join(repo, "README.md"), "# fixture\n");
  return repo;
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
      {
        id: "1",
        command: "configure",
        harness: "opencode",
        storage_dir: AFT_STORAGE_DIR,
        config: [
          {
            tier: "user",
            source: "inline:gather-distill",
            doc: JSON.stringify({ semantic_search: false, search_index: true }),
          },
        ],
      },
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
    process.respond({ id: queued[1].id, success: true, text: `${String(queued[1].arguments?.query)} result` });
    process.respond({ id: queued[0].id, success: true, text: `${String(queued[0].arguments?.query)} result` });
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

test("ignores interleaved AFT notifications until the id-matched response", async () => {
  const repo = await fixtureRepo();
  const frames = (await readFile(join(import.meta.dir, "fixtures/aft-notification-interleave.ndjson"), "utf8"))
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line) as Record<string, unknown>);
  const { factory } = recordedProcessFactory((request, process) => {
    if (respondToConfigure(request, process)) return;
    for (const frame of frames) process.respond(frame);
  });
  const client = new AftClient({ processFactory: factory });

  try {
    await expect(client.call(repo, "search", { query: "notifications" })).resolves.toBe("id-matched tool result");
  } finally {
    await client.close();
  }
});

test("rejects every path-bearing argument that escapes the configured repository", async () => {
  const repo = await fixtureRepo();
  let processStarts = 0;
  const client = new AftClient({
    processFactory: () => {
      processStarts += 1;
      throw new Error("foreign bytes must never be requested");
    },
  });
  const calls: Array<[string, Record<string, unknown>]> = [
    ["outline", { target: "/", files: true }],
    ["zoom", { targets: [{ filePath: "../foreign.ts", symbol: "foreign" }] }],
    ["zoom", { url: "https://example.invalid/foreign", symbols: "Foreign" }],
    ["read", { filePath: "/tmp/foreign.txt" }],
    ["search", { query: "foreign", path: "/tmp" }],
    ["callgraph", { op: "trace_to_symbol", toFile: "/tmp/foreign.ts" }],
    ["inspect", { scope: ["/"] }],
  ];

  try {
    for (const [name, input] of calls) {
      const result = await executeTool(repo, name, input, client);
      expect(result.ok).toBeFalse();
      expect(result.output).toContain("path outside project");
      expect(result.output).not.toContain("foreign bytes");
    }
    expect(processStarts).toBe(0);
  } finally {
    await client.close();
  }
});

test("rejects a repository-relative symlink that resolves outside the repository", async () => {
  const repo = await fixtureRepo();
  const foreign = await fixtureRepo("gather-aft-foreign-");
  await writeFile(join(foreign, "secret.txt"), "foreign filesystem bytes\n");
  await symlink(foreign, join(repo, "escape"), "dir");
  const client = new AftClient({
    processFactory: () => {
      throw new Error("AFT must not receive escaped symlink paths");
    },
  });

  try {
    const result = await executeTool(repo, "read", { filePath: "escape/secret.txt" }, client);
    expect(result.ok).toBeFalse();
    expect(result.output).toContain("path outside project");
    expect(result.output).not.toContain("foreign filesystem bytes");
  } finally {
    await client.close();
  }
});

test("search warm-up retries degraded results until the index returns ranked hits", async () => {
  const repo = await fixtureRepo();
  let searches = 0;
  const coldThenWarm = ["Semantic search is not enabled.", DEGRADED_SEARCH, DEGRADED_SEARCH, WARM_LEXICAL_SEARCH];
  const { factory, processes } = recordedProcessFactory((request, process) => {
    if (respondToConfigure(request, process)) return;
    searches += 1;
    process.respond({ id: request.id, success: true, text: coldThenWarm[searches - 1] ?? WARM_LEXICAL_SEARCH });
  });
  const client = new AftClient({ processFactory: factory });
  const warmups = new AftWarmupCoordinator({ timeoutMs: 1_000, initialBackoffMs: 1, maxBackoffMs: 1 });

  try {
    const result = await warmups.ensureWarmed(repo, client);
    expect(result).toMatchObject({ ok: true, searchAttempts: 4 });
    expect(result.timedOut).toBeUndefined();
    expect(searches).toBe(4);
    expect(processes[0].requests.filter((request) => request.name === "search")).toHaveLength(4);
  } finally {
    await client.close();
  }
});

test("search warm-up returns a warning and proceeds after its bounded timeout", async () => {
  const repo = await fixtureRepo();
  let searches = 0;
  const { factory } = recordedProcessFactory((request, process) => {
    if (respondToConfigure(request, process)) return;
    searches += 1;
    process.respond({ id: request.id, success: true, text: DEGRADED_SEARCH });
  });
  const client = new AftClient({ processFactory: factory });
  const warmups = new AftWarmupCoordinator({ timeoutMs: 15, initialBackoffMs: 2, maxBackoffMs: 2 });

  try {
    const result = await warmups.ensureWarmed(repo, client);
    expect(result.ok).toBeTrue();
    expect(result.timedOut).toBeTrue();
    expect(result.warning).toContain("proceeding with cold search");
    expect(searches).toBeGreaterThan(0);
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

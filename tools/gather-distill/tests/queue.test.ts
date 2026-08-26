import { describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { balanceJobs, pendingJobs, pendingJobsAfterCrash } from "../src/queue.ts";
import type { BankedRow, GatherJob, LedgerEntry } from "../src/types.ts";
import { stableJobId } from "../src/utils.ts";

const tags = {
  request_class: "bug_investigation" as const,
  expected_difficulty: 3,
  specificity: "med" as const,
};

function job(dir: string, request: string, difficulty = 3): GatherJob {
  return { dir, request, tags: { ...tags, expected_difficulty: difficulty } };
}

describe("generation queue", () => {
  test("resume never regenerates a completed dir/request pair", () => {
    const jobs = [job("/repos/a", "Where is request A handled?"), job("/repos/b", "Where is request B handled?")];
    const completed: LedgerEntry = {
      job_id: stableJobId(jobs[0].dir, jobs[0].request),
      dir: jobs[0].dir,
      request: jobs[0].request,
      tags,
      outcome: "banked",
      duration_ms: 10,
      input_tokens: 100,
      output_tokens: 20,
      account: "test-account-1",
      valid: true,
      ts: new Date(0).toISOString(),
    };

    expect(pendingJobs(jobs, [completed])).toEqual([jobs[1]]);
  });

  test("recovers a row banked just before a ledger-write crash", async () => {
    const root = await mkdtemp(join(tmpdir(), "gather-resume-"));
    const repo = join(root, "owner__a");
    await mkdir(repo);
    const sha = "a".repeat(40);
    await writeFile(
      join(repo, ".gather-corpus-manifest.json"),
      JSON.stringify({ fullName: "owner/a", sha, language: "Rust", size_mb: 1 }),
    );
    const completedJob = job(repo, "Where is crash recovery handled?");
    const row: BankedRow = {
      request: completedJob.request,
      repo_full: "owner/a",
      repo_sha: sha,
      tags,
      full_trajectory: [],
      final_json: { interpretation: "x", scope: [], snippets: [], omissions: [] },
      budget_outcome: "natural",
      input_tokens: 1,
      output_tokens: 1,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      thinking_tokens: 0,
      model: "test",
      account: "test-account-1",
      ts: new Date(0).toISOString(),
      valid: true,
    };

    expect(await pendingJobsAfterCrash([completedJob], [], [row])).toEqual([]);
  });

  test("round-robins cells and keeps highest-value question first within each cell", async () => {
    const root = await mkdtemp(join(tmpdir(), "gather-queue-"));
    const a = join(root, "owner__a");
    const b = join(root, "owner__b");
    await Promise.all([mkdir(a), mkdir(b)]);
    await Promise.all([
      writeFile(join(a, ".gather-corpus-manifest.json"), JSON.stringify({ fullName: "owner/a", sha: "a".repeat(40), language: "Rust", size_mb: 1 })),
      writeFile(join(b, ".gather-corpus-manifest.json"), JSON.stringify({ fullName: "owner/b", sha: "b".repeat(40), language: "Rust", size_mb: 1 })),
    ]);
    const ordered = await balanceJobs([
      job(a, "low value question in a?", 1),
      job(a, "high value question in a?", 5),
      job(b, "only question in b?", 3),
    ]);

    expect(ordered.map((item) => item.request)).toEqual([
      "high value question in a?",
      "only question in b?",
      "low value question in a?",
    ]);
  });
});

import type { BankedRow, GatherJob, LedgerEntry } from "./types.ts";
import { loadManifest } from "./repo.ts";
import { stableJobId } from "./utils.ts";

function valueScore(job: GatherJob): number {
  const specificity = { low: 0, med: 1, high: 2 }[job.tags.specificity] ?? 0;
  return job.tags.expected_difficulty * 10 + specificity;
}

export async function balanceJobs(jobs: GatherJob[]): Promise<GatherJob[]> {
  const cells = new Map<string, GatherJob[]>();
  for (const job of jobs) {
    const manifest = await loadManifest(job.dir);
    job.tags.language ??= manifest.language;
    const cell = `${manifest.language}\0${job.tags.request_class}\0${manifest.fullName}`;
    const group = cells.get(cell) ?? [];
    group.push(job);
    cells.set(cell, group);
  }
  for (const group of cells.values()) {
    group.sort((a, b) => valueScore(b) - valueScore(a) || a.request.localeCompare(b.request));
  }
  const keys = [...cells.keys()].sort();
  const balanced: GatherJob[] = [];
  let emitted = true;
  while (emitted) {
    emitted = false;
    for (const key of keys) {
      const job = cells.get(key)?.shift();
      if (!job) continue;
      balanced.push(job);
      emitted = true;
    }
  }
  return balanced;
}

export function pendingJobs(jobs: GatherJob[], ledger: LedgerEntry[]): GatherJob[] {
  const completedIds = new Set(
    ledger
      .filter((entry) => entry.outcome === "banked" || entry.outcome === "rejected")
      .map((entry) => entry.job_id || stableJobId(entry.dir, entry.request)),
  );
  return jobs.filter((job) => !completedIds.has(stableJobId(job.dir, job.request)));
}

export async function pendingJobsAfterCrash(
  jobs: GatherJob[],
  ledger: LedgerEntry[],
  rows: BankedRow[],
): Promise<GatherJob[]> {
  const pending = pendingJobs(jobs, ledger);
  const completedRows = new Set(
    rows
      .filter((row) => row.budget_outcome !== "api_error")
      .map((row) => `${row.repo_full}\0${row.repo_sha}\0${row.request.trim()}`),
  );
  const remaining: GatherJob[] = [];
  for (const job of pending) {
    const manifest = await loadManifest(job.dir);
    const key = `${manifest.fullName}\0${manifest.sha}\0${job.request.trim()}`;
    if (!completedRows.has(key)) remaining.push(job);
  }
  return remaining;
}

import type { BankedRow, GatherJob, LedgerEntry } from "./types.ts";
import { appendJsonl, stableJobId } from "./utils.ts";

export function ledgerEntry(
  job: GatherJob,
  row: BankedRow,
  durationMs: number,
  outcome: LedgerEntry["outcome"],
): LedgerEntry {
  return {
    job_id: stableJobId(job.dir, job.request),
    dir: job.dir,
    request: job.request,
    tags: job.tags,
    outcome,
    duration_ms: durationMs,
    input_tokens: row.input_tokens,
    output_tokens: row.output_tokens,
    account: row.account,
    valid: row.valid,
    reason: row.reason,
    ts: new Date().toISOString(),
  };
}

export async function appendBankedResult(
  rowsPath: string,
  ledgerPath: string,
  job: GatherJob,
  row: BankedRow,
  durationMs: number,
): Promise<void> {
  await appendJsonl(rowsPath, row);
  const outcome = row.budget_outcome === "api_error" ? "failed" : row.valid ? "banked" : "rejected";
  await appendJsonl(ledgerPath, ledgerEntry(job, row, durationMs, outcome));
}

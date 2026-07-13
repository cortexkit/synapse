import { posix } from "node:path";
import { expandHome, readLineRange } from "./repo.ts";
import { validateFinalJson } from "./schema.ts";
import type { BankedRow } from "./types.ts";
import { repoDirForRow, validateBankedRow } from "./validate.ts";

interface LineSpan {
  start: number;
  end: number;
}

export type CitationLineCaps = ReadonlyMap<string, ReadonlyMap<string, number>>;

export interface MetricSummary {
  count: number;
  mean: number | null;
  median: number | null;
}

export interface JobScore {
  job_id: string;
  candidate_model: string;
  contract_valid: boolean;
  file_f1: number;
  line_overlap: number;
  tool_efficiency: number | null;
  candidate_tool_calls: number;
  gold_tool_calls: number;
  budget_outcome: BankedRow["budget_outcome"];
  output_tokens: number;
  thinking_tokens: number;
}

export interface ScoreAggregate {
  jobs: number;
  contract_valid: MetricSummary;
  file_f1: MetricSummary;
  line_overlap: MetricSummary;
  tool_efficiency: MetricSummary;
  output_tokens: MetricSummary;
  thinking_tokens: MetricSummary;
  budget_outcomes: Record<BankedRow["budget_outcome"], number>;
}

export interface ScoreSummaryRow {
  model: string;
  jobs: number;
  contract_valid_mean: number | null;
  contract_valid_median: number | null;
  file_f1_mean: number | null;
  file_f1_median: number | null;
  line_overlap_mean: number | null;
  line_overlap_median: number | null;
  tool_efficiency_mean: number | null;
  tool_efficiency_median: number | null;
  output_tokens_mean: number | null;
  output_tokens_median: number | null;
  thinking_tokens_mean: number | null;
  thinking_tokens_median: number | null;
  budget_natural: number;
  budget_budget_finalize: number;
  budget_api_error: number;
  budget_invalid_final: number;
}

export interface ScoreReport {
  model: string;
  jobs: JobScore[];
  aggregate: ScoreAggregate;
  summary_row: ScoreSummaryRow;
  unmatched_candidate_job_ids: string[];
  unmatched_gold_job_ids: string[];
}

function jobKey(row: BankedRow): string {
  return `${row.repo_full}\0${row.repo_sha}\0${row.request.trim()}`;
}

function rowJobId(row: BankedRow): string {
  return row.job_id && row.job_id.trim().length > 0 ? row.job_id : jobKey(row);
}

function latestRowsByJob(rows: BankedRow[]): Map<string, BankedRow> {
  const latest = new Map<string, BankedRow>();
  for (const row of rows) latest.set(jobKey(row), row);
  return latest;
}

/** Normalize a citation path without allowing absolute paths to affect overlap grouping. */
export function normalizeCitationPath(path: string): string {
  const normalized = posix.normalize(path.replace(/\\/g, "/")).replace(/^\.\/+/, "");
  return normalized === "." ? "" : normalized;
}

/** Convert malformed ranges into non-empty, positive, inclusive spans before scoring. */
export function clampCitationSpan(startLine: unknown, endLine: unknown): LineSpan {
  const start = typeof startLine === "number" && Number.isFinite(startLine) ? Math.max(1, Math.floor(startLine)) : 1;
  const end = typeof endLine === "number" && Number.isFinite(endLine) ? Math.max(start, Math.floor(endLine)) : start;
  return { start, end };
}

function citationSpans(row: BankedRow, lineCaps?: CitationLineCaps): Map<string, LineSpan[]> {
  const byPath = new Map<string, LineSpan[]>();
  for (const snippet of row.final_json?.snippets ?? []) {
    if (typeof snippet.path !== "string") continue;
    const path = normalizeCitationPath(snippet.path);
    if (path.length === 0 || path.startsWith("/") || path.split("/").includes("..")) continue;
    const spans = byPath.get(path) ?? [];
    const span = clampCitationSpan(snippet.startLine, snippet.endLine);
    const lineCap = lineCaps?.get(jobKey(row))?.get(path);
    const end = typeof lineCap === "number" && Number.isFinite(lineCap) ? Math.min(span.end, Math.floor(lineCap)) : span.end;
    if (span.start <= end) spans.push({ start: span.start, end });
    // Keep the cited path even when the whole span is beyond EOF, because file
    // F1 measures citations while line overlap measures usable line ranges.
    byPath.set(path, spans);
  }
  return byPath;
}

function mergeSpans(spans: LineSpan[]): LineSpan[] {
  const sorted = [...spans].sort((left, right) => left.start - right.start || left.end - right.end);
  const merged: LineSpan[] = [];
  for (const span of sorted) {
    const prior = merged[merged.length - 1];
    if (prior && span.start <= prior.end + 1) prior.end = Math.max(prior.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
}

function spanLength(spans: LineSpan[]): number {
  return spans.reduce((total, span) => total + span.end - span.start + 1, 0);
}

function intersectionLength(left: LineSpan[], right: LineSpan[]): number {
  let total = 0;
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length && rightIndex < right.length) {
    const a = left[leftIndex]!;
    const b = right[rightIndex]!;
    total += Math.max(0, Math.min(a.end, b.end) - Math.max(a.start, b.start) + 1);
    if (a.end < b.end) leftIndex += 1;
    else rightIndex += 1;
  }
  return total;
}

function fileF1(candidate: Set<string>, gold: Set<string>): number {
  if (candidate.size === 0 && gold.size === 0) return 1;
  const shared = [...candidate].filter((path) => gold.has(path)).length;
  return (2 * shared) / (candidate.size + gold.size);
}

function lineOverlap(candidate: Map<string, LineSpan[]>, gold: Map<string, LineSpan[]>): number {
  const sharedPaths = [...candidate.keys()].filter((path) => gold.has(path));
  if (sharedPaths.length === 0) return candidate.size === 0 && gold.size === 0 ? 1 : 0;
  const overlaps = sharedPaths.map((path) => {
    const candidateSpans = mergeSpans(candidate.get(path)!);
    const goldSpans = mergeSpans(gold.get(path)!);
    if (candidateSpans.length === 0 || goldSpans.length === 0) return 0;
    const intersection = intersectionLength(candidateSpans, goldSpans);
    const union = spanLength(candidateSpans) + spanLength(goldSpans) - intersection;
    return union === 0 ? 1 : intersection / union;
  });
  return overlaps.reduce((total, overlap) => total + overlap, 0) / overlaps.length;
}

function toolCallCount(row: BankedRow): number {
  return row.full_trajectory.reduce(
    (total, message) =>
      total + (Array.isArray(message.content) ? message.content.filter((block) => block.type === "tool_use").length : 0),
    0,
  );
}

function nonNegativeNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : 0;
}

function summarize(values: Array<number | null>): MetricSummary {
  const present = values.filter((value): value is number => value !== null);
  if (present.length === 0) return { count: 0, mean: null, median: null };
  const sorted = [...present].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return {
    count: sorted.length,
    mean: present.reduce((total, value) => total + value, 0) / present.length,
    median: sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2,
  };
}

function modelName(rows: BankedRow[]): string {
  const names = [...new Set(rows.map((row) => row.model).filter((model) => model.length > 0))];
  return names.length === 1 ? names[0]! : names.length === 0 ? "unknown" : "multiple";
}

/**
 * Score the final package only. The caller supplies full validator results
 * when corpus clones are available; schema validation is the safe fallback
 * for an in-memory fixture or historical row with no local clone.
 */
export function scoreRows(
  candidateRows: BankedRow[],
  goldRows: BankedRow[],
  contractValidity = new Map<string, boolean>(),
  lineCaps?: CitationLineCaps,
): ScoreReport {
  const candidates = latestRowsByJob(candidateRows);
  const gold = latestRowsByJob(goldRows);
  const pairedKeys = [...candidates.keys()].filter((key) => gold.has(key)).sort();
  const jobs: JobScore[] = pairedKeys.map((key) => {
    const candidate = candidates.get(key)!;
    const goldRow = gold.get(key)!;
    const candidateCitations = citationSpans(candidate, lineCaps);
    const goldCitations = citationSpans(goldRow, lineCaps);
    const candidateToolCalls = toolCallCount(candidate);
    const goldToolCalls = toolCallCount(goldRow);
    return {
      job_id: rowJobId(candidate),
      candidate_model: candidate.model,
      contract_valid: contractValidity.get(key) ?? validateFinalJson(candidate.final_json).valid,
      file_f1: fileF1(new Set(candidateCitations.keys()), new Set(goldCitations.keys())),
      line_overlap: lineOverlap(candidateCitations, goldCitations),
      tool_efficiency: goldToolCalls === 0 ? null : candidateToolCalls / goldToolCalls,
      candidate_tool_calls: candidateToolCalls,
      gold_tool_calls: goldToolCalls,
      budget_outcome: candidate.budget_outcome,
      output_tokens: nonNegativeNumber(candidate.output_tokens),
      thinking_tokens: nonNegativeNumber(candidate.thinking_tokens),
    };
  });
  const budgetOutcomes: Record<BankedRow["budget_outcome"], number> = {
    natural: 0,
    budget_finalize: 0,
    api_error: 0,
    invalid_final: 0,
  };
  for (const job of jobs) budgetOutcomes[job.budget_outcome] += 1;
  const aggregate: ScoreAggregate = {
    jobs: jobs.length,
    contract_valid: summarize(jobs.map((job) => (job.contract_valid ? 1 : 0))),
    file_f1: summarize(jobs.map((job) => job.file_f1)),
    line_overlap: summarize(jobs.map((job) => job.line_overlap)),
    tool_efficiency: summarize(jobs.map((job) => job.tool_efficiency)),
    output_tokens: summarize(jobs.map((job) => job.output_tokens)),
    thinking_tokens: summarize(jobs.map((job) => job.thinking_tokens)),
    budget_outcomes: budgetOutcomes,
  };
  const model = modelName([...candidates.values()]);
  return {
    model,
    jobs,
    aggregate,
    summary_row: {
      model,
      jobs: aggregate.jobs,
      contract_valid_mean: aggregate.contract_valid.mean,
      contract_valid_median: aggregate.contract_valid.median,
      file_f1_mean: aggregate.file_f1.mean,
      file_f1_median: aggregate.file_f1.median,
      line_overlap_mean: aggregate.line_overlap.mean,
      line_overlap_median: aggregate.line_overlap.median,
      tool_efficiency_mean: aggregate.tool_efficiency.mean,
      tool_efficiency_median: aggregate.tool_efficiency.median,
      output_tokens_mean: aggregate.output_tokens.mean,
      output_tokens_median: aggregate.output_tokens.median,
      thinking_tokens_mean: aggregate.thinking_tokens.mean,
      thinking_tokens_median: aggregate.thinking_tokens.median,
      budget_natural: budgetOutcomes.natural,
      budget_budget_finalize: budgetOutcomes.budget_finalize,
      budget_api_error: budgetOutcomes.api_error,
      budget_invalid_final: budgetOutcomes.invalid_final,
    },
    unmatched_candidate_job_ids: [...candidates.entries()]
      .filter(([key]) => !gold.has(key))
      .map(([, row]) => rowJobId(row))
      .sort(),
    unmatched_gold_job_ids: [...gold.entries()]
      .filter(([key]) => !candidates.has(key))
      .map(([, row]) => rowJobId(row))
      .sort(),
  };
}

/**
 * Read each cited file once so ranges ending past EOF compare as the same
 * clamped span that citation validation resolves against the pinned clone.
 */
export async function citationLineCapsForScore(
  rows: BankedRow[],
  corpusRoot = "~/Work/OSS/gather-corpus",
): Promise<Map<string, Map<string, number>>> {
  const root = expandHome(corpusRoot);
  const caps = new Map<string, Map<string, number>>();
  const fileCounts = new Map<string, number>();
  for (const row of rows) {
    let repoDir: string;
    try {
      repoDir = repoDirForRow(root, row);
    } catch {
      continue;
    }
    const jobCaps = caps.get(jobKey(row)) ?? new Map<string, number>();
    caps.set(jobKey(row), jobCaps);
    for (const snippet of row.final_json?.snippets ?? []) {
      if (typeof snippet.path !== "string") continue;
      const path = normalizeCitationPath(snippet.path);
      if (path.length === 0 || path.startsWith("/") || path.split("/").includes("..")) continue;
      const cacheKey = `${repoDir}\0${path}`;
      let lineCount = fileCounts.get(cacheKey);
      if (lineCount === undefined) {
        try {
          lineCount = (await readLineRange(repoDir, path, 1, Number.MAX_SAFE_INTEGER)).lineCount;
          fileCounts.set(cacheKey, lineCount);
        } catch {
          continue;
        }
      }
      jobCaps.set(path, lineCount);
    }
  }
  return caps;
}

/** Re-run the existing full row validator so score results do not trust stale valid flags. */
export async function validateCandidateRowsForScore(
  rows: BankedRow[],
  corpusRoot = "~/Work/OSS/gather-corpus",
): Promise<Map<string, boolean>> {
  const results = new Map<string, boolean>();
  for (const row of latestRowsByJob(rows).values()) {
    try {
      const result = await validateBankedRow(row, repoDirForRow(expandHome(corpusRoot), row));
      results.set(jobKey(row), result.valid);
    } catch {
      results.set(jobKey(row), false);
    }
  }
  return results;
}

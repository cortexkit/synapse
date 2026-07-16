#!/usr/bin/env bun
import { AccountPool, CredentialStore } from "./auth.ts";
import { failedGatherJob, runGatherJob, type GatherBackend } from "./gather.ts";
import { appendBankedResult } from "./ledger.ts";
import { updateBurnRate } from "./meter.ts";
import { balanceJobs, pendingJobsAfterCrash } from "./queue.ts";
import { discoverRepos, expandHome } from "./repo.ts";
import { generateQuestions } from "./qgen.ts";
import type { BankedRow, GatherJob, LedgerEntry, TrajectoryMessage } from "./types.ts";
import { isRecord, parseJsonText, readJsonl, stableJobId, writeJsonAtomic, writeJsonl } from "./utils.ts";
import { repoDirForRow, validateBankedRow } from "./validate.ts";
import { citationLineCapsForScore, scoreRows, validateCandidateRowsForScore, type ScoreReport } from "./scorer.ts";
import {
  buildControlPackages,
  evaluateCalibrationGate,
  estimateJudgeCost,
  hydrateJudgePackage,
  isJudgeableRow,
  judgePromptForIteration,
  JUDGE_MODEL_DEFAULT,
  JUDGE_TEMPERATURE,
  JUDGE_TOPUP_BUDGET,
  runJudgeEvaluation,
  skippedJudgeRow,
  summarizeJudgeRows,
  type ControlPackageSpec,
  type JudgePackageKind,
  type JudgeRequestOptions,
  type JudgeVerdictRow,
} from "./judge.ts";
import { AftClientPool, AftWarmupCoordinator, AFT_WARMUP_TIMEOUT_MS } from "./tools.ts";
import { OPENAI_CODEX_RESPONSES_URL } from "./openai-oauth.ts";
import { appendFile, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

interface ParsedArgs {
  command: string;
  flags: Map<string, string[]>;
}

function parseArgs(argv: string[]): ParsedArgs {
  const [command = "help", ...rest] = argv;
  const flags = new Map<string, string[]>();
  for (let index = 0; index < rest.length; index += 1) {
    const token = rest[index];
    if (!token.startsWith("--")) throw new Error(`unexpected argument: ${token}`);
    const equals = token.indexOf("=");
    const name = token.slice(2, equals === -1 ? undefined : equals);
    const value = equals !== -1 ? token.slice(equals + 1) : rest[index + 1]?.startsWith("--") || rest[index + 1] === undefined ? "true" : rest[++index];
    flags.set(name, [...(flags.get(name) ?? []), value]);
  }
  return { command, flags };
}

function one(args: ParsedArgs, name: string, fallback?: string): string | undefined {
  return args.flags.get(name)?.at(-1) ?? fallback;
}

function required(args: ParsedArgs, name: string): string {
  const value = one(args, name);
  if (!value) throw new Error(`--${name} is required`);
  return value;
}

function numberFlag(args: ParsedArgs, name: string, fallback: number): number {
  const value = Number(one(args, name, String(fallback)));
  if (!Number.isFinite(value) || value <= 0) throw new Error(`--${name} must be a positive number`);
  return value;
}

function enabled(args: ParsedArgs, name: string): boolean {
  return ["true", "1", "yes"].includes(String(one(args, name, "false")).toLowerCase());
}

function accountPool(args: ParsedArgs): AccountPool {
  return new AccountPool({
    store: new CredentialStore(one(args, "accounts-file") ?? process.env.GATHER_DISTILL_ACCOUNTS_FILE),
    inFlightCap: numberFlag(args, "account-inflight", 2),
    cooldownMs: numberFlag(args, "auth-cooldown-ms", 300_000),
  });
}

function gatherBackend(args: ParsedArgs): GatherBackend {
  const backend = one(args, "backend", "anthropic");
  if (backend === "anthropic" || backend === "openai") return backend;
  throw new Error(`--backend must be anthropic or openai (got ${backend})`);
}

const LEXICAL_SEARCH_DISCLOSURE = "Semantic search unavailable; returning lexical-only fallback results.";

async function runGit(repo: string, ...args: string[]): Promise<void> {
  const process = Bun.spawn(["git", "-C", repo, ...args], { stdout: "pipe", stderr: "pipe" });
  const [status, _stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  if (status !== 0) throw new Error(`AFT canary could not run git ${args.join(" ")}: ${stderr.trim()}`);
}

async function runAftCanary(pool: AftClientPool): Promise<void> {
  const repo = await mkdtemp(join(tmpdir(), "gather-aft-canary-"));
  try {
    await runGit(repo, "init", "-q");
    await runGit(repo, "config", "user.name", "AFT Canary");
    await runGit(repo, "config", "user.email", "aft-canary@example.invalid");
    await writeFile(join(repo, "README.md"), "# lexicalCanary\n");
    await runGit(repo, "add", "README.md");
    await runGit(repo, "commit", "-qm", "canary");
    await pool.withClient(repo, async (client) => {
      const text = await client.call(repo, "search", { query: "lexicalCanary", topK: 1 });
      if (!text.includes(LEXICAL_SEARCH_DISCLOSURE)) {
        throw new Error("AFT canary did not confirm lexical-only search output");
      }
    });
  } finally {
    await rm(repo, { recursive: true, force: true });
  }
}

function effortFlag(args: ParsedArgs): "low" | "medium" | "high" | undefined {
  const value = one(args, "effort");
  if (value === undefined) return undefined;
  if (value === "low" || value === "medium" || value === "high") return value;
  throw new Error(`--effort must be low, medium, or high (got ${value})`);
}

async function qgenCommand(args: ParsedArgs): Promise<void> {
  const explicit = args.flags.get("repo")?.map(expandHome) ?? [];
  const repos = explicit.length > 0 ? explicit : await discoverRepos(one(args, "corpus-root"));
  if (repos.length === 0) throw new Error("qgen found no pinned repositories");
  const pool = accountPool(args);
  // --avoid-from: prior job files whose questions must not be duplicated
  // (deeper-coverage reruns over the same repos).
  const avoidByDir = new Map<string, string[]>();
  for (const path of args.flags.get("avoid-from") ?? []) {
    for (const job of await readJsonl<GatherJob>(path)) {
      const list = avoidByDir.get(job.dir) ?? [];
      list.push(job.request);
      avoidByDir.set(job.dir, list);
    }
  }
  const jobs: GatherJob[] = [];
  for (const repo of repos) {
    jobs.push(
      ...(await generateQuestions(repo, {
        pool,
        model: one(args, "model", "claude-sonnet-5-0"),
        count: numberFlag(args, "count", 20),
        maxTokens: numberFlag(args, "max-response-tokens", 6_000),
        effort: effortFlag(args),
        avoid: avoidByDir.get(repo) ?? [],
      })),
    );
  }
  const balanced = await balanceJobs(jobs);
  const output = one(args, "output", "data/jobs.jsonl")!;
  await writeJsonl(output, balanced);
  console.log(JSON.stringify({ lane: "qgen", repos: repos.length, jobs: balanced.length, output }));
}

async function gatherCommand(args: ParsedArgs): Promise<void> {
  const jobsPath = required(args, "jobs");
  const rowsPath = one(args, "rows", "data/rows.jsonl")!;
  const ledgerPath = one(args, "ledger", "data/ledger.jsonl")!;
  const statusPath = one(args, "status", "data/status.json")!;
  const jobs = await readJsonl<GatherJob>(jobsPath);
  const ledger = await readJsonl<LedgerEntry>(ledgerPath);
  const existingRows = await readJsonl<BankedRow>(rowsPath);
  const queue = await balanceJobs(await pendingJobsAfterCrash(jobs, ledger, existingRows));
  const backend = gatherBackend(args);
  // Local OpenAI-compatible servers do not use account files, credentials, or OAuth.
  const pool = backend === "anthropic" ? accountPool(args) : undefined;
  const concurrency = Math.floor(numberFlag(args, "concurrency", 2));
  const model = one(args, "model", backend === "openai" ? "local-model" : "claude-opus-4-8");
  const baseUrl = backend === "openai" ? one(args, "base-url", "http://127.0.0.1:8080/v1") : undefined;
  const requestTimeoutMs = backend === "openai" ? numberFlag(args, "request-timeout", 300) * 1_000 : undefined;
  const aftPool = new AftClientPool(concurrency);
  const warmups = new AftWarmupCoordinator();
  const reportedWarmupWarnings = new Set<string>();
  const aftRetries = new Map<string, number>();
  let cursor = 0;
  let bankChain = Promise.resolve();

  const bank = async (
    job: GatherJob,
    row: BankedRow,
    duration: number,
    index: number,
    warnings: string[] = [],
  ): Promise<void> => {
    bankChain = bankChain.then(async () => {
      await appendBankedResult(rowsPath, ledgerPath, job, row, duration, warnings);
      await updateBurnRate(rowsPath, statusPath);
      console.log(JSON.stringify({ lane: "gather", job: index, total: queue.length, row }));
    });
    await bankChain;
  };

  const worker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      const job = queue[index];
      if (!job) return;
      const started = Date.now();
      let skippedForWarmup = false;
      let ledgerWarnings: string[] = [];
      const row = await aftPool.withClient(job.dir, async (aftClient) => {
        const warmup = await warmups.ensureWarmed(job.dir, aftClient);
        if (!warmup.ok) {
          skippedForWarmup = true;
          const timeout = warmup.timedOut || warmup.durationMs >= AFT_WARMUP_TIMEOUT_MS;
          const queueNote = timeout
            ? "AFT warm-up request exceeded its timeout; skipped repository"
            : "AFT warm-up failed; skipped repository";
          console.warn(
            JSON.stringify({ lane: "gather", queue_note: queueNote, repo: job.dir, duration_ms: warmup.durationMs, error: warmup.error }),
          );
          return failedGatherJob(job, `AFT warm-up: ${warmup.error ?? queueNote}`, { backend, model });
        }
        if (warmup.warning && !reportedWarmupWarnings.has(job.dir)) {
          reportedWarmupWarnings.add(job.dir);
          ledgerWarnings = [warmup.warning];
          console.warn(
            JSON.stringify({
              lane: "gather",
              queue_note: "AFT SEARCH INDEX WARM-UP TIMED OUT; PROCEEDING COLD",
              repo: job.dir,
              duration_ms: warmup.durationMs,
              search_attempts: warmup.searchAttempts,
              warning: warmup.warning,
            }),
          );
        }
        return runGatherJob(job, {
          backend,
          baseUrl,
          requestTimeoutMs,
          pool,
          model,
          maxSteps: numberFlag(args, "max-steps", 40),
          maxPackageTokens: numberFlag(args, "max-package-tokens", 40_000),
          tokenCeiling: numberFlag(args, "token-ceiling", 200_000),
          maxResponseTokens: numberFlag(args, "max-response-tokens", 8_000),
          finalizeMode: one(args, "finalize-mode", "tool_choice_none_full_toolset") as
            | "tool_choice_none_full_toolset"
            | "tools_empty",
          inlineValidate: enabled(args, "inline-validate"),
          aftClient,
        });
      });
      const duration = Date.now() - started;
      await bank(job, row, duration, index + 1, ledgerWarnings);

      const retryKey = stableJobId(job.dir, job.request);
      const retries = aftRetries.get(retryKey) ?? 0;
      if (!skippedForWarmup && row.budget_outcome === "api_error" && row.reason?.startsWith("AFT transport:") && retries === 0) {
        aftRetries.set(retryKey, retries + 1);
        queue.push(job);
        console.warn(JSON.stringify({ lane: "gather", queue_note: "retrying job after AFT transport failure", repo: job.dir }));
      }
    }
  };

  try {
    if (queue.length > 0) await runAftCanary(aftPool);
    await Promise.all(Array.from({ length: concurrency }, () => worker()));
  } finally {
    await aftPool.close();
  }
  console.log(JSON.stringify({ lane: "gather", completed: queue.length, rows: rowsPath, ledger: ledgerPath, status: statusPath }));
}

async function validateCommand(args: ParsedArgs): Promise<void> {
  const input = required(args, "rows");
  const output = one(args, "output", `${input}.validated.jsonl`)!;
  if (output === input) throw new Error("validation output must differ from input; rows are never repaired in place");
  const corpusRoot = expandHome(one(args, "corpus-root", "~/Work/OSS/gather-corpus")!);
  const rows = await readJsonl<BankedRow>(input);
  const validated: BankedRow[] = [];
  let rejected = 0;
  for (const row of rows) {
    const result = await validateBankedRow(row, repoDirForRow(corpusRoot, row));
    if (!result.valid) rejected += 1;
    validated.push({ ...row, valid: result.valid, reason: result.errors.join("; ") || undefined });
  }
  await writeJsonl(output, validated);
  console.log(JSON.stringify({ lane: "validate", rows: rows.length, valid: rows.length - rejected, rejected, output }));
  if (rejected > 0) process.exitCode = 1;
}

async function scoreCommand(args: ParsedArgs): Promise<void> {
  const candidatePath = required(args, "candidate");
  const goldPath = required(args, "gold");
  const output = required(args, "output");
  const corpusRoot = one(args, "corpus-root", "~/Work/OSS/gather-corpus")!;
  const candidateRows = await readJsonl<BankedRow>(candidatePath);
  const goldRows = await readJsonl<BankedRow>(goldPath);
  const contractValidity = await validateCandidateRowsForScore(candidateRows, corpusRoot);
  const lineCaps = await citationLineCapsForScore([...candidateRows, ...goldRows], corpusRoot);
  const report = scoreRows(candidateRows, goldRows, contractValidity, lineCaps);
  await writeJsonAtomic(output, report);
  console.log(
    JSON.stringify({
      lane: "score",
      candidate: candidatePath,
      gold: goldPath,
      output,
      ...report.summary_row,
      unmatched_candidate_jobs: report.unmatched_candidate_job_ids.length,
      unmatched_gold_jobs: report.unmatched_gold_job_ids.length,
    }),
  );
}

function isBudgetOutcome(value: unknown): value is BankedRow["budget_outcome"] {
  return value === "natural" || value === "budget_finalize" || value === "api_error" || value === "invalid_final";
}

function normalizeTrajectory(value: unknown): TrajectoryMessage[] {
  if (!Array.isArray(value)) return [];
  const messages: TrajectoryMessage[] = [];
  for (const candidate of value) {
    if (!isRecord(candidate)) continue;
    const role = candidate.role;
    if (role !== "user" && role !== "assistant") continue;
    if (typeof candidate.content === "string") {
      messages.push({ role, content: candidate.content });
      continue;
    }
    if (!Array.isArray(candidate.content)) continue;
    const blocks: Exclude<TrajectoryMessage["content"], string> = [];
    for (const block of candidate.content) {
      if (!isRecord(block)) continue;
      if (block.type === "text" && typeof block.text === "string") {
        blocks.push({ type: "text", text: block.text });
      } else if (block.type === "tool_use" && typeof block.id === "string" && typeof block.name === "string") {
        blocks.push({ type: "tool_use", id: block.id, name: block.name, input: block.input });
      } else if (block.type === "tool_result" && typeof block.tool_use_id === "string" && typeof block.content === "string") {
        blocks.push({
          type: "tool_result",
          tool_use_id: block.tool_use_id,
          content: block.content,
          ...(typeof block.is_error === "boolean" ? { is_error: block.is_error } : {}),
        });
      }
    }
    messages.push({ role, content: blocks });
  }
  return messages;
}

function trajectoryFromCandidate(payload: unknown): TrajectoryMessage[] {
  if (Array.isArray(payload)) return normalizeTrajectory(payload);
  if (!isRecord(payload)) return [];
  return normalizeTrajectory(payload.full_trajectory ?? payload.trajectory);
}

function finalPackageFromTrajectory(trajectory: TrajectoryMessage[]): unknown {
  for (const message of [...trajectory].reverse()) {
    const texts =
      typeof message.content === "string"
        ? [message.content]
        : message.content.flatMap((block) => (block.type === "text" ? [block.text] : []));
    for (const text of texts.reverse()) {
      try {
        return parseJsonText(text);
      } catch {
        // Earlier assistant text can be commentary rather than the final package.
      }
    }
  }
  return null;
}

function finalPackageFromCandidate(payload: unknown, trajectory: TrajectoryMessage[]): BankedRow["final_json"] {
  if (isRecord(payload)) {
    if ("final_json" in payload) return payload.final_json as BankedRow["final_json"];
    if ("final_package" in payload) return payload.final_package as BankedRow["final_json"];
    if (Array.isArray(payload.full_trajectory) || Array.isArray(payload.trajectory)) {
      return finalPackageFromTrajectory(trajectory) as BankedRow["final_json"];
    }
  }
  if (Array.isArray(payload)) return finalPackageFromTrajectory(trajectory) as BankedRow["final_json"];
  return payload as BankedRow["final_json"];
}

function candidateRowForScoreOne(payload: unknown, gold: BankedRow): BankedRow {
  const source = isRecord(payload) ? payload : {};
  const trajectory = trajectoryFromCandidate(payload);
  return {
    ...gold,
    full_trajectory: trajectory,
    final_json: finalPackageFromCandidate(payload, trajectory),
    budget_outcome: isBudgetOutcome(source.budget_outcome) ? source.budget_outcome : "natural",
    model: typeof source.model === "string" && source.model.length > 0 ? source.model : "trace-candidate",
    account: typeof source.account === "string" && source.account.length > 0 ? source.account : "trace",
    valid: false,
  };
}

async function readCandidateFile(path: string): Promise<unknown> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    throw new Error(`cannot read candidate file ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`candidate file ${path} must contain one JSON value: ${error instanceof Error ? error.message : String(error)}`);
  }
}

async function scoreOneCommand(args: ParsedArgs): Promise<void> {
  const jobId = required(args, "job");
  const candidatePath = required(args, "candidate-file");
  const goldPath = required(args, "gold");
  const goldRows = await readJsonl<BankedRow>(goldPath);
  const gold = goldRows.filter((row) => row.job_id === jobId);
  if (gold.length !== 1) {
    throw new Error(`--job ${jobId} matched ${gold.length} gold rows; score-one requires one uniquely identified gold row`);
  }

  const candidate = candidateRowForScoreOne(await readCandidateFile(candidatePath), gold[0]!);
  // Distributed trainers often mount gold rows without their pinned source clones.
  // scoreRows then applies the production final-package schema as its contract check;
  // the batch score command still performs clone-backed citation validation.
  const report = scoreRows([candidate], [gold[0]!]);
  const score = report.jobs[0];
  if (!score) throw new Error(`score-one could not pair candidate with gold job ${jobId}`);
  const naturalCompletion = candidate.budget_outcome === "natural";
  const reward = score.contract_valid && naturalCompletion ? score.file_f1 : 0;
  console.log(
    JSON.stringify({
      lane: "score-one",
      job_id: jobId,
      reward,
      diagnostics: {
        file_f1: score.file_f1,
        line_jaccard: score.line_overlap,
        contract_valid: score.contract_valid,
        tool_calls: score.candidate_tool_calls,
        budget_outcome: candidate.budget_outcome,
        natural_completion: naturalCompletion,
      },
    }),
  );
}

interface JudgeAssignment {
  label: string;
  path: string;
}

function assignments(args: ParsedArgs, name: string): JudgeAssignment[] {
  return (args.flags.get(name) ?? []).map((value) => {
    const separator = value.indexOf("=");
    if (separator <= 0 || separator === value.length - 1) {
      throw new Error(`--${name} values must use LABEL=PATH`);
    }
    const label = value.slice(0, separator);
    const path = value.slice(separator + 1);
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(label)) {
      throw new Error(`--${name} label must contain only letters, digits, dot, underscore, and dash`);
    }
    return { label, path };
  });
}

function optionalRate(args: ParsedArgs, name: string): number | undefined {
  const raw = one(args, name);
  if (raw === undefined) return undefined;
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) throw new Error(`--${name} must be a non-negative number`);
  return value;
}

function judgeRowForJob(rows: BankedRow[], job: GatherJob): BankedRow | undefined {
  const id = stableJobId(job.dir, job.request);
  return rows.find((row) => row.job_id === id) ?? rows.find((row) => row.request.trim() === job.request.trim());
}

function judgePlaceholderRow(job: GatherJob): BankedRow {
  return {
    job_id: stableJobId(job.dir, job.request),
    request: job.request,
    repo_full: "unknown",
    repo_sha: "unknown",
    tags: job.tags,
    full_trajectory: [],
    final_json: null,
    budget_outcome: "invalid_final",
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 0,
    model: "unknown",
    account: "unknown",
    ts: new Date().toISOString(),
    valid: false,
  };
}

function judgeErrorRow(
  job: GatherJob,
  label: string,
  kind: JudgePackageKind,
  settings: JudgeRequestOptions,
  error: unknown,
): JudgeVerdictRow {
  const prompt = settings.prompt ?? judgePromptForIteration(1);
  return {
    ...skippedJudgeRow(label, kind, judgePlaceholderRow(job), settings, "judge call failed"),
    job_id: stableJobId(job.dir, job.request),
    repo_full: "unknown",
    repo_sha: "unknown",
    status: "error",
    phase1_sufficiency: null,
    sufficiency: null,
    missing_evidence: [],
    package_score: null,
    score_rationale: "",
    judge_model: settings.model ?? JUDGE_MODEL_DEFAULT,
    judge_prompt_sha: prompt.sha,
    judge_temperature: settings.oauth ? null : settings.temperature ?? JUDGE_TEMPERATURE,
    judge_topup_budget: settings.topupBudget ?? JUDGE_TOPUP_BUDGET,
    error: error instanceof Error ? error.message : String(error),
  };
}

interface JudgeTask {
  question: GatherJob;
  label: string;
  kind: JudgePackageKind;
  package: import("./types.ts").GatherFinalJson | null;
  sourceRepoDir: string;
  row: BankedRow;
  skipReason?: string;
}

async function runJudgeTasks(
  tasks: JudgeTask[],
  settings: JudgeRequestOptions,
  concurrency: number,
  onRow?: (row: JudgeVerdictRow) => Promise<void>,
): Promise<JudgeVerdictRow[]> {
  const pool = new AftClientPool(concurrency);
  const output: Array<JudgeVerdictRow | undefined> = Array.from({ length: tasks.length });
  let cursor = 0;
  const record = async (index: number, row: JudgeVerdictRow): Promise<void> => {
    output[index] = row;
    await onRow?.(row);
  };
  const worker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      const task = tasks[index];
      if (!task) return;
      if (task.skipReason || !task.package) {
        await record(index, skippedJudgeRow(task.label, task.kind, task.row, settings, task.skipReason ?? "package was not supplied"));
        continue;
      }
      try {
        const package_ = await hydrateJudgePackage(task.sourceRepoDir, task.package);
        const evaluation = await pool.withClient(task.question.dir, (client) =>
          runJudgeEvaluation(task.question.request, task.question.dir, package_, settings, client),
        );
        await record(index, {
          job_id: stableJobId(task.question.dir, task.question.request),
          request: task.question.request,
          repo_full: task.row.repo_full,
          repo_sha: task.row.repo_sha,
          label: task.label,
          package_kind: task.kind,
          status: evaluation.status,
          phase1_sufficiency: evaluation.phase1.phase1_sufficiency,
          answer_draft: evaluation.phase1.answer_draft,
          sufficiency: evaluation.verdict.sufficiency,
          topup_tool_calls: evaluation.topup_calls.length,
          topup_tokens: evaluation.topup_tokens_measured,
          missing_evidence: evaluation.verdict.missing_evidence,
          package_score: evaluation.verdict.package_score,
          score_rationale: evaluation.verdict.score_rationale,
          answer: evaluation.verdict.answer,
          topup_trace: evaluation.topup_calls,
          judge_model: settings.model ?? JUDGE_MODEL_DEFAULT,
          judge_prompt_sha: (settings.prompt ?? judgePromptForIteration(1)).sha,
          judge_temperature: settings.oauth ? null : settings.temperature ?? JUDGE_TEMPERATURE,
          judge_topup_budget: settings.topupBudget ?? JUDGE_TOPUP_BUDGET,
          judge_input_tokens: evaluation.judge_input_tokens,
          judge_output_tokens: evaluation.judge_output_tokens,
          phase1_tokens: evaluation.phase1_tokens,
        });
      } catch (error) {
        await record(index, judgeErrorRow(task.question, task.label, task.kind, settings, error));
      }
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.max(1, concurrency) }, () => worker()));
  } finally {
    await pool.close();
  }
  return output.filter((row): row is JudgeVerdictRow => row !== undefined);
}

function judgeRowKey(row: Pick<JudgeVerdictRow, "job_id" | "label" | "package_kind">): string {
  return `${row.label}\0${row.package_kind}\0${row.job_id}`;
}

function judgeTaskKey(task: JudgeTask): string {
  return `${task.label}\0${task.kind}\0${stableJobId(task.question.dir, task.question.request)}`;
}

async function readJudgeRowsIfPresent(path: string): Promise<JudgeVerdictRow[]> {
  try {
    return await readJsonl<JudgeVerdictRow>(path);
  } catch (error) {
    if ((error as { code?: unknown }).code === "ENOENT") return [];
    throw error;
  }
}

async function readFullJudgeProgress(outputDir: string, candidates: JudgeAssignment[]): Promise<JudgeVerdictRow[]> {
  const paths = [
    join(outputDir, "full-progress.jsonl"),
    join(outputDir, "gold-control-verdicts.jsonl"),
    ...candidates.map((candidate) => join(outputDir, `${candidate.label}-verdicts.jsonl`)),
  ];
  const rows = await Promise.all(paths.map((path) => readJudgeRowsIfPresent(path)));
  return rows.flat();
}

function controlTaskFor(
  question: GatherJob,
  spec: ControlPackageSpec,
  goldRows: BankedRow[],
  jobs: GatherJob[],
): JudgeTask {
  const questionRow = judgeRowForJob(goldRows, question) ?? judgePlaceholderRow(question);
  const sourceJob = spec.package_job_id
    ? jobs.find((job) => stableJobId(job.dir, job.request) === spec.package_job_id)
    : spec.package_request
      ? jobs.find((job) => job.request.trim() === spec.package_request?.trim())
      : question;
  return {
    question,
    label: "gold-control",
    kind: spec.kind,
    package: spec.package,
    sourceRepoDir: sourceJob?.dir ?? question.dir,
    row: questionRow,
  };
}

function candidateTasksFor(
  question: GatherJob,
  label: string,
  rows: BankedRow[],
): JudgeTask {
  const row = judgeRowForJob(rows, question) ?? judgePlaceholderRow(question);
  let skipReason: string | undefined;
  if (!row.final_json) skipReason = "candidate has no final package";
  else if (row.budget_outcome !== "natural") skipReason = `candidate completion was ${row.budget_outcome}, not natural`;
  else if (!row.valid) skipReason = "candidate package failed pinned-repository validation";
  else if (!isJudgeableRow(row)) skipReason = "candidate package failed final-package validation";
  return {
    question,
    label,
    kind: "candidate",
    package: row.final_json,
    sourceRepoDir: question.dir,
    row,
    skipReason,
  };
}

function f1ForScore(report: ScoreReport | undefined): number | null {
  if (!report) return null;
  const natural = report.jobs.filter((job) => job.budget_outcome === "natural");
  const values = natural.map((job) => job.file_f1);
  return values.length === 0 ? report.summary_row.file_f1_mean : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function numberText(value: number | null, digits = 2): string {
  return value === null ? "n/a" : value.toFixed(digits);
}

function summarizePhase1Sufficiency(rows: JudgeVerdictRow[]): { full: number; partial: number; none: number } {
  const completed = rows.filter((row) => row.status === "completed");
  return {
    full: completed.filter((row) => row.phase1_sufficiency === "answerable_fully").length,
    partial: completed.filter((row) => row.phase1_sufficiency === "answerable_partially").length,
    none: completed.filter((row) => row.phase1_sufficiency === "not_answerable").length,
  };
}

function markdownCell(value: string): string {
  return value.replace(/\s+/g, " ").replace(/\|/g, "\\|").trim();
}

function goldAnchorDifficultyLines(rows: JudgeVerdictRow[]): string[] {
  const completed = rows.filter((row) => row.status === "completed");
  if (completed.length === 0) {
    return ["## Gold-anchor difficulty spread", "", "No completed gold-control verdicts were available.", ""];
  }
  const needingTopups = completed
    .filter((row) => row.topup_tool_calls > 0)
    .sort((left, right) => right.topup_tool_calls - left.topup_tool_calls || left.job_id.localeCompare(right.job_id));
  const byCalls = new Map<number, number>();
  for (const row of completed) byCalls.set(row.topup_tool_calls, (byCalls.get(row.topup_tool_calls) ?? 0) + 1);
  const lines = [
    "## Gold-anchor difficulty spread",
    "",
    `${needingTopups.length} of ${completed.length} gold packages needed repository top-ups after phase 1.`,
    "",
    "| top-up calls | gold jobs |",
    "| ---: | ---: |",
    ...[...byCalls.entries()].sort(([left], [right]) => left - right).map(([calls, count]) => `| ${calls} | ${count} |`),
  ];
  if (needingTopups.length === 0) {
    lines.push("", "Every completed gold package was answerable without repository top-ups.", "");
    return lines;
  }
  lines.push("", "| job ID | request | phase-1 result | top-up calls | final result |", "| --- | --- | --- | ---: | --- |");
  for (const row of needingTopups) {
    const phase1 = row.phase1_sufficiency === "answerable_fully"
      ? "full"
      : row.phase1_sufficiency === "answerable_partially"
        ? "partial"
        : "not answerable";
    lines.push(`| ${row.job_id} | ${markdownCell(row.request)} | ${phase1} | ${row.topup_tool_calls} | ${row.sufficiency ?? "n/a"} |`);
  }
  lines.push("");
  return lines;
}

function renderUtilityJudgeReport(
  rowsByLabel: Map<string, JudgeVerdictRow[]>,
  scores: Map<string, ScoreReport>,
  calibration: {
    gate: ReturnType<typeof evaluateCalibrationGate>;
    cost: ReturnType<typeof estimateJudgeCost>;
    prompt_iteration?: number;
    prompt_sha?: string;
    prompt_change?: string;
  },
  calibrationRows: JudgeVerdictRow[],
  jobs: GatherJob[],
): string {
  const labels = [...rowsByLabel.keys()].sort();
  const summaries = labels.map((label) => ({ label, summary: summarizeJudgeRows(rowsByLabel.get(label)!), f1: label === "gold-control" ? 1 : f1ForScore(scores.get(label)) }));
  const ranked = summaries.filter((item) => item.label !== "gold-control" && item.summary.completed > 0);
  const utilityOrder = [...ranked].sort((left, right) => (left.summary.topup_calls_mean ?? Number.POSITIVE_INFINITY) - (right.summary.topup_calls_mean ?? Number.POSITIVE_INFINITY));
  const f1Order = [...ranked].sort((left, right) => (right.f1 ?? Number.NEGATIVE_INFINITY) - (left.f1 ?? Number.NEGATIVE_INFINITY));
  const utilityRanking = utilityOrder.map((item) => item.label).join(" < ");
  const f1Ranking = f1Order.map((item) => item.label).join(" > ");
  const rankingAgrees = ranked.length > 0 && utilityOrder.map((item) => item.label).join("\0") === f1Order.map((item) => item.label).join("\0");

  const examples: string[] = [];
  for (let index = 0; index < ranked.length && examples.length < 2; index += 1) {
    for (let next = index + 1; next < ranked.length && examples.length < 2; next += 1) {
      const left = ranked[index]!.label;
      const right = ranked[next]!.label;
      const leftScoreJobs = scores.get(left)?.jobs ?? [];
      const rightScoreJobs = scores.get(right)?.jobs ?? [];
      const leftRows = rowsByLabel.get(left) ?? [];
      const rightRows = rowsByLabel.get(right) ?? [];
      const sharedJobIds = leftRows.map((row) => row.job_id).filter((jobId) => rightRows.some((row) => row.job_id === jobId));
      for (const jobId of sharedJobIds) {
        const leftScore = leftScoreJobs.find((job) => job.job_id === jobId);
        const rightScore = rightScoreJobs.find((job) => job.job_id === jobId);
        const leftVerdict = leftRows.find((row) => row.job_id === jobId);
        const rightVerdict = rightRows.find((row) => row.job_id === jobId);
        if (!leftScore || !rightScore || !leftVerdict || !rightVerdict) continue;
        if ((leftScore.file_f1 - rightScore.file_f1) * (leftVerdict.topup_tool_calls - rightVerdict.topup_tool_calls) < 0) {
          const job = jobs.find((candidate) => stableJobId(candidate.dir, candidate.request) === jobId);
          examples.push(`${left} vs ${right} on ${jobId}${job ? ` (${job.request})` : ""}: ${left} F1 ${leftScore.file_f1.toFixed(2)} with ${leftVerdict.topup_tool_calls} top-up calls versus ${right} F1 ${rightScore.file_f1.toFixed(2)} with ${rightVerdict.topup_tool_calls} calls.`);
          break;
        }
      }
    }
  }
  if (examples.length === 0) {
    examples.push("No pairwise ranking reversal was found in the available aggregate data.", "The per-job verdict files remain the source for concrete package-level comparisons.");
  }

  const lines = [
    "# Utility judge evaluation",
    "",
    "The judge is blind to candidate labels and receives hydrated snippet bytes. Top-up calls are the headline utility cost; package score is secondary. F1 is the existing gold-overlap file F1, restricted to natural completions when a score report is available.",
    "",
    `Calibration gate: **${calibration.gate.pass ? "PASS" : "FAIL"}** (gold mean top-up ${numberText(calibration.gate.gold_mean_topup_calls)}, empty mean ${numberText(calibration.gate.empty_mean_topup_calls)}, mismatched none ${calibration.gate.mismatch_none}/${calibration.gate.mismatch_rows}).`,
    calibration.gate.reasons.length > 0 ? `Gate reasons: ${calibration.gate.reasons.join("; ")}.` : "",
    `Calibration prompt: iteration ${calibration.prompt_iteration ?? "unknown"}, SHA ${calibration.prompt_sha ?? "unknown"}; ${calibration.prompt_change ?? "change not recorded"}.`,
    `Calibration cost projection: ${calibration.cost.projected_usd === null ? "unpriced; provide endpoint-specific token rates" : `$${calibration.cost.projected_usd.toFixed(2)}`} for ${calibration.cost.projected_packages} full-matrix packages (sample rows: ${calibration.cost.sample_rows}).`,
    "",
    "## Calibration evidence",
    "| control | rows | full / partial / none | top-up calls mean |",
    "| --- | ---: | ---: | ---: |",
    ...(["gold", "empty", "mismatched"] as const).map((kind) => {
      const summary = summarizeJudgeRows(calibrationRows.filter((row) => row.package_kind === kind));
      return `| ${kind} | ${summary.rows} | ${summary.sufficiency.full} / ${summary.sufficiency.partial} / ${summary.sufficiency.none} | ${numberText(summary.topup_calls_mean)} |`;
    }),
    "",
    ...goldAnchorDifficultyLines(rowsByLabel.get("gold-control") ?? []),
    "| system | phase-1 full / partial / none | final full / partial / none | top-up calls mean | top-up calls median | top-up tokens mean | score mean | F1 | skipped invalid | errors |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ];
  for (const item of summaries) {
    const summary = item.summary;
    const phase1 = summarizePhase1Sufficiency(rowsByLabel.get(item.label) ?? []);
    lines.push(`| ${item.label} | ${phase1.full} / ${phase1.partial} / ${phase1.none} | ${summary.sufficiency.full} / ${summary.sufficiency.partial} / ${summary.sufficiency.none} | ${numberText(summary.topup_calls_mean)} | ${numberText(summary.topup_calls_median)} | ${numberText(summary.topup_tokens_mean, 0)} | ${numberText(summary.package_score_mean)} | ${numberText(item.f1)} | ${summary.skipped_invalid} | ${summary.errors} |`);
  }
  const rankingConclusion = ranked.length === 0
    ? "## Ranking conclusion\nNo candidate package completed judging, so utility-versus-F1 ranking agreement is not assessable."
    : `## Ranking conclusion\nUtility ranking (lower top-up is better): **${utilityRanking}**. F1 ranking (higher is better): **${f1Ranking}**. The rankings ${rankingAgrees ? "agree" : "diverge"}.`;
  lines.push(
    "",
    rankingConclusion,
    "",
    "## Two concrete divergence examples",
    ...examples.map((example) => `- ${example}`),
    "",
    "Invalid or forced rows are recorded as `sufficiency=none` with zero judge calls and are reported separately as skipped; they are not evidence of a cheap sufficient package.",
    "",
  );
  return lines.filter((line, index) => !(line === "" && lines[index - 1] === "")).join("\n");
}

async function discoverStudentAssignments(directory: string): Promise<JudgeAssignment[]> {
  try {
    const entries = await readdir(directory, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile() && entry.name.endsWith("-rows.jsonl"))
      .map((entry) => ({ label: entry.name.slice(0, -"-rows.jsonl".length), path: join(directory, entry.name) }))
      .sort((left, right) => left.label.localeCompare(right.label));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
}

async function judgeCommand(args: ParsedArgs): Promise<void> {
  const phase = one(args, "phase", "calibration");
  if (phase !== "calibration" && phase !== "full") throw new Error("--phase must be calibration or full");
  const jobsPath = required(args, "jobs");
  const goldPath = required(args, "gold");
  const outputDir = one(args, "output-dir", "data/students/judge")!;
  const oauthMode = one(args, "oauth", process.env.JUDGE_OAUTH);
  if (oauthMode !== undefined && oauthMode !== "opencode") {
    throw new Error("--oauth (or JUDGE_OAUTH) must be opencode when enabled");
  }
  const oauth = oauthMode === "opencode"
    ? {
        authFile: one(args, "oauth-auth-file", process.env.JUDGE_OAUTH_AUTH_FILE),
        sessionId: one(args, "session-id", process.env.JUDGE_SESSION_ID),
      }
    : undefined;
  const baseUrl = oauth
    ? undefined
    : one(args, "base-url", process.env.JUDGE_BASE_URL ?? process.env.GATHER_DISTILL_JUDGE_BASE_URL);
  if (!oauth && !baseUrl) throw new Error("--base-url (or JUDGE_BASE_URL) is required unless --oauth opencode is enabled");
  const apiKey = oauth
    ? undefined
    : one(args, "api-key", process.env.JUDGE_API_KEY ?? process.env.GATHER_DISTILL_JUDGE_API_KEY ?? process.env.OPENAI_API_KEY);
  if (!oauth && !apiKey) throw new Error("--api-key (or JUDGE_API_KEY) is required unless --oauth opencode is enabled");
  const jobs = await readJsonl<GatherJob>(jobsPath);
  const goldRows = await readJsonl<BankedRow>(goldPath);
  if (jobs.length === 0) throw new Error("judge found no jobs");
  const explicitCandidates = assignments(args, "candidate");
  const candidates = explicitCandidates.length > 0
    ? explicitCandidates
    : phase === "full"
      ? await discoverStudentAssignments(one(args, "students-dir", "data/students")!)
      : [];
  if (phase === "full" && candidates.length === 0) {
    throw new Error("full judge phase found no candidate rows; pass --candidate LABEL=ROWS.jsonl or stage data/students/*-rows.jsonl");
  }
  if (phase === "full" && explicitCandidates.length === 0 && !candidates.some((candidate) => candidate.label.includes("deepseek"))) {
    console.warn("judge: no deepseek student rows found under data/students; skipping that optional lane");
  }
  const candidateRows = new Map<string, BankedRow[]>();
  for (const candidate of candidates) candidateRows.set(candidate.label, await readJsonl<BankedRow>(candidate.path));

  const prompt = judgePromptForIteration(Number(one(args, "prompt-iteration", "1")));
  const settings: JudgeRequestOptions = {
    baseUrl,
    apiKey,
    oauth,
    apiKeyHeader: one(args, "api-key-header", process.env.JUDGE_KEY_HEADER ?? "authorization"),
    model: one(args, "judge-model", process.env.JUDGE_MODEL ?? (oauth ? "gpt-5.6-luna" : JUDGE_MODEL_DEFAULT)),
    temperature: Number(one(args, "temperature", String(JUDGE_TEMPERATURE))),
    topupBudget: JUDGE_TOPUP_BUDGET,
    maxResponseTokens: numberFlag(args, "max-response-tokens", 4_000),
    requestTimeoutMs: numberFlag(args, "request-timeout", 300) * 1_000,
    prompt,
  };
  if (settings.temperature === undefined || !Number.isFinite(settings.temperature) || settings.temperature < 0) {
    throw new Error("--temperature must be a non-negative number");
  }
  const concurrency = Math.floor(numberFlag(args, "concurrency", 2));
  const selectedJobs = phase === "calibration" ? jobs.slice(0, Math.min(5, jobs.length)) : jobs;
  if (phase === "calibration" && selectedJobs.length < 5) console.warn(`judge calibration has only ${selectedJobs.length} jobs; expected 5`);

  let calibrationRows: JudgeVerdictRow[] = [];
  let calibrationReport: {
    gate: ReturnType<typeof evaluateCalibrationGate>;
    cost: ReturnType<typeof estimateJudgeCost>;
    prompt_iteration?: number;
    prompt_sha?: string;
    prompt_change?: string;
  } | undefined;
  if (phase === "full") {
    const calibrationPath = one(args, "calibration-report", join(outputDir, "calibration-report.json"))!;
    const parsed = JSON.parse(await readFile(calibrationPath, "utf8")) as {
      gate?: ReturnType<typeof evaluateCalibrationGate>;
      cost?: ReturnType<typeof estimateJudgeCost>;
      prompt_iteration?: number;
      prompt_sha?: string;
      prompt_change?: string;
      rows?: JudgeVerdictRow[];
    };
    if (!parsed.gate?.pass) throw new Error(`calibration gate is not passing in ${calibrationPath}; run calibration first and fix the judge prompt`);
    calibrationRows = parsed.rows ?? await readJsonl<JudgeVerdictRow>(join(outputDir, "calibration-verdicts.jsonl"));
    calibrationReport = {
      gate: parsed.gate,
      cost: parsed.cost ?? estimateJudgeCost([], jobs.length * (candidates.length + 1)),
      prompt_iteration: parsed.prompt_iteration,
      prompt_sha: parsed.prompt_sha,
      prompt_change: parsed.prompt_change,
    };
  }

  const tasks: JudgeTask[] = [];
  for (const job of selectedJobs) {
    if (phase === "calibration") {
      const controls = buildControlPackages(job, goldRows, jobs);
      tasks.push(...controls.map((control) => controlTaskFor(job, control, goldRows, jobs)));
      for (const candidate of candidates) tasks.push(candidateTasksFor(job, candidate.label, candidateRows.get(candidate.label)!));
    } else {
      const goldRow = judgeRowForJob(goldRows, job) ?? judgePlaceholderRow(job);
      tasks.push({
        question: job,
        label: "gold-control",
        kind: "gold",
        package: goldRow.final_json,
        sourceRepoDir: job.dir,
        row: goldRow,
        skipReason: isJudgeableRow(goldRow) ? undefined : "gold control package is invalid or did not complete naturally",
      });
      for (const candidate of candidates) tasks.push(candidateTasksFor(job, candidate.label, candidateRows.get(candidate.label)!));
    }
  }

  let projectedCost: ReturnType<typeof estimateJudgeCost> | undefined;
  let reusedRows: JudgeVerdictRow[] = [];
  let pendingTasks = tasks;
  let persistRow: ((row: JudgeVerdictRow) => Promise<void>) | undefined;
  if (phase === "full") {
    await mkdir(outputDir, { recursive: true });
    const priorByKey = new Map<string, JudgeVerdictRow>();
    for (const row of await readFullJudgeProgress(outputDir, candidates)) priorByKey.set(judgeRowKey(row), row);
    const expectedModel = settings.model ?? JUDGE_MODEL_DEFAULT;
    const remaining: JudgeTask[] = [];
    for (const task of tasks) {
      const prior = priorByKey.get(judgeTaskKey(task));
      if (
        prior
        && (prior.status === "completed" || prior.status === "skipped_invalid")
        && prior.judge_model === expectedModel
        && prior.judge_prompt_sha === prompt.sha
      ) {
        reusedRows.push(prior);
      } else {
        remaining.push(task);
      }
    }
    pendingTasks = remaining;
    const progressPath = join(outputDir, "full-progress.jsonl");
    let progressWrite: Promise<void> = Promise.resolve();
    persistRow = async (row) => {
      progressWrite = progressWrite.then(() => appendFile(progressPath, `${JSON.stringify(row)}\n`, "utf8"));
      await progressWrite;
    };

    const quota = Number(one(args, "quota-usd", "30"));
    if (!Number.isFinite(quota) || quota <= 0) throw new Error("--quota-usd must be positive");
    const inputRate = optionalRate(args, "input-usd-per-million");
    const outputRate = optionalRate(args, "output-usd-per-million");
    projectedCost = estimateJudgeCost(calibrationRows, tasks.length, inputRate, outputRate);
    if (projectedCost.projected_usd === null && calibrationReport && calibrationReport.cost.projected_usd !== null) {
      projectedCost = calibrationReport.cost;
    }
    if (projectedCost.projected_usd !== null && projectedCost.projected_usd > quota) {
      throw new Error(`projected judge spend $${projectedCost.projected_usd.toFixed(2)} exceeds --quota-usd ${quota.toFixed(2)}; full matrix stopped`);
    }
    console.log(JSON.stringify({
      lane: "judge",
      phase,
      event: "preflight",
      projected_cost: projectedCost,
      quota_usd: quota,
      tasks: tasks.length,
      reused_rows: reusedRows.length,
      pending_tasks: pendingTasks.length,
    }));
  }

  const rows = [...reusedRows, ...await runJudgeTasks(pendingTasks, settings, concurrency, persistRow)];
  const calibrationPath = join(outputDir, "calibration-verdicts.jsonl");
  if (phase === "calibration") {
    const gate = evaluateCalibrationGate(rows);
    const projectedPackages = jobs.length * (candidates.length + 1 || 1);
    const cost = estimateJudgeCost(
      rows,
      projectedPackages,
      optionalRate(args, "input-usd-per-million"),
      optionalRate(args, "output-usd-per-million"),
    );
    await writeJsonl(calibrationPath, rows);
    await writeJsonAtomic(join(outputDir, "calibration-report.json"), {
      phase: "calibration",
      prompt_iteration: prompt.iteration,
      prompt_sha: prompt.sha,
      prompt_change: prompt.change,
      model: settings.model,
      transport: oauth ? "opencode-oauth-responses" : "openai-compatible-chat-completions",
      endpoint: oauth ? OPENAI_CODEX_RESPONSES_URL : baseUrl,
      temperature_requested: settings.temperature,
      temperature_pinned: oauth ? null : settings.temperature,
      topup_budget: settings.topupBudget,
      gate,
      cost,
      rows,
    });
    console.log(JSON.stringify({ lane: "judge", phase, output: calibrationPath, gate, cost, prompt_sha: prompt.sha }));
    if (!gate.pass) process.exitCode = 1;
    return;
  }

  const rowsByLabel = new Map<string, JudgeVerdictRow[]>();
  rowsByLabel.set("gold-control", rows.filter((row) => row.label === "gold-control"));
  for (const candidate of candidates) rowsByLabel.set(candidate.label, rows.filter((row) => row.label === candidate.label));
  await mkdir(outputDir, { recursive: true });
  await writeJsonl(join(outputDir, "gold-control-verdicts.jsonl"), rowsByLabel.get("gold-control")!);
  for (const candidate of candidates) await writeJsonl(join(outputDir, `${candidate.label}-verdicts.jsonl`), rowsByLabel.get(candidate.label)!);
  const scoreReports = new Map<string, ScoreReport>();
  for (const score of assignments(args, "score")) {
    scoreReports.set(score.label, JSON.parse(await readFile(score.path, "utf8")) as ScoreReport);
  }
  const report = renderUtilityJudgeReport(rowsByLabel, scoreReports, {
    gate: calibrationReport!.gate,
    cost: calibrationReport!.cost,
    prompt_iteration: calibrationReport!.prompt_iteration,
    prompt_sha: calibrationReport!.prompt_sha,
    prompt_change: calibrationReport!.prompt_change,
  }, calibrationRows, jobs);
  await writeFile(join("train", "UTILITY-JUDGE.md"), report, "utf8");
  console.log(JSON.stringify({ lane: "judge", phase, output: outputDir, projected_cost: projectedCost, prompt_sha: prompt.sha }));
}

function help(): void {
  console.log(`gather-distill

Subcommands:
  qgen     --repo DIR [--repo DIR...] --output data/jobs.jsonl [--model MODEL]
  gather   --jobs data/jobs.jsonl [--backend anthropic|openai] [--base-url URL] [--request-timeout SECONDS]
           [--rows PATH --ledger PATH --status PATH] [--inline-validate]
  validate --rows data/rows.jsonl [--output PATH --corpus-root DIR]
  score    --candidate data/model-rows.jsonl --gold data/eval-gold.jsonl --output data/model-scores.json
            [--corpus-root DIR]
   score-one --job JOB_ID --candidate-file candidate.json --gold data/eval-gold-rows.jsonl
  judge --phase calibration|full --jobs data/eval-jobs.jsonl --gold data/eval-gold-rows.jsonl
          --base-url URL --api-key KEY | --oauth opencode [--oauth-auth-file PATH]
          [--candidate LABEL=rows.jsonl ...] [--score LABEL=scores.json]
          [--output-dir data/students/judge --prompt-iteration 1|2|3]

  The judge uses a pinned GPT model through either an explicitly supplied OpenAI-compatible endpoint or
  OpenCode's read-only OAuth access token at ${OPENAI_CODEX_RESPONSES_URL}. OAuth never refreshes auth.json.
  Calibration runs gold, empty, and mismatched controls on the first five jobs; full runs gold controls plus candidates.
  Judge credentials are never written to verdict rows. Anthropic qgen and gather use GATHER_DISTILL_API_KEY or GATHER_DISTILL_ACCOUNTS_FILE/--accounts-file.
  OpenAI-compatible gather is local-only and never reads account credentials or OAuth settings.`);
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  if (args.command === "qgen") await qgenCommand(args);
  else if (args.command === "gather") await gatherCommand(args);
  else if (args.command === "validate") await validateCommand(args);
  else if (args.command === "score") await scoreCommand(args);
  else if (args.command === "score-one") await scoreOneCommand(args);
  else if (args.command === "judge") await judgeCommand(args);
  else help();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

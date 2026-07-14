#!/usr/bin/env bun
import { AccountPool, CredentialStore } from "./auth.ts";
import { failedGatherJob, runGatherJob, type GatherBackend } from "./gather.ts";
import { appendBankedResult } from "./ledger.ts";
import { updateBurnRate } from "./meter.ts";
import { balanceJobs, pendingJobsAfterCrash } from "./queue.ts";
import { discoverRepos, expandHome } from "./repo.ts";
import { generateQuestions } from "./qgen.ts";
import type { BankedRow, GatherJob, LedgerEntry } from "./types.ts";
import { readJsonl, stableJobId, writeJsonAtomic, writeJsonl } from "./utils.ts";
import { repoDirForRow, validateBankedRow } from "./validate.ts";
import { citationLineCapsForScore, scoreRows, validateCandidateRowsForScore } from "./scorer.ts";
import { AftClientPool, AftWarmupCoordinator, AFT_WARMUP_TIMEOUT_MS } from "./tools.ts";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
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

function help(): void {
  console.log(`gather-distill

Subcommands:
  qgen     --repo DIR [--repo DIR...] --output data/jobs.jsonl [--model MODEL]
  gather   --jobs data/jobs.jsonl [--backend anthropic|openai] [--base-url URL] [--request-timeout SECONDS]
           [--rows PATH --ledger PATH --status PATH] [--inline-validate]
  validate --rows data/rows.jsonl [--output PATH --corpus-root DIR]
  score    --candidate data/model-rows.jsonl --gold data/eval-gold.jsonl --output data/model-scores.json
           [--corpus-root DIR]

Anthropic qgen and gather use GATHER_DISTILL_API_KEY or GATHER_DISTILL_ACCOUNTS_FILE/--accounts-file.
OpenAI-compatible gather is local-only and never reads account credentials or OAuth settings.`);
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  if (args.command === "qgen") await qgenCommand(args);
  else if (args.command === "gather") await gatherCommand(args);
  else if (args.command === "validate") await validateCommand(args);
  else if (args.command === "score") await scoreCommand(args);
  else help();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

#!/usr/bin/env bun
import { AccountPool, CredentialStore } from "./auth.ts";
import { runGatherJob } from "./gather.ts";
import { appendBankedResult } from "./ledger.ts";
import { updateBurnRate } from "./meter.ts";
import { balanceJobs, pendingJobsAfterCrash } from "./queue.ts";
import { discoverRepos, expandHome } from "./repo.ts";
import { generateQuestions } from "./qgen.ts";
import type { BankedRow, GatherJob, LedgerEntry } from "./types.ts";
import { readJsonl, writeJsonl } from "./utils.ts";
import { repoDirForRow, validateBankedRow } from "./validate.ts";

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

async function qgenCommand(args: ParsedArgs): Promise<void> {
  const explicit = args.flags.get("repo")?.map(expandHome) ?? [];
  const repos = explicit.length > 0 ? explicit : await discoverRepos(one(args, "corpus-root"));
  if (repos.length === 0) throw new Error("qgen found no pinned repositories");
  const pool = accountPool(args);
  const jobs: GatherJob[] = [];
  for (const repo of repos) {
    jobs.push(
      ...(await generateQuestions(repo, {
        pool,
        model: one(args, "model", "claude-sonnet-5-0"),
        count: numberFlag(args, "count", 20),
        maxTokens: numberFlag(args, "max-response-tokens", 6_000),
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
  const pool = accountPool(args);
  const concurrency = Math.floor(numberFlag(args, "concurrency", 2));
  let cursor = 0;
  let bankChain = Promise.resolve();

  const worker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      const job = queue[index];
      if (!job) return;
      const started = Date.now();
      const row = await runGatherJob(job, {
        pool,
        model: one(args, "model", "claude-opus-4-8"),
        maxSteps: numberFlag(args, "max-steps", 40),
        maxPackageTokens: numberFlag(args, "max-package-tokens", 40_000),
        tokenCeiling: numberFlag(args, "token-ceiling", 200_000),
        maxResponseTokens: numberFlag(args, "max-response-tokens", 8_000),
        finalizeMode: one(args, "finalize-mode", "tool_choice_none_full_toolset") as
          | "tool_choice_none_full_toolset"
          | "tools_empty",
        inlineValidate: enabled(args, "inline-validate"),
      });
      const duration = Date.now() - started;
      bankChain = bankChain.then(async () => {
        await appendBankedResult(rowsPath, ledgerPath, job, row, duration);
        await updateBurnRate(rowsPath, statusPath);
        console.log(JSON.stringify({ lane: "gather", job: index + 1, total: queue.length, row }));
      });
      await bankChain;
    }
  };
  await Promise.all(Array.from({ length: concurrency }, () => worker()));
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

function help(): void {
  console.log(`gather-distill

Subcommands:
  qgen     --repo DIR [--repo DIR...] --output data/jobs.jsonl [--model MODEL]
  gather   --jobs data/jobs.jsonl [--rows PATH --ledger PATH --status PATH] [--inline-validate]
  validate --rows data/rows.jsonl [--output PATH --corpus-root DIR]

Authentication comes only from GATHER_DISTILL_API_KEY or GATHER_DISTILL_ACCOUNTS_FILE/--accounts-file.`);
}

async function main(): Promise<void> {
  const args = parseArgs(Bun.argv.slice(2));
  if (args.command === "qgen") await qgenCommand(args);
  else if (args.command === "gather") await gatherCommand(args);
  else if (args.command === "validate") await validateCommand(args);
  else help();
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});

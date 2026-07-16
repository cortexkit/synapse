#!/usr/bin/env bun
/**
 * Generate a no-LLM gather candidate from CodeGraph's one-shot MCP explore tool.
 *
 * The script deliberately keeps CodeGraph's source ranges intact. Validation, not
 * this adapter, decides whether a returned range is usable against the pinned
 * corpus clone.
 */
import { mkdir, stat } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { homedir } from "node:os";
import { pathToFileURL } from "node:url";
import { loadManifest, readLineRange } from "../src/repo.ts";
import { validateBankedRow } from "../src/validate.ts";
import type { BankedRow, GatherFinalJson, GatherJob, RepoManifest } from "../src/types.ts";
import { isRecord, readJsonl, stableJobId, writeJsonAtomic, writeJsonl } from "../src/utils.ts";

type JsonRecord = Record<string, unknown>;

const DEFAULT_CODEGRAPH_REVISION = "246aee837341183912c82b3e727410e9fe1a1567";
const DEFAULT_CODEGRAPH_DIR = "~/Work/OSS/codegraph";
const DEFAULT_SCRATCH_ROOT = "/tmp/codegraph-eval";
const MCP_PROTOCOL_VERSION = "2024-11-05";
const EXPLORE_TIMEOUT_MS = 120_000;

interface ParsedArgs {
  flags: Map<string, string[]>;
}

interface Options {
  jobs: string;
  outputRows: string;
  rawOutput: string;
  metricsOutput: string;
  scratchRoot: string;
  codegraphDir: string;
  codegraphRevision: string;
  corpusRoot?: string;
  comparisonRows: Array<{ label: string; path: string }>;
}

interface McpTool {
  name: string;
  description?: string;
  inputSchema?: unknown;
}

interface RawExploreRecord {
  job_id: string;
  request: string;
  repo_full: string;
  repo_sha: string;
  scratch_repo: string;
  codegraph_revision: string;
  wall_ms: number;
  response?: unknown;
  error?: string;
}

interface CodeGraphMetadata {
  codegraph_revision: string;
  scratch_repo: string;
  wall_ms: number;
  snippet_count: number;
  snippet_bytes: number | null;
  hydrated_tokens_o200k: number | null;
  raw_output_path: string;
}

type CodeGraphRow = BankedRow & { codegraph_explore?: CodeGraphMetadata };

interface AdaptedExplore {
  package_: GatherFinalJson | null;
  error?: string;
}

interface HydratedMeasurement {
  serialized: string;
  snippetBytes: number;
  snippetCount: number;
}

interface PackageMeasurement {
  job_id: string;
  snippet_count: number;
  snippet_bytes: number;
  hydrated_tokens_o200k: number;
  wall_ms: number | null;
}

interface MetricSummary {
  count: number;
  mean: number | null;
  median: number | null;
  p95: number | null;
}

interface SystemMetrics {
  label: string;
  rows: number;
  package_rows: number;
  missing_or_invalid_rows: number;
  hydrated_tokens_o200k: MetricSummary;
  snippet_bytes: MetricSummary;
  snippet_count: MetricSummary;
  wall_ms: MetricSummary;
  packages: PackageMeasurement[];
}

function parseArgs(argv: string[]): ParsedArgs {
  const flags = new Map<string, string[]>();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token?.startsWith("--")) throw new Error(`unexpected argument: ${token ?? ""}`);
    const equals = token.indexOf("=");
    const name = token.slice(2, equals === -1 ? undefined : equals);
    const value = equals === -1 ? argv[index + 1] : token.slice(equals + 1);
    if (!value || value.startsWith("--")) throw new Error(`--${name} requires a value`);
    if (equals === -1) index += 1;
    flags.set(name, [...(flags.get(name) ?? []), value]);
  }
  return { flags };
}

function one(args: ParsedArgs, name: string, fallback?: string): string | undefined {
  return args.flags.get(name)?.at(-1) ?? fallback;
}

function required(args: ParsedArgs, name: string): string {
  const value = one(args, name);
  if (!value) throw new Error(`--${name} is required`);
  return value;
}

function expandHome(path: string): string {
  return path === "~" ? homedir() : path.startsWith("~/") ? join(homedir(), path.slice(2)) : path;
}

function parseAssignment(value: string, flag: string): { label: string; path: string } {
  const separator = value.indexOf("=");
  if (separator <= 0 || separator === value.length - 1) {
    throw new Error(`--${flag} must use LABEL=PATH`);
  }
  return { label: value.slice(0, separator), path: value.slice(separator + 1) };
}

function optionsFrom(args: ParsedArgs): Options {
  const outputRows = required(args, "output-rows");
  const comparisons = (args.flags.get("comparison") ?? []).map((value) => parseAssignment(value, "comparison"));
  if (!comparisons.some((item) => item.label === "codegraph-explore")) {
    comparisons.unshift({ label: "codegraph-explore", path: outputRows });
  }
  return {
    jobs: required(args, "jobs"),
    outputRows,
    rawOutput: required(args, "raw-output"),
    metricsOutput: required(args, "metrics-output"),
    scratchRoot: expandHome(one(args, "scratch-root", DEFAULT_SCRATCH_ROOT)!),
    codegraphDir: expandHome(one(args, "codegraph-dir", DEFAULT_CODEGRAPH_DIR)!),
    codegraphRevision: one(args, "codegraph-revision", DEFAULT_CODEGRAPH_REVISION)!,
    corpusRoot: one(args, "corpus-root"),
    comparisonRows: comparisons,
  };
}

async function command(command_: string, args: string[], cwd?: string): Promise<string> {
  const process = Bun.spawn([command_, ...args], { cwd, stdout: "pipe", stderr: "pipe" });
  const [status, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  if (status !== 0) {
    throw new Error(`${command_} ${args.join(" ")} failed (${status}): ${stderr.trim() || stdout.trim()}`);
  }
  return stdout.trim();
}

async function exists(path: string): Promise<boolean> {
  try {
    await stat(path);
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

async function gitHead(repoDir: string): Promise<string> {
  return command("git", ["-C", repoDir, "rev-parse", "HEAD"]);
}

async function assertCodeGraphRevision(options: Options): Promise<void> {
  const actual = await gitHead(options.codegraphDir);
  if (actual !== options.codegraphRevision) {
    throw new Error(
      `CodeGraph revision mismatch: expected ${options.codegraphRevision}, found ${actual}. ` +
        "Use a scratch checkout at the requested revision instead of silently changing the benchmark.",
    );
  }
}

async function scratchRepoFor(source: string, expectedSha: string, scratchRoot: string): Promise<string> {
  const destination = join(scratchRoot, basename(source));
  if (!(await exists(destination))) {
    await mkdir(scratchRoot, { recursive: true });
    await command("git", ["clone", "--no-local", source, destination]);
  }
  const head = await gitHead(destination);
  if (head !== expectedSha) {
    throw new Error(`scratch clone ${destination} is at ${head}, expected pinned ${expectedSha}`);
  }
  return destination;
}

async function indexRepo(codegraphDir: string, repoDir: string): Promise<void> {
  const cli = join(codegraphDir, "dist", "bin", "codegraph.js");
  if (!(await exists(join(repoDir, ".codegraph")))) {
    await command("node", [cli, "init", repoDir, "--index"], codegraphDir);
    return;
  }
  await command("node", [cli, "sync", repoDir], codegraphDir);
}

function errorText(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

interface PendingMcpRequest {
  resolve(value: unknown): void;
  reject(error: Error): void;
  timer: ReturnType<typeof setTimeout>;
}

/** Minimal JSON-RPC client for the standard MCP stdio transport. */
class McpStdioClient {
  private readonly process: ReturnType<typeof Bun.spawn>;
  private readonly pending = new Map<number, PendingMcpRequest>();
  private readonly stderr: string[] = [];
  private nextId = 1;

  constructor(command_: string, args: string[], cwd: string) {
    this.process = Bun.spawn([command_, ...args], {
      cwd,
      env: { ...process.env, CODEGRAPH_NO_DAEMON: "1" },
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
    void this.consumeStdout();
    void this.consumeStderr();
    void this.process.exited.then((status) => {
      if (status === 0 || this.pending.size === 0) return;
      this.failPending(new Error(`CodeGraph MCP server exited with ${status}: ${this.stderr.join("").trim()}`));
    });
  }

  async initialize(repoDir: string): Promise<McpTool[]> {
    await this.request("initialize", {
      protocolVersion: MCP_PROTOCOL_VERSION,
      capabilities: {},
      clientInfo: { name: "gather-distill-codegraph-adapter", version: "1" },
      rootUri: pathToFileURL(repoDir).href,
    });
    await this.notify("notifications/initialized", {});
    const response = await this.request("tools/list", {});
    if (!isRecord(response) || !Array.isArray(response.tools)) throw new Error("CodeGraph MCP tools/list response has no tools array");
    return response.tools.filter(isRecord).map((tool) => ({
      name: typeof tool.name === "string" ? tool.name : "",
      ...(typeof tool.description === "string" ? { description: tool.description } : {}),
      ...(tool.inputSchema !== undefined ? { inputSchema: tool.inputSchema } : {}),
    }));
  }

  async callTool(name: string, arguments_: JsonRecord): Promise<unknown> {
    return this.request("tools/call", { name, arguments: arguments_ }, EXPLORE_TIMEOUT_MS);
  }

  async close(): Promise<void> {
    this.failPending(new Error("CodeGraph MCP client closed"));
    try {
      this.process.kill();
    } catch {
      // The server may have already exited after its last response.
    }
    await this.process.exited.catch(() => undefined);
  }

  private async notify(method: string, params: JsonRecord): Promise<void> {
    await Promise.resolve(this.process.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", method, params })}\n`));
  }

  private request(method: string, params: JsonRecord, timeoutMs = 30_000): Promise<unknown> {
    const id = this.nextId++;
    return new Promise<unknown>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`CodeGraph MCP ${method} request exceeded ${timeoutMs}ms`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      Promise.resolve(this.process.stdin.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`)).catch((error) => {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(new Error(`CodeGraph MCP stdin write failed: ${errorText(error)}`));
      });
    });
  }

  private async consumeStdout(): Promise<void> {
    let buffered = "";
    const decoder = new TextDecoder();
    try {
      for await (const chunk of this.process.stdout) {
        buffered += decoder.decode(chunk, { stream: true });
        let newline = buffered.indexOf("\n");
        while (newline !== -1) {
          this.handleLine(buffered.slice(0, newline).replace(/\r$/, ""));
          buffered = buffered.slice(newline + 1);
          newline = buffered.indexOf("\n");
        }
      }
      buffered += decoder.decode();
      if (buffered.trim()) this.handleLine(buffered.trim());
    } catch (error) {
      this.failPending(new Error(`CodeGraph MCP stdout failed: ${errorText(error)}`));
    }
  }

  private async consumeStderr(): Promise<void> {
    const decoder = new TextDecoder();
    try {
      for await (const chunk of this.process.stderr) {
        this.stderr.push(decoder.decode(chunk, { stream: true }));
      }
      this.stderr.push(decoder.decode());
    } catch {
      // The actual RPC error is more useful than an unavailable stderr stream.
    }
  }

  private handleLine(line: string): void {
    if (!line.trim()) return;
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      this.failPending(new Error(`CodeGraph MCP emitted non-JSON stdout: ${line.slice(0, 300)}`));
      return;
    }
    if (!isRecord(value) || typeof value.id !== "number") return;
    const pending = this.pending.get(value.id);
    if (!pending) return;
    this.pending.delete(value.id);
    clearTimeout(pending.timer);
    if (value.error !== undefined) {
      pending.reject(new Error(`CodeGraph MCP error: ${JSON.stringify(value.error)}`));
      return;
    }
    pending.resolve(value.result);
  }

  private failPending(error: Error): void {
    for (const [id, pending] of this.pending) {
      this.pending.delete(id);
      clearTimeout(pending.timer);
      pending.reject(error);
    }
  }
}

function propertyNames(schema: unknown): string[] {
  return isRecord(schema) && isRecord(schema.properties) ? Object.keys(schema.properties) : [];
}

function exploreArguments(tool: McpTool, request: string, repoDir: string): JsonRecord {
  const properties = propertyNames(tool.inputSchema);
  const queryKey = ["query", "question", "request", "prompt"].find((key) => properties.includes(key));
  if (!queryKey) throw new Error(`codegraph_explore schema has no query-like input: ${properties.join(", ")}`);
  const arguments_: JsonRecord = { [queryKey]: request };
  const rootKey = ["path", "projectPath", "project_path", "repoPath", "repo_path", "root", "directory"].find((key) => properties.includes(key));
  if (rootKey) arguments_[rootKey] = repoDir;
  return arguments_;
}

function asPositiveInteger(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : undefined;
}

function sourcePath(value: JsonRecord): string | undefined {
  for (const key of ["path", "filePath", "file_path", "file", "sourceFile", "source_file"]) {
    const candidate = value[key];
    if (typeof candidate === "string" && candidate.length > 0) return candidate.replace(/^\.\//, "");
  }
  return undefined;
}

function sourceRange(value: JsonRecord): { startLine: number; endLine: number } | undefined {
  const start = asPositiveInteger(value.startLine ?? value.start_line ?? value.lineStart ?? value.line_start);
  const end = asPositiveInteger(value.endLine ?? value.end_line ?? value.lineEnd ?? value.line_end);
  if (start && end && end >= start) return { startLine: start, endLine: end };
  if (isRecord(value.range)) {
    const nestedStart = asPositiveInteger(value.range.startLine ?? value.range.start_line ?? value.range.start);
    const nestedEnd = asPositiveInteger(value.range.endLine ?? value.range.end_line ?? value.range.end);
    if (nestedStart && nestedEnd && nestedEnd >= nestedStart) return { startLine: nestedStart, endLine: nestedEnd };
  }
  return undefined;
}

function relationshipWhy(value: JsonRecord): string {
  const candidates = [value.relationship, value.role, value.reason, value.kind, value.matchType, value.match_type, value.name]
    .filter((candidate): candidate is string => typeof candidate === "string" && candidate.trim().length > 0);
  return candidates[0] ?? "codegraph_explore source block";
}

function collectSnippetRecords(value: unknown, inheritedPath?: string, out: Array<{ path: string; startLine: number; endLine: number; why: string }> = []): Array<{ path: string; startLine: number; endLine: number; why: string }> {
  if (Array.isArray(value)) {
    for (const item of value) collectSnippetRecords(item, inheritedPath, out);
    return out;
  }
  if (!isRecord(value)) return out;
  const path = sourcePath(value) ?? inheritedPath;
  const range = sourceRange(value);
  if (path && range) out.push({ path, ...range, why: relationshipWhy(value) });
  for (const [key, nested] of Object.entries(value)) {
    if (["code", "source", "content", "text", "raw"].includes(key)) continue;
    if (nested && typeof nested === "object") collectSnippetRecords(nested, path, out);
  }
  return out;
}

function markdownTextsInToolResult(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.content)) return [];
  return value.content.flatMap((block) => (isRecord(block) && typeof block.text === "string" ? [block.text] : []));
}

/**
 * CodeGraph's MCP tool renders each file as a bold backticked heading followed
 * by numbered source lines in a fenced block. A gap in those line numbers is a
 * separate returned source block, not an invented span across omitted text.
 */
export function snippetsFromExploreMarkdown(markdown: string): GatherFinalJson["snippets"] {
  const snippets: GatherFinalJson["snippets"] = [];
  const heading = /^\*\*`([^`]+)`\*\*(?:\s*[—-]\s*(.+))?\s*$/;
  let path: string | undefined;
  let role = "explore source block";
  let inFence = false;
  let blockStart: number | undefined;
  let previousLine: number | undefined;
  const flush = () => {
    if (!path || blockStart === undefined || previousLine === undefined) return;
    snippets.push({ path, startLine: blockStart, endLine: previousLine, why: `codegraph_explore ${role}` });
    blockStart = undefined;
    previousLine = undefined;
  };
  for (const line of markdown.split(/\r?\n/)) {
    const match = line.match(heading);
    if (match) {
      flush();
      path = match[1]!.replace(/^\.\//, "");
      role = match[2]?.trim() || "explore source block";
      inFence = false;
      continue;
    }
    if (!path) continue;
    if (line.startsWith("```")) {
      if (inFence) flush();
      inFence = !inFence;
      continue;
    }
    if (!inFence) continue;
    const numbered = line.match(/^(\d+)\t/);
    if (!numbered) continue;
    const lineNumber = Number(numbered[1]);
    if (!Number.isSafeInteger(lineNumber) || lineNumber < 1) continue;
    if (previousLine !== undefined && lineNumber !== previousLine + 1) flush();
    if (blockStart === undefined) blockStart = lineNumber;
    previousLine = lineNumber;
  }
  flush();
  return snippets;
}

/** Collect every repository file named by CodeGraph's source, relation, and blast-radius sections. */
export function scopePathsFromExploreMarkdown(markdown: string): string[] {
  const paths = new Set(snippetsFromExploreMarkdown(markdown).map((snippet) => snippet.path));
  const citedPath = /(?:@|\()([A-Za-z0-9_.@/+-]+(?:\.[A-Za-z0-9_.-]+)+):\d+/g;
  for (const match of markdown.matchAll(citedPath)) {
    const path = match[1]!.replace(/^\.\//, "");
    if (!path.startsWith("/") && !path.split(/[\\/]/).includes("..")) paths.add(path);
  }
  return [...paths];
}

/**
 * Convert CodeGraph's Markdown MCP response into the gather pointer contract.
 * Any response without faithful file and line data is intentionally rejected.
 */
export function adaptExploreResponse(request: string, response: unknown): AdaptedExplore {
  if (isRecord(response) && response.isError === true) {
    return { package_: null, error: `codegraph_explore error: ${markdownTextsInToolResult(response).join(" ") || "unknown error"}` };
  }
  const markdown = markdownTextsInToolResult(response);
  const markdownSnippets = markdown.flatMap(snippetsFromExploreMarkdown);
  const markdownScope = markdown.flatMap(scopePathsFromExploreMarkdown);
  // Keep a structured fallback for older CodeGraph versions, but the pinned
  // v1.4.1 MCP tool uses the Markdown form above.
  const structuredSnippets = collectSnippetRecords(response);
  const unique = new Map<string, GatherFinalJson["snippets"][number]>();
  for (const snippet of [...markdownSnippets, ...structuredSnippets]) {
    if (snippet.path.startsWith("/") || snippet.path.split(/[\\/]/).includes("..")) continue;
    const key = `${snippet.path}\0${snippet.startLine}\0${snippet.endLine}\0${snippet.why}`;
    unique.set(key, snippet);
  }
  const mapped = [...unique.values()];
  if (mapped.length === 0) {
    return { package_: null, error: "codegraph_explore returned no source blocks with faithful file and line ranges" };
  }
  return {
    package_: {
      // CodeGraph does not emit a one-sentence interpretation separate from its
      // retrieval payload, so preserving the caller's exact request is lossless.
      interpretation: request,
      scope: [...new Set([...markdownScope, ...mapped.map((snippet) => snippet.path)])],
      snippets: mapped,
      // A graph dump records retrieved evidence, not deliberate omissions.
      omissions: [],
    },
  };
}

function emptyRow(job: GatherJob, manifest: RepoManifest, outcome: BankedRow["budget_outcome"], reason: string): CodeGraphRow {
  return {
    job_id: stableJobId(job.dir, job.request),
    request: job.request,
    repo_full: manifest.fullName,
    repo_sha: manifest.sha,
    tags: job.tags,
    full_trajectory: [],
    final_json: null,
    budget_outcome: outcome,
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 0,
    model: "codegraph-explore",
    account: "local",
    ts: new Date().toISOString(),
    valid: false,
    reason,
  };
}

function candidateRow(job: GatherJob, manifest: RepoManifest, package_: GatherFinalJson): CodeGraphRow {
  return {
    ...emptyRow(job, manifest, "natural", ""),
    final_json: package_,
    budget_outcome: "natural",
    reason: undefined,
  };
}

async function runExploreForRepo(
  codegraphDir: string,
  repoDir: string,
  jobs: Array<{ job: GatherJob; manifest: RepoManifest }>,
  scratchRepo: string,
  revision: string,
): Promise<Array<{ row: CodeGraphRow; raw: RawExploreRecord }>> {
  const cli = join(codegraphDir, "dist", "bin", "codegraph.js");
  const client = new McpStdioClient("node", [cli, "serve", "--mcp"], repoDir);
  try {
    const tools = await client.initialize(repoDir);
    const tool = tools.find((candidate) => candidate.name === "codegraph_explore");
    if (!tool) throw new Error(`CodeGraph MCP did not advertise codegraph_explore (tools: ${tools.map((item) => item.name).join(", ")})`);
    const results: Array<{ row: CodeGraphRow; raw: RawExploreRecord }> = [];
    for (const { job, manifest } of jobs) {
      const started = Date.now();
      let response: unknown;
      let row: CodeGraphRow;
      try {
        response = await client.callTool(tool.name, exploreArguments(tool, job.request, repoDir));
        const adapted = adaptExploreResponse(job.request, response);
        row = adapted.package_
          ? candidateRow(job, manifest, adapted.package_)
          : emptyRow(job, manifest, "invalid_final", adapted.error ?? "codegraph_explore returned no package");
      } catch (error) {
        row = emptyRow(job, manifest, "api_error", `codegraph_explore failed: ${errorText(error)}`);
      }
      const wallMs = Date.now() - started;
      results.push({
        row: {
          ...row,
          codegraph_explore: {
            codegraph_revision: revision,
            scratch_repo: scratchRepo,
            wall_ms: wallMs,
            snippet_count: row.final_json?.snippets.length ?? 0,
            snippet_bytes: null,
            hydrated_tokens_o200k: null,
            raw_output_path: "",
          },
        },
        raw: {
          job_id: row.job_id!,
          request: job.request,
          repo_full: manifest.fullName,
          repo_sha: manifest.sha,
          scratch_repo: scratchRepo,
          codegraph_revision: revision,
          wall_ms: wallMs,
          ...(response === undefined ? { error: row.reason ?? "codegraph_explore failed" } : { response }),
        },
      });
    }
    return results;
  } finally {
    await client.close();
  }
}

async function hydrateForMeasurement(repoDir: string, package_: GatherFinalJson): Promise<HydratedMeasurement> {
  const snippets: Array<GatherFinalJson["snippets"][number] & { text: string }> = [];
  let snippetBytes = 0;
  for (const snippet of package_.snippets) {
    const range = await readLineRange(repoDir, snippet.path, snippet.startLine, snippet.endLine);
    snippetBytes += Buffer.byteLength(range.text);
    snippets.push({ ...snippet, text: range.text });
  }
  const hydrated = {
    interpretation: package_.interpretation,
    scope: [...package_.scope],
    snippets,
    omissions: [...package_.omissions],
  };
  // This is the exact hydrated package serialization supplied before the utility judge can request more repository reads.
  return { serialized: JSON.stringify(hydrated, null, 2), snippetBytes, snippetCount: snippets.length };
}

async function tokenizeO200k(serializedPackages: string[]): Promise<number[]> {
  if (serializedPackages.length === 0) return [];
  const program = [
    "import json, sys, tiktoken",
    "encoder = tiktoken.get_encoding('o200k_base')",
    "for line in sys.stdin:",
    "  print(len(encoder.encode(json.loads(line))))",
  ].join("\n");
  const process = Bun.spawn(["python3", "-c", program], { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  await Promise.resolve(process.stdin.write(serializedPackages.map((value) => JSON.stringify(value)).join("\n") + "\n"));
  process.stdin.end();
  const [status, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  if (status !== 0) throw new Error(`tiktoken o200k_base failed: ${stderr.trim()}`);
  const counts = stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => Number(line));
  if (counts.length !== serializedPackages.length || counts.some((count) => !Number.isInteger(count) || count < 0)) {
    throw new Error("tiktoken o200k_base returned an invalid token count sequence");
  }
  return counts;
}

function summarize(values: Array<number | null>): MetricSummary {
  const present = values.filter((value): value is number => value !== null).sort((left, right) => left - right);
  if (present.length === 0) return { count: 0, mean: null, median: null, p95: null };
  const medianIndex = Math.floor(present.length / 2);
  return {
    count: present.length,
    mean: present.reduce((total, value) => total + value, 0) / present.length,
    median: present.length % 2 === 1 ? present[medianIndex]! : (present[medianIndex - 1]! + present[medianIndex]!) / 2,
    p95: present[Math.ceil(present.length * 0.95) - 1]!,
  };
}

function metadataFor(row: CodeGraphRow): CodeGraphMetadata | undefined {
  return row.codegraph_explore;
}

function repoDirForRow(corpusRoot: string, row: BankedRow): string {
  const [owner, repository] = row.repo_full.split("/");
  if (!owner || !repository) throw new Error(`invalid repo_full in row ${row.job_id ?? row.request}: ${row.repo_full}`);
  return join(corpusRoot, `${owner}__${repository}`);
}

async function measureSystem(label: string, rows: BankedRow[], corpusRoot: string): Promise<SystemMetrics> {
  const packages: Array<PackageMeasurement & { serialized: string }> = [];
  for (const row of rows) {
    // This matches the judge loader: forced, invalid, and missing packages are
    // never hydrated for a downstream consumer and must not dilute size metrics.
    if (!row.final_json || !row.valid || row.budget_outcome !== "natural") continue;
    try {
      const hydrated = await hydrateForMeasurement(repoDirForRow(corpusRoot, row), row.final_json);
      packages.push({
        job_id: row.job_id ?? stableJobId(row.repo_full, row.request),
        snippet_count: hydrated.snippetCount,
        snippet_bytes: hydrated.snippetBytes,
        hydrated_tokens_o200k: 0,
        wall_ms: metadataFor(row as CodeGraphRow)?.wall_ms ?? null,
        serialized: hydrated.serialized,
      });
    } catch {
      // A stale comparison row with an unreadable pointer has no consumer package.
    }
  }
  const tokens = await tokenizeO200k(packages.map((item) => item.serialized));
  for (const [index, tokenCount] of tokens.entries()) packages[index]!.hydrated_tokens_o200k = tokenCount;
  return {
    label,
    rows: rows.length,
    package_rows: packages.length,
    missing_or_invalid_rows: rows.length - packages.length,
    hydrated_tokens_o200k: summarize(packages.map((item) => item.hydrated_tokens_o200k)),
    snippet_bytes: summarize(packages.map((item) => item.snippet_bytes)),
    snippet_count: summarize(packages.map((item) => item.snippet_count)),
    wall_ms: summarize(packages.map((item) => item.wall_ms)),
    packages: packages.map(({ serialized: _serialized, ...item }) => item),
  };
}

async function main(): Promise<void> {
  const options = optionsFrom(parseArgs(process.argv.slice(2)));
  await assertCodeGraphRevision(options);
  const jobs = await readJsonl<GatherJob>(options.jobs);
  if (jobs.length === 0) throw new Error("jobs input is empty");

  const grouped = new Map<string, Array<{ job: GatherJob; manifest: RepoManifest; scratchRepo: string }>>();
  for (const job of jobs) {
    const manifest = await loadManifest(job.dir);
    const sourceHead = await gitHead(job.dir);
    if (sourceHead !== manifest.sha) throw new Error(`${job.dir} is ${sourceHead}, expected pinned ${manifest.sha}`);
    const scratchRepo = await scratchRepoFor(job.dir, manifest.sha, options.scratchRoot);
    const group = grouped.get(job.dir) ?? [];
    group.push({ job, manifest, scratchRepo });
    grouped.set(job.dir, group);
  }

  const generated: Array<{ row: CodeGraphRow; raw: RawExploreRecord }> = [];
  for (const group of grouped.values()) {
    const scratchRepo = group[0]!.scratchRepo;
    await indexRepo(options.codegraphDir, scratchRepo);
    generated.push(...(await runExploreForRepo(
      options.codegraphDir,
      scratchRepo,
      group.map(({ job, manifest }) => ({ job, manifest })),
      scratchRepo,
      options.codegraphRevision,
    )));
  }

  for (const item of generated) {
    if (item.row.codegraph_explore) item.row.codegraph_explore.raw_output_path = options.rawOutput;
    if (!item.row.final_json) continue;
    const validation = await validateBankedRow(item.row, jobs.find((job) => stableJobId(job.dir, job.request) === item.row.job_id)!.dir);
    item.row.valid = validation.valid;
    item.row.reason = validation.errors.join("; ") || undefined;
    if (!validation.valid) item.row.budget_outcome = "invalid_final";
  }

  await writeJsonl(options.outputRows, generated.map((item) => item.row));
  await writeJsonl(options.rawOutput, generated.map((item) => item.raw));

  const firstJobDir = jobs[0]!.dir;
  const inferredCorpusRoot = options.corpusRoot ? expandHome(options.corpusRoot) : dirname(firstJobDir);
  const comparisonMetrics: SystemMetrics[] = [];
  for (const comparison of options.comparisonRows) {
    const rows = await readJsonl<BankedRow>(comparison.path);
    comparisonMetrics.push(await measureSystem(comparison.label, rows, inferredCorpusRoot));
  }

  const byJob = new Map(generated.map((item) => [item.row.job_id!, item.row]));
  for (const system of comparisonMetrics) {
    if (system.label !== "codegraph-explore") continue;
    for (const package_ of system.packages) {
      const row = byJob.get(package_.job_id);
      if (!row?.codegraph_explore) continue;
      row.codegraph_explore.snippet_bytes = package_.snippet_bytes;
      row.codegraph_explore.hydrated_tokens_o200k = package_.hydrated_tokens_o200k;
    }
  }
  await writeJsonl(options.outputRows, generated.map((item) => item.row));
  await writeJsonAtomic(options.metricsOutput, {
    method: "codegraph_explore",
    codegraph_revision: options.codegraphRevision,
    tokenizer: {
      name: "tiktoken/o200k_base",
      serialization: "JSON.stringify(hydrateJudgePackage(package), null, 2)",
      definition: "The hydrated package includes interpretation, scope, every snippet's exact source bytes, why string, and omissions.",
    },
    systems: comparisonMetrics,
  });

  const valid = generated.filter((item) => item.row.valid).length;
  const natural = generated.filter((item) => item.row.budget_outcome === "natural").length;
  console.log(JSON.stringify({ lane: "codegraph-candidate", jobs: jobs.length, valid, natural, output_rows: options.outputRows, raw_output: options.rawOutput, metrics_output: options.metricsOutput }));
}

if (import.meta.main) {
  await main();
}

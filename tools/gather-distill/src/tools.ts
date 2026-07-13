import { realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";
import { PRODUCTION_AFT_TOOL_SCHEMAS } from "./aft-tool-catalog.ts";
import { expandHome } from "./repo.ts";
import type { ToolResult } from "./types.ts";
import { isRecord } from "./utils.ts";

export interface ToolDeclaration {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

// These schemas are copied verbatim from the v0.46.0 AFT catalog used by the
// production gather request builder at CortexKit/alfonso commit 3ff7970.
const PRODUCTION_TOOL_NAMES = ["search", "outline", "zoom", "callgraph", "read", "grep", "glob", "inspect", "conflicts"] as const;

export const GATHER_TOOLS: ToolDeclaration[] = PRODUCTION_TOOL_NAMES.map((name) => {
  const inputSchema = PRODUCTION_AFT_TOOL_SCHEMAS[name];
  return {
    name,
    description: inputSchema.description,
    input_schema: inputSchema as unknown as Record<string, unknown>,
  };
});

export const AFT_STORAGE_DIR = "/tmp/gather-campaign-aft";
export const AFT_REQUEST_TIMEOUT_MS = 10 * 60_000;
export const AFT_WARMUP_TIMEOUT_MS = 60_000;
export const DEFAULT_AFT_BINARY = resolve(import.meta.dir, "../bin/aft-v0.46.0");

const AFT_WARMUP_INITIAL_BACKOFF_MS = 100;
const AFT_WARMUP_MAX_BACKOFF_MS = 2_000;
const PATH_ARGUMENT_KEYS = new Set(["filePath", "target", "path", "toFile", "scope", "url"]);

export type AftToolName = "search" | "outline" | "zoom" | "callgraph" | "read" | "grep" | "glob" | "inspect" | "conflicts";

// The production model catalog and the v0.46.0 NDJSON manifest both use
// bare names. Keep this mapping explicit because the two namespaces evolve
// independently even when their current spellings are identical.
const MODEL_TOOL_TO_AFT: Record<string, AftToolName> = {
  search: "search",
  outline: "outline",
  zoom: "zoom",
  callgraph: "callgraph",
  read: "read",
  grep: "grep",
  glob: "glob",
  inspect: "inspect",
  conflicts: "conflicts",
};

export class AftTransportError extends Error {
  constructor(message: string) {
    super(`AFT transport: ${message}`);
    this.name = "AftTransportError";
  }
}

class AftProcessDiedError extends AftTransportError {
  constructor(message: string) {
    super(message);
    this.name = "AftProcessDiedError";
  }
}

class AftRequestTimeoutError extends AftTransportError {
  constructor(message: string) {
    super(message);
    this.name = "AftRequestTimeoutError";
  }
}

class AftProtocolError extends AftTransportError {
  constructor(message: string) {
    super(message);
    this.name = "AftProtocolError";
  }
}

class AftToolCallError extends Error {
  constructor(readonly text: string) {
    super(text);
    this.name = "AftToolCallError";
  }
}

export interface AftProcess {
  stdin: { write(data: string): unknown };
  stdout: ReadableStream<Uint8Array>;
  exited: Promise<number>;
  kill(signal?: number): void;
  pid?: number;
}

export type AftProcessFactory = (binary: string) => AftProcess;

export interface AftClientOptions {
  binary?: string;
  processFactory?: AftProcessFactory;
  requestTimeoutMs?: number;
}

export interface AftCallOptions {
  timeoutMs?: number;
  retryOnDeath?: boolean;
}

interface AftWireResponse {
  id: string;
  success?: boolean;
  text?: string;
  message?: string;
}

interface PendingRequest {
  process: AftProcess;
  requireText: boolean;
  resolve: (response: AftWireResponse) => void;
  reject: (error: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

interface AftWireProtocol {
  configure(id: string, projectRoot: string): Record<string, unknown>;
  toolCall(id: string, name: AftToolName, arguments_: Record<string, unknown>): Record<string, unknown>;
}

// Keep framing in one seam: protocol owners can replace a request envelope
// without changing process lifecycle, pooling, or model-tool mapping.
const DOCUMENTED_NDJSON_PROTOCOL: AftWireProtocol = {
  configure(id, projectRoot) {
    return {
      id,
      command: "configure",
      harness: "opencode",
      project_root: projectRoot,
      session_id: "trainer",
      storage_dir: AFT_STORAGE_DIR,
      config: [
        {
          tier: "user",
          source: "inline:gather-distill",
          doc: JSON.stringify({ semantic_search: false, search_index: true }),
        },
      ],
    };
  },
  toolCall(id, name, arguments_) {
    return {
      id,
      command: "tool_call",
      name,
      arguments: arguments_,
      session_id: "trainer",
    };
  },
};

function spawnAftProcess(binary: string): AftProcess {
  const process = Bun.spawn([binary], { stdin: "pipe", stdout: "pipe", stderr: "ignore" });
  lowerAftProcessPriority(process.pid);
  return process as unknown as AftProcess;
}

function lowerAftProcessPriority(pid: number | undefined): void {
  if (pid === undefined || process.platform === "win32") return;
  try {
    // Start AFT directly, then lower only the child priority without wrapping
    // its command line or adding arguments to the protocol process.
    const renice = Bun.spawn(["renice", "19", "-p", String(pid)], { stdout: "ignore", stderr: "ignore" });
    void renice.exited.catch(() => {});
  } catch {
    // Some platforms do not expose renice to unprivileged processes.
  }
}

export async function canonicalRepoRoot(repoDir: string): Promise<string> {
  return realpath(expandHome(repoDir));
}

function escapesRoot(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate);
  return pathFromRoot === ".." || pathFromRoot.startsWith(`..${sep}`) || isAbsolute(pathFromRoot);
}

function pathGuardError(key: string, requestedPath: string): Error {
  return new Error(`path outside project: ${key} must resolve inside the configured repository (${JSON.stringify(requestedPath)})`);
}

async function validateRepositoryPath(root: string, key: string, requestedPath: string): Promise<void> {
  if (isAbsolute(requestedPath) || requestedPath.startsWith("~") || /^[a-z][a-z\d+.-]*:\/\//i.test(requestedPath)) {
    throw pathGuardError(key, requestedPath);
  }
  const candidate = resolve(root, requestedPath);
  if (escapesRoot(root, candidate)) throw pathGuardError(key, requestedPath);

  let canonicalPath: string;
  try {
    canonicalPath = await realpath(candidate);
  } catch (error) {
    throw new Error(`repository path ${key} could not be resolved: ${errorMessage(error)}`);
  }
  if (escapesRoot(root, canonicalPath)) throw pathGuardError(key, requestedPath);
}

async function validatePathValue(root: string, key: string, value: unknown): Promise<void> {
  const paths = Array.isArray(value) ? value : [value];
  for (const requestedPath of paths) {
    if (typeof requestedPath !== "string") {
      throw new Error(`repository path ${key} must be a string or an array of strings`);
    }
    await validateRepositoryPath(root, key, requestedPath);
  }
}

async function validateToolPaths(root: string, value: unknown): Promise<void> {
  if (Array.isArray(value)) {
    for (const item of value) await validateToolPaths(root, item);
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, nested] of Object.entries(value)) {
    if (PATH_ARGUMENT_KEYS.has(key)) await validatePathValue(root, key, nested);
    else await validateToolPaths(root, nested);
  }
}

function isColdSearchOutput(text: string): boolean {
  const normalized = text.trim();
  const lower = normalized.toLowerCase();
  return (
    lower.includes("fully degraded") ||
    lower.includes("lexical-only fallback returned 0 result") ||
    normalized === "Semantic search is not enabled."
  );
}

function sleep(delayMs: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, delayMs));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRecoverable(error: unknown): error is AftProcessDiedError | AftRequestTimeoutError {
  return error instanceof AftProcessDiedError || error instanceof AftRequestTimeoutError;
}

export class AftClient {
  private readonly binary: string;
  private readonly processFactory: AftProcessFactory;
  private readonly requestTimeoutMs: number;
  private process: AftProcess | undefined;
  private configuredRoot: string | undefined;
  private configuring: Promise<void> | undefined;
  private configuringRoot: string | undefined;
  private readonly pending = new Map<string, PendingRequest>();
  private nextId = 1;
  private activeRoot: string | undefined;
  private activeCalls = 0;
  private switching = false;
  private stateWaiters: Array<() => void> = [];
  private closed = false;

  constructor(options: AftClientOptions = {}) {
    this.binary = options.binary ?? DEFAULT_AFT_BINARY;
    this.processFactory = options.processFactory ?? spawnAftProcess;
    this.requestTimeoutMs = options.requestTimeoutMs ?? AFT_REQUEST_TIMEOUT_MS;
  }

  async configure(repoDir: string, options: AftCallOptions = {}): Promise<void> {
    const root = await canonicalRepoRoot(repoDir);
    const release = await this.reserveRoot(root, options);
    release();
  }

  async call(
    repoDir: string,
    name: AftToolName,
    arguments_: Record<string, unknown>,
    options: AftCallOptions = {},
  ): Promise<string> {
    const root = await canonicalRepoRoot(repoDir);
    await validateToolPaths(root, arguments_);
    const release = await this.reserveRoot(root, options);
    try {
      const attempts = options.retryOnDeath === false ? 1 : 2;
      let lastError: unknown;
      for (let attempt = 0; attempt < attempts; attempt += 1) {
        try {
          await this.configureRoot(root, options);
          const response = await this.sendWithRetry(
            (id) => DOCUMENTED_NDJSON_PROTOCOL.toolCall(id, name, arguments_),
            true,
            { ...options, retryOnDeath: false },
          );
          if (response.success === false) throw new AftToolCallError(response.text ?? response.message ?? "AFT tool call failed");
          return response.text!;
        } catch (error) {
          lastError = error;
          if (!isRecoverable(error) || attempt + 1 === attempts) throw error;
          this.resetProcess(error);
        }
      }
      throw lastError;
    } finally {
      release();
    }
  }

  async warm(
    repoDir: string,
    timeoutMs = AFT_WARMUP_TIMEOUT_MS,
    initialBackoffMs = AFT_WARMUP_INITIAL_BACKOFF_MS,
    maxBackoffMs = AFT_WARMUP_MAX_BACKOFF_MS,
  ): Promise<{ ready: boolean; attempts: number }> {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("AFT warm-up timeout must be positive");
    if (!Number.isFinite(initialBackoffMs) || initialBackoffMs <= 0) throw new Error("AFT warm-up backoff must be positive");
    if (!Number.isFinite(maxBackoffMs) || maxBackoffMs < initialBackoffMs) {
      throw new Error("AFT warm-up maximum backoff must be at least the initial backoff");
    }

    const started = Date.now();
    const remaining = (): number => timeoutMs - (Date.now() - started);
    const options = { retryOnDeath: false } as const;
    await this.configure(repoDir, { ...options, timeoutMs: timeoutMs });

    let attempts = 0;
    let backoffMs = initialBackoffMs;
    for (;;) {
      const requestTimeoutMs = remaining();
      if (requestTimeoutMs <= 0) return { ready: false, attempts };
      let text: string;
      try {
        text = await this.call(
          repoDir,
          "search",
          { query: "function" },
          { ...options, timeoutMs: requestTimeoutMs },
        );
      } catch (error) {
        if (error instanceof AftRequestTimeoutError) return { ready: false, attempts };
        throw error;
      }
      attempts += 1;
      if (!isColdSearchOutput(text)) return { ready: true, attempts };

      const delayMs = Math.min(backoffMs, remaining());
      if (delayMs <= 0) return { ready: false, attempts };
      await sleep(delayMs);
      backoffMs = Math.min(maxBackoffMs, backoffMs * 2);
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    this.resetProcess(new AftTransportError("client closed"));
  }

  private async reserveRoot(root: string, options: AftCallOptions): Promise<() => void> {
    for (;;) {
      if (this.switching || (this.activeCalls > 0 && this.activeRoot !== root)) {
        await this.waitForStateChange();
        continue;
      }
      if (this.activeRoot === root) {
        this.activeCalls += 1;
        return () => this.releaseRoot(root);
      }

      this.switching = true;
      try {
        await this.configureRoot(root, options);
        this.activeRoot = root;
        this.activeCalls = 1;
        return () => this.releaseRoot(root);
      } finally {
        this.switching = false;
        this.notifyStateChange();
      }
    }
  }

  private releaseRoot(root: string): void {
    if (this.activeRoot !== root || this.activeCalls === 0) return;
    this.activeCalls -= 1;
    if (this.activeCalls === 0) this.notifyStateChange();
  }

  private waitForStateChange(): Promise<void> {
    return new Promise((resolve) => this.stateWaiters.push(resolve));
  }

  private notifyStateChange(): void {
    const waiters = this.stateWaiters;
    this.stateWaiters = [];
    for (const resolve of waiters) resolve();
  }

  private async configureRoot(root: string, options: AftCallOptions): Promise<void> {
    if (this.configuredRoot === root && this.process) return;
    if (this.configuring && this.configuringRoot === root) return this.configuring;
    if (this.configuring) {
      await this.configuring;
      return this.configureRoot(root, options);
    }

    const configuring = (async () => {
      const response = await this.sendWithRetry((id) => DOCUMENTED_NDJSON_PROTOCOL.configure(id, root), false, options);
      if (response.success === false) throw new AftProtocolError(response.text ?? response.message ?? "AFT configure failed");
      this.configuredRoot = root;
    })();
    this.configuring = configuring;
    this.configuringRoot = root;
    try {
      await configuring;
    } finally {
      if (this.configuring === configuring) {
        this.configuring = undefined;
        this.configuringRoot = undefined;
      }
    }
  }

  private async sendWithRetry(
    request: (id: string) => Record<string, unknown>,
    requireText: boolean,
    options: AftCallOptions,
  ): Promise<AftWireResponse> {
    const attempts = options.retryOnDeath === false ? 1 : 2;
    let lastError: unknown;
    for (let attempt = 0; attempt < attempts; attempt += 1) {
      try {
        return await this.sendRequest(request(String(this.nextId++)), requireText, options.timeoutMs ?? this.requestTimeoutMs);
      } catch (error) {
        lastError = error;
        if (!isRecoverable(error) || attempt + 1 === attempts) throw error;
        this.resetProcess(error);
      }
    }
    throw lastError;
  }

  private sendRequest(request: Record<string, unknown>, requireText: boolean, timeoutMs: number): Promise<AftWireResponse> {
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) {
      throw new AftRequestTimeoutError(`request timeout must be positive, received ${timeoutMs}`);
    }
    const process = this.ensureProcess();
    const id = String(request.id);
    return new Promise<AftWireResponse>((resolve, reject) => {
      const timer = setTimeout(() => {
        const pending = this.pending.get(id);
        if (!pending) return;
        this.pending.delete(id);
        const error = new AftRequestTimeoutError(`request ${id} exceeded ${timeoutMs}ms`);
        pending.reject(error);
        this.resetProcess(error);
      }, timeoutMs);
      this.pending.set(id, { process, requireText, resolve, reject, timer });
      try {
        Promise.resolve(process.stdin.write(`${JSON.stringify(request)}\n`)).catch((error) => {
          this.resetProcess(new AftProcessDiedError(`stdin write failed: ${errorMessage(error)}`));
        });
      } catch (error) {
        this.resetProcess(new AftProcessDiedError(`stdin write failed: ${errorMessage(error)}`));
      }
    });
  }

  private ensureProcess(): AftProcess {
    if (this.closed) throw new AftTransportError("client is closed");
    if (this.process) return this.process;
    let process: AftProcess;
    try {
      process = this.processFactory(this.binary);
    } catch (error) {
      throw new AftTransportError(`could not start ${this.binary}: ${errorMessage(error)}`);
    }
    this.process = process;
    void this.consumeStdout(process);
    void process.exited.then(
      (code) => this.failProcess(process, new AftProcessDiedError(`process exited with code ${code}`)),
      (error) => this.failProcess(process, new AftProcessDiedError(`process exit failed: ${errorMessage(error)}`)),
    );
    return process;
  }

  private async consumeStdout(process: AftProcess): Promise<void> {
    let buffered = "";
    const decoder = new TextDecoder();
    try {
      for await (const chunk of process.stdout) {
        buffered += decoder.decode(chunk, { stream: true });
        let newline = buffered.indexOf("\n");
        while (newline !== -1) {
          const line = buffered.slice(0, newline).replace(/\r$/, "");
          buffered = buffered.slice(newline + 1);
          if (line.length > 0) this.handleLine(process, line);
          newline = buffered.indexOf("\n");
        }
      }
      buffered += decoder.decode();
      if (buffered.trim().length > 0) this.handleLine(process, buffered.trim());
    } catch (error) {
      this.resetProcess(new AftProcessDiedError(`stdout read failed: ${errorMessage(error)}`));
    }
  }

  private handleLine(process: AftProcess, line: string): void {
    let value: unknown;
    try {
      value = JSON.parse(line);
    } catch {
      this.resetProcess(new AftProtocolError("AFT emitted a non-JSON NDJSON line"));
      return;
    }
    // AFT emits JSON notifications between responses. Only a frame whose id
    // matches an outstanding request can complete that request.
    if (!isRecord(value) || typeof value.id !== "string") return;
    const pending = this.pending.get(value.id);
    if (!pending) return;
    this.pending.delete(value.id);
    clearTimeout(pending.timer);
    if (pending.requireText && typeof value.text !== "string") {
      const detail = typeof value.message === "string" ? value.message : "tool response is missing text";
      pending.reject(new AftProtocolError(detail));
      return;
    }
    pending.resolve({
      id: value.id,
      success: typeof value.success === "boolean" ? value.success : undefined,
      text: typeof value.text === "string" ? value.text : undefined,
      message: typeof value.message === "string" ? value.message : undefined,
    });
  }

  private resetProcess(error: Error): void {
    const process = this.process;
    if (!process) return;
    this.failProcess(process, error);
    try {
      process.kill();
    } catch {
      // The child may already have exited after a timeout or broken pipe.
    }
  }

  private failProcess(process: AftProcess, error: Error): void {
    if (this.process !== process) return;
    this.process = undefined;
    this.configuredRoot = undefined;
    for (const [id, pending] of this.pending) {
      if (pending.process !== process) continue;
      this.pending.delete(id);
      clearTimeout(pending.timer);
      pending.reject(error);
    }
    this.notifyStateChange();
  }
}

interface PoolEntry {
  client: AftClient;
  repoRoot?: string;
  leases: number;
  lastUsed: number;
}

export interface AftClientPoolOptions {
  clientFactory?: () => AftClient;
}

// The pool keeps one process per active repository. When a process becomes
// idle, the least recently used process is reconfigured for the next repo.
export class AftClientPool {
  private readonly entries: PoolEntry[];
  private waiters: Array<() => void> = [];
  private closed = false;

  constructor(size: number, options: AftClientPoolOptions = {}) {
    if (!Number.isInteger(size) || size < 1) throw new Error("AFT client pool size must be at least one");
    const create = options.clientFactory ?? (() => new AftClient());
    this.entries = Array.from({ length: size }, () => ({ client: create(), leases: 0, lastUsed: 0 }));
  }

  async withClient<T>(repoDir: string, work: (client: AftClient) => Promise<T>): Promise<T> {
    const entry = await this.acquire(await canonicalRepoRoot(repoDir));
    try {
      return await work(entry.client);
    } finally {
      entry.leases -= 1;
      entry.lastUsed = Date.now();
      this.notify();
    }
  }

  async close(): Promise<void> {
    this.closed = true;
    this.notify();
    await Promise.all(this.entries.map((entry) => entry.client.close()));
  }

  private async acquire(repoRoot: string): Promise<PoolEntry> {
    for (;;) {
      if (this.closed) throw new AftTransportError("client pool is closed");
      const existing = this.entries.find((entry) => entry.repoRoot === repoRoot);
      if (existing) {
        existing.leases += 1;
        return existing;
      }
      const idle = this.entries.filter((entry) => entry.leases === 0).sort((a, b) => a.lastUsed - b.lastUsed)[0];
      if (idle) {
        idle.repoRoot = repoRoot;
        idle.leases = 1;
        return idle;
      }
      await new Promise<void>((resolve) => this.waiters.push(resolve));
    }
  }

  private notify(): void {
    const waiters = this.waiters;
    this.waiters = [];
    for (const resolve of waiters) resolve();
  }
}

export interface AftWarmupResult {
  ok: boolean;
  durationMs: number;
  error?: string;
  timedOut?: boolean;
  warning?: string;
  searchAttempts?: number;
}

export interface AftWarmupCoordinatorOptions {
  timeoutMs?: number;
  initialBackoffMs?: number;
  maxBackoffMs?: number;
}

export class AftWarmupCoordinator {
  private readonly warmups = new Map<string, Promise<AftWarmupResult>>();
  private readonly timeoutMs: number;
  private readonly initialBackoffMs: number;
  private readonly maxBackoffMs: number;

  constructor(options: AftWarmupCoordinatorOptions = {}) {
    this.timeoutMs = options.timeoutMs ?? AFT_WARMUP_TIMEOUT_MS;
    this.initialBackoffMs = options.initialBackoffMs ?? AFT_WARMUP_INITIAL_BACKOFF_MS;
    this.maxBackoffMs = options.maxBackoffMs ?? AFT_WARMUP_MAX_BACKOFF_MS;
  }

  async ensureWarmed(repoDir: string, client: AftClient): Promise<AftWarmupResult> {
    const root = await canonicalRepoRoot(repoDir);
    let warmup = this.warmups.get(root);
    if (!warmup) {
      warmup = (async () => {
        const started = Date.now();
        try {
          const search = await client.warm(root, this.timeoutMs, this.initialBackoffMs, this.maxBackoffMs);
          const durationMs = Date.now() - started;
          if (!search.ready) {
            return {
              ok: true,
              durationMs,
              timedOut: true,
              searchAttempts: search.attempts,
              warning: `AFT search index remained degraded after ${durationMs}ms; proceeding with cold search`,
            };
          }
          return { ok: true, durationMs, searchAttempts: search.attempts };
        } catch (error) {
          return {
            ok: false,
            durationMs: Date.now() - started,
            error: errorMessage(error),
            timedOut: error instanceof AftRequestTimeoutError,
          };
        }
      })();
      this.warmups.set(root, warmup);
    }
    return warmup;
  }
}

let defaultClient: AftClient | undefined;

function sharedClient(): AftClient {
  defaultClient ??= new AftClient();
  return defaultClient;
}

export async function executeTool(
  repoDir: string,
  name: string,
  rawInput: unknown,
  client: AftClient = sharedClient(),
): Promise<ToolResult> {
  try {
    if (!isRecord(rawInput)) throw new Error("tool input must be an object");
    const aftName = MODEL_TOOL_TO_AFT[name];
    if (!aftName) throw new Error(`unknown production read-only tool: ${name}`);
    return { ok: true, output: await client.call(repoDir, aftName, rawInput) };
  } catch (error) {
    if (error instanceof AftTransportError) throw error;
    if (error instanceof AftToolCallError) return { ok: false, output: error.text, error: error.text };
    const message = errorMessage(error);
    return { ok: false, output: message, error: message };
  }
}

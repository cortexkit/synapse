import { readFile } from "node:fs/promises";
import { homedir, platform, release } from "node:os";
import { join } from "node:path";
import type { MessageRequest, MessageResponse } from "./anthropic.ts";
import { openAiResponsesBody, openAiResponsesResponseShape, readOpenAiResponsesStream } from "./openai-responses.ts";
import { sleep, isRecord } from "./utils.ts";

export const OPENAI_CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses";
const DEFAULT_AUTH_FILES = [
  join(homedir(), ".local/share/opencode/auth.json"),
  join(homedir(), ".config/opencode/auth.json"),
];
const DEFAULT_TIMEOUT_MS = 300_000;
const DEFAULT_TRANSIENT_RETRIES = 2;
const DEFAULT_EXPIRED_RETRIES = 5;
const DEFAULT_EXPIRED_WAIT_MS = 10_000;

export interface OpenAiOAuthRequestOptions {
  authFile?: string;
  requestTimeoutMs?: number;
  transientRetries?: number;
  expiredRetries?: number;
  expiredWaitMs?: number;
  sessionId?: string;
}

interface OAuthEntry {
  access: string;
  expires: number;
}

interface JwtClaims {
  chatgpt_account_id?: string;
  organizations?: Array<{ id?: string }>;
  [key: string]: unknown;
}

export class OpenAiOAuthExpiredError extends Error {
  readonly authFile: string;

  constructor(authFile: string) {
    super(`OpenAI OAuth access token is expired; wait for OpenCode to refresh ${authFile} and retry`);
    this.name = "OpenAiOAuthExpiredError";
    this.authFile = authFile;
  }
}

export class OpenAiOAuthHttpError extends Error {
  readonly status: number;

  constructor(status: number, body: string) {
    super(`OpenAI OAuth Responses request failed (${status})${body ? `: ${body.slice(0, 1_000)}` : ""}`);
    this.name = "OpenAiOAuthHttpError";
    this.status = status;
  }
}

function expandHome(path: string): string {
  return path === "~" ? homedir() : path.startsWith("~/") ? join(homedir(), path.slice(2)) : path;
}

function parseClaims(token: string): JwtClaims | undefined {
  const parts = token.split(".");
  if (parts.length !== 3) return undefined;
  try {
    const parsed: unknown = JSON.parse(Buffer.from(parts[1]!, "base64url").toString("utf8"));
    return isRecord(parsed) ? parsed as JwtClaims : undefined;
  } catch {
    return undefined;
  }
}

function accountIdFromAccess(access: string): string | undefined {
  const claims = parseClaims(access);
  if (!claims) return undefined;
  const nested = claims["https://api.openai.com/auth"];
  return claims.chatgpt_account_id
    ?? (isRecord(nested) && typeof nested.chatgpt_account_id === "string" ? nested.chatgpt_account_id : undefined)
    ?? claims.organizations?.find((organization) => typeof organization.id === "string")?.id;
}

async function readAuthFile(path: string): Promise<OAuthEntry | undefined> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
    throw new Error(`cannot read OpenCode auth file ${path}: ${error instanceof Error ? error.message : String(error)}`);
  }
  const openai = isRecord(parsed) ? parsed.openai : undefined;
  if (!isRecord(openai) || openai.type !== "oauth") return undefined;
  if (typeof openai.access !== "string" || openai.access.length === 0 || typeof openai.expires !== "number") {
    throw new Error(`OpenCode auth file ${path} has an incomplete OpenAI OAuth entry`);
  }
  return { access: openai.access, expires: openai.expires };
}

async function readFreshOAuth(options: OpenAiOAuthRequestOptions): Promise<{ entry: OAuthEntry; authFile: string }> {
  const paths = options.authFile ? [expandHome(options.authFile)] : DEFAULT_AUTH_FILES;
  for (const path of paths) {
    const entry = await readAuthFile(path);
    if (entry) return { entry, authFile: path };
  }
  throw new Error(`no OpenAI OAuth entry found in ${paths.join(" or ")}`);
}

/** Read access credentials without ever exchanging or refreshing the refresh token. */
export async function readFreshOpenAiOAuth(options: OpenAiOAuthRequestOptions = {}): Promise<{
  access: string;
  accountId?: string;
  authFile: string;
}> {
  const retries = options.expiredRetries ?? DEFAULT_EXPIRED_RETRIES;
  const waitMs = options.expiredWaitMs ?? DEFAULT_EXPIRED_WAIT_MS;
  if (!Number.isInteger(retries) || retries < 0) throw new Error("OpenAI OAuth expired retries must be a non-negative integer");
  if (!Number.isFinite(waitMs) || waitMs < 0) throw new Error("OpenAI OAuth expired wait must be non-negative");

  for (let attempt = 0; ; attempt += 1) {
    const current = await readFreshOAuth(options);
    if (current.entry.expires > Date.now() + 1_000) {
      return {
        access: current.entry.access,
        accountId: accountIdFromAccess(current.entry.access),
        authFile: current.authFile,
      };
    }
    if (attempt >= retries) throw new OpenAiOAuthExpiredError(current.authFile);
    await sleep(waitMs);
  }
}

function requestHeaders(access: string, accountId: string | undefined, sessionId: string): Headers {
  const headers = new Headers({
    authorization: `Bearer ${access}`,
    "content-type": "application/json",
    originator: "opencode",
    "User-Agent": `opencode/local (${platform()} ${release()}; ${process.arch})`,
    "session-id": sessionId,
  });
  if (accountId) headers.set("ChatGPT-Account-Id", accountId);
  return headers;
}

async function fetchWithTimeout(url: string, body: string, timeoutMs: number, headers: Headers): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { method: "POST", headers, body, signal: controller.signal });
  } catch (error) {
    if (controller.signal.aborted) throw new Error(`OpenAI OAuth request timed out after ${timeoutMs}ms`);
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

function transientStatus(status: number): boolean {
  return status === 408 || status === 409 || status === 425 || status === 429 || status >= 500;
}

function transientError(error: unknown): boolean {
  return error instanceof OpenAiOAuthHttpError
    ? transientStatus(error.status)
    : error instanceof TypeError || (error instanceof Error && error.name === "AbortError");
}

/** Send one streaming Codex Responses request with the access token OpenCode owns. */
export async function sendOpenAiOAuthMessage(
  request: MessageRequest,
  options: OpenAiOAuthRequestOptions = {},
): Promise<MessageResponse> {
  const timeoutMs = options.requestTimeoutMs ?? DEFAULT_TIMEOUT_MS;
  const retries = options.transientRetries ?? DEFAULT_TRANSIENT_RETRIES;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error("OpenAI OAuth request timeout must be positive");
  if (!Number.isInteger(retries) || retries < 0) throw new Error("OpenAI OAuth transient retries must be a non-negative integer");
  const sessionId = options.sessionId ?? crypto.randomUUID();
  const body = JSON.stringify(openAiResponsesBody(request));

  for (let attempt = 0; ; attempt += 1) {
    // Re-read auth.json for every request so OpenCode remains the only token-family refresher.
    const credential = await readFreshOpenAiOAuth(options);
    try {
      const response = await fetchWithTimeout(
        OPENAI_CODEX_RESPONSES_URL,
        body,
        timeoutMs,
        requestHeaders(credential.access, credential.accountId, sessionId),
      );
      if (!response.ok) throw new OpenAiOAuthHttpError(response.status, await response.text());
      return openAiResponsesResponseShape(await readOpenAiResponsesStream(response));
    } catch (error) {
      if (!transientError(error) || attempt >= retries) throw error;
      await sleep(250 * 2 ** attempt);
    }
  }
}

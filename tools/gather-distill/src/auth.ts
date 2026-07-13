import { readFile, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { isRecord, sleep } from "./utils.ts";

export interface Credential {
  name: string;
  secret: string;
  kind: "oauth" | "api_key";
}

interface AccountState {
  inFlight: number;
  cooldownUntil: number;
}

export interface AccountLease {
  credential: Credential;
  released: boolean;
}

function parseAccounts(value: unknown): Credential[] {
  const source = Array.isArray(value)
    ? value
    : isRecord(value) && Array.isArray(value.accounts)
      ? value.accounts
      : isRecord(value) && isRecord(value.accounts)
        ? Object.entries(value.accounts).map(([name, token]) =>
            isRecord(token) ? { name, ...token } : { name, access_token: token },
          )
        : isRecord(value)
          ? Object.entries(value).map(([name, token]) =>
              isRecord(token) ? { name, ...token } : { name, access_token: token },
            )
          : [];
  const credentials: Credential[] = [];
  for (const [index, item] of source.entries()) {
    if (!isRecord(item)) continue;
    const secret = item.access_token ?? item.accessToken ?? item.token;
    if (typeof secret !== "string" || secret.trim().length === 0) continue;
    const name = typeof item.name === "string" && item.name.trim() ? item.name.trim() : `account-${index + 1}`;
    credentials.push({ name, secret, kind: "oauth" });
  }
  const seen = new Set<string>();
  const unique = credentials.filter((credential) => {
    if (seen.has(credential.name)) throw new Error(`duplicate account name: ${credential.name}`);
    seen.add(credential.name);
    return true;
  });
  if (unique.length === 0) throw new Error("credentials file contains no usable access tokens");
  return unique;
}

export class CredentialStore {
  readonly filePath: string;
  readonly cacheMs: number;
  #cached: Credential[] | null = null;
  #loadedAt = 0;
  #mtimeMs = -1;

  constructor(filePath = process.env.GATHER_DISTILL_ACCOUNTS_FILE ?? "./accounts.json", cacheMs = 30_000) {
    this.filePath = resolve(filePath);
    this.cacheMs = cacheMs;
  }

  async load(force = false): Promise<Credential[]> {
    const apiKey = process.env.GATHER_DISTILL_API_KEY;
    if (apiKey) return [{ name: "api-key", secret: apiKey, kind: "api_key" }];
    const now = Date.now();
    if (!force && this.#cached && now - this.#loadedAt < this.cacheMs) return this.#cached;
    const metadata = await stat(this.filePath);
    if (!force && this.#cached && metadata.mtimeMs === this.#mtimeMs) {
      this.#loadedAt = now;
      return this.#cached;
    }
    const parsed = parseAccounts(JSON.parse(await readFile(this.filePath, "utf8")));
    this.#cached = parsed;
    this.#mtimeMs = metadata.mtimeMs;
    this.#loadedAt = now;
    return parsed;
  }
}

export class AccountPool {
  readonly store: CredentialStore;
  readonly inFlightCap: number;
  readonly cooldownMs: number;
  #states = new Map<string, AccountState>();
  #cursor = 0;

  constructor(options: {
    store?: CredentialStore;
    inFlightCap?: number;
    cooldownMs?: number;
  } = {}) {
    this.store = options.store ?? new CredentialStore();
    this.inFlightCap = options.inFlightCap ?? Number(process.env.GATHER_DISTILL_ACCOUNT_INFLIGHT ?? 2);
    this.cooldownMs = options.cooldownMs ?? Number(process.env.GATHER_DISTILL_AUTH_COOLDOWN_MS ?? 300_000);
    if (!Number.isInteger(this.inFlightCap) || this.inFlightCap < 1) {
      throw new Error("account in-flight cap must be a positive integer");
    }
  }

  async acquire(exclude = new Set<string>()): Promise<AccountLease> {
    for (;;) {
      const credentials = await this.store.load();
      const now = Date.now();
      for (let offset = 0; offset < credentials.length; offset += 1) {
        const index = (this.#cursor + offset) % credentials.length;
        const credential = credentials[index];
        if (exclude.has(credential.name)) continue;
        const state = this.#states.get(credential.name) ?? { inFlight: 0, cooldownUntil: 0 };
        this.#states.set(credential.name, state);
        if (state.inFlight >= this.inFlightCap || state.cooldownUntil > now) continue;
        state.inFlight += 1;
        this.#cursor = (index + 1) % credentials.length;
        return { credential, released: false };
      }
      await sleep(100);
    }
  }

  release(lease: AccountLease): void {
    if (lease.released) return;
    lease.released = true;
    const state = this.#states.get(lease.credential.name);
    if (state) state.inFlight = Math.max(0, state.inFlight - 1);
  }

  async coolDown(lease: AccountLease): Promise<void> {
    const state = this.#states.get(lease.credential.name) ?? { inFlight: 0, cooldownUntil: 0 };
    state.cooldownUntil = Date.now() + this.cooldownMs;
    this.#states.set(lease.credential.name, state);
    this.release(lease);
    await this.store.load(true);
  }
}

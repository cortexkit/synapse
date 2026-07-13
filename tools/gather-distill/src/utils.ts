import { appendFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function parseJsonText(text: string): unknown {
  const trimmed = text.trim();
  // The model may emit a prose preamble ("I have the picture...") before a
  // ```json fenced block, and/or a trailing sign-off — so the fence is not
  // anchored to the whole string. Prefer the LAST fenced block (the final
  // answer), then fall back to the outermost brace span, then the raw text.
  const fences = [...trimmed.matchAll(/```(?:json)?\s*([\s\S]*?)\s*```/gi)];
  if (fences.length > 0) {
    return JSON.parse(fences[fences.length - 1]![1]!);
  }
  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return JSON.parse(trimmed.slice(firstBrace, lastBrace + 1));
  }
  return JSON.parse(trimmed);
}

export async function readJsonl<T>(path: string): Promise<T[]> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .map((line, index) => {
      try {
        return JSON.parse(line) as T;
      } catch (error) {
        throw new Error(`${path}:${index + 1}: invalid JSON: ${String(error)}`);
      }
    });
}

export async function appendJsonl(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, `${JSON.stringify(value)}\n`, { encoding: "utf8", mode: 0o600 });
}

export async function writeJsonl(path: string, values: unknown[]): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const body = values.map((value) => JSON.stringify(value)).join("\n");
  await writeFile(path, body.length > 0 ? `${body}\n` : "", { encoding: "utf8", mode: 0o600 });
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
  await rename(temporary, path);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function stableJobId(dir: string, request: string): string {
  const normalized = `${dir.replace(/\/+$/, "")}\n${request.trim()}`;
  return new Bun.CryptoHasher("sha256").update(normalized).digest("hex");
}

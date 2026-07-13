import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, relative, resolve, sep } from "node:path";
import type { RepoManifest } from "./types.ts";
import { isRecord } from "./utils.ts";

export const DEFAULT_CORPUS_ROOT = "~/Work/OSS/gather-corpus";

export function expandHome(path: string): string {
  if (path === "~") return homedir();
  return path.startsWith(`~${sep}`) ? join(homedir(), path.slice(2)) : path;
}

export async function loadManifest(repoDir: string): Promise<RepoManifest> {
  const path = join(expandHome(repoDir), ".gather-corpus-manifest.json");
  const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
  if (
    !isRecord(parsed) ||
    typeof parsed.fullName !== "string" ||
    typeof parsed.sha !== "string" ||
    typeof parsed.language !== "string" ||
    typeof parsed.size_mb !== "number"
  ) {
    throw new Error(`${path}: expected fullName, sha, language, and numeric size_mb`);
  }
  if (!/^[0-9a-f]{40}$/i.test(parsed.sha)) {
    throw new Error(`${path}: sha must be a full 40-character commit hash`);
  }
  return parsed as unknown as RepoManifest;
}

export async function verifyPinnedHead(repoDir: string, expectedSha: string): Promise<void> {
  const process = Bun.spawn(["git", "-C", expandHome(repoDir), "rev-parse", "HEAD"], {
    stdout: "pipe",
    stderr: "pipe",
  });
  const [status, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  if (status !== 0) throw new Error(`cannot read repository HEAD: ${stderr.trim()}`);
  if (stdout.trim() !== expectedSha) {
    throw new Error(`pinned clone HEAD ${stdout.trim()} does not match manifest sha ${expectedSha}`);
  }
}

async function resolveRepoEntry(repoDir: string, requestedPath: string): Promise<string> {
  if (requestedPath.length === 0 || isAbsolute(requestedPath)) {
    throw new Error("path must be a non-empty repository-relative path");
  }
  const root = await realpath(expandHome(repoDir));
  const candidate = resolve(root, requestedPath);
  const lexical = relative(root, candidate);
  if (lexical === ".." || lexical.startsWith(`..${sep}`) || isAbsolute(lexical)) {
    throw new Error("path escapes the pinned repository");
  }
  const resolved = await realpath(candidate);
  const canonical = relative(root, resolved);
  if (canonical === ".." || canonical.startsWith(`..${sep}`) || isAbsolute(canonical)) {
    throw new Error("symlink escapes the pinned repository");
  }
  return resolved;
}

export async function resolveRepoFile(repoDir: string, requestedPath: string): Promise<string> {
  const resolved = await resolveRepoEntry(repoDir, requestedPath);
  const metadata = await stat(resolved);
  if (!metadata.isFile()) throw new Error("path is not a file");
  return resolved;
}



export function splitLinesInclusive(text: string): string[] {
  return text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
}

export async function readLineRange(
  repoDir: string,
  path: string,
  startLine: number,
  endLine: number,
): Promise<{ text: string; lineCount: number }> {
  const resolved = await resolveRepoFile(repoDir, path);
  const bytes = await readFile(resolved);
  if (bytes.includes(0)) throw new Error("binary files cannot be read as source text");
  const lines = splitLinesInclusive(bytes.toString("utf8"));
  if (!Number.isInteger(startLine) || !Number.isInteger(endLine) || startLine < 1) {
    throw new Error("line ranges use positive 1-based integers");
  }
  if (endLine < startLine) throw new Error("endLine must be at least startLine");
  if (endLine > lines.length) {
    throw new Error(`line range ${startLine}-${endLine} exceeds file length ${lines.length}`);
  }
  return { text: lines.slice(startLine - 1, endLine).join(""), lineCount: lines.length };
}

export async function discoverRepos(corpusRoot = DEFAULT_CORPUS_ROOT): Promise<string[]> {
  const root = expandHome(corpusRoot);
  const entries = await readdir(root, { withFileTypes: true });
  const repos: string[] = [];
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue;
    const dir = join(root, entry.name);
    try {
      await loadManifest(dir);
      repos.push(dir);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return repos;
}

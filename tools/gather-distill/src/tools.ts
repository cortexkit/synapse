import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { basename, join, relative } from "node:path";
import {
  expandHome,
  readLineRange,
  resolveRepoDirectory,
  resolveRepoFile,
  splitLinesInclusive,
} from "./repo.ts";
import type { LocalToolResult, ToolProvenance } from "./types.ts";
import { isRecord } from "./utils.ts";

export interface ToolDeclaration {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export const GATHER_TOOLS: ToolDeclaration[] = [
  {
    name: "search",
    description: "Search repository source by concept, identifier, literal, or regex and return ranked hits.",
    input_schema: {
      type: "object",
      properties: {
        query: { type: "string" },
        hint: { type: "string", enum: ["auto", "semantic", "literal", "regex"] },
        path: { type: "string" },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "outline",
    description: "List a directory tree or the declarations and line numbers in a source file.",
    input_schema: {
      type: "object",
      properties: { target: { type: "string" }, files: { type: "boolean" } },
      required: ["target"],
      additionalProperties: false,
    },
  },
  {
    name: "zoom",
    description: "Locate a named symbol in one file and return its nearby source range.",
    input_schema: {
      type: "object",
      properties: {
        filePath: { type: "string" },
        symbols: { oneOf: [{ type: "string" }, { type: "array", items: { type: "string" }, minItems: 1 }] },
      },
      required: ["filePath", "symbols"],
      additionalProperties: false,
    },
  },
  {
    name: "callgraph",
    description: "Find repository definitions and lexical call or reference sites for a symbol.",
    input_schema: {
      type: "object",
      properties: { symbol: { type: "string" }, path: { type: "string" } },
      required: ["symbol"],
      additionalProperties: false,
    },
  },
  {
    name: "read",
    description: "Read a repository-relative source file or an inclusive 1-based line range.",
    input_schema: {
      type: "object",
      properties: {
        filePath: { type: "string" },
        startLine: { type: "integer", minimum: 1 },
        endLine: { type: "integer", minimum: 1 },
        offset: { type: "integer", minimum: 1 },
        limit: { type: "integer", minimum: 1 },
      },
      required: ["filePath"],
      additionalProperties: false,
    },
  },
  {
    name: "grep",
    description: "Search source text with a regular expression and return path, line, and matching text.",
    input_schema: {
      type: "object",
      properties: {
        pattern: { type: "string" },
        path: { type: "string" },
        include: { type: "string" },
      },
      required: ["pattern"],
      additionalProperties: false,
    },
  },
  {
    name: "glob",
    description: "Find repository files matching a glob pattern.",
    input_schema: {
      type: "object",
      properties: { pattern: { type: "string" }, path: { type: "string" } },
      required: ["pattern"],
      additionalProperties: false,
    },
  },
  {
    name: "tree",
    description: "Return a bounded repository file tree below a directory.",
    input_schema: {
      type: "object",
      properties: { path: { type: "string" }, depth: { type: "integer", minimum: 1, maximum: 8 } },
      additionalProperties: false,
    },
  },
];

const SKIP_DIRS = new Set([".git", "node_modules", "target", ".venv", "vendor"]);
const SOURCE_LIMIT_BYTES = 2 * 1024 * 1024;
const RESULT_LIMIT = 100;

async function walkFiles(repoDir: string, subpath = ".", maxDepth = 20): Promise<string[]> {
  const root = await realpath(expandHome(repoDir));
  let base: string;
  try {
    base = await resolveRepoDirectory(repoDir, subpath);
  } catch {
    const file = await resolveRepoFile(repoDir, subpath);
    return [relative(root, file)];
  }
  const files: string[] = [];
  async function visit(directory: string, depth: number): Promise<void> {
    if (depth > maxDepth || files.length >= 20_000) return;
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name === ".gather-corpus-manifest.json") continue;
      const absolute = join(directory, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRS.has(entry.name)) await visit(absolute, depth + 1);
      } else if (entry.isFile()) {
        files.push(relative(root, absolute));
      }
      if (files.length >= 20_000) return;
    }
  }
  await visit(base, 0);
  return files;
}

function globRegex(pattern: string): RegExp {
  let source = "^";
  for (let index = 0; index < pattern.length; index += 1) {
    const character = pattern[index];
    if (character === "*") {
      if (pattern[index + 1] === "*") {
        source += pattern[index + 2] === "/" ? "(?:.*/)?" : ".*";
        index += pattern[index + 2] === "/" ? 2 : 1;
      } else source += "[^/]*";
    } else if (character === "?") source += "[^/]";
    else source += character.replace(/[\\^$+?.()|{}\[\]]/g, "\\$&");
  }
  return new RegExp(`${source}$`);
}

function numbered(text: string, startLine: number): string {
  return splitLinesInclusive(text)
    .map((line, index) => `${startLine + index}: ${line.replace(/\n$/, "")}`)
    .join("\n");
}

async function safeText(repoDir: string, path: string): Promise<string | null> {
  const resolved = await resolveRepoFile(repoDir, path);
  const metadata = await stat(resolved);
  if (metadata.size > SOURCE_LIMIT_BYTES) return null;
  const bytes = await readFile(resolved);
  if (bytes.includes(0)) return null;
  return bytes.toString("utf8");
}

async function readTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const filePath = String(input.filePath ?? "");
  const resolved = await resolveRepoFile(repoDir, filePath);
  const bytes = await readFile(resolved);
  if (bytes.includes(0)) throw new Error("binary files cannot be read as source text");
  const lines = splitLinesInclusive(bytes.toString("utf8"));
  const start = Number(input.startLine ?? input.offset ?? 1);
  const defaultEnd = input.limit === undefined ? lines.length : start + Number(input.limit) - 1;
  const requestedEnd = Number(input.endLine ?? defaultEnd);
  const end = Math.min(requestedEnd, lines.length);
  if (end < start) throw new Error(`startLine ${start} exceeds file length ${lines.length}`);
  let actualEnd = end;
  let text = (await readLineRange(repoDir, filePath, start, actualEnd)).text;
  while (Buffer.byteLength(text) > 50_000 && actualEnd > start) {
    actualEnd = Math.max(start, actualEnd - Math.max(1, Math.floor((actualEnd - start) / 2)));
    text = (await readLineRange(repoDir, filePath, start, actualEnd)).text;
  }
  if (Buffer.byteLength(text) > 50_000) throw new Error("single line exceeds the 50KB read limit");
  return {
    ok: true,
    output: `${filePath}:${start}-${actualEnd}\n${numbered(text, start)}`,
    provenance: [{ path: filePath, startLine: start, endLine: actualEnd, text }],
  };
}

async function grepTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const pattern = String(input.pattern ?? "");
  const regex = new RegExp(pattern);
  const include = input.include ? globRegex(String(input.include)) : null;
  const files = await walkFiles(repoDir, String(input.path ?? "."));
  const hits: string[] = [];
  const provenance: ToolProvenance[] = [];
  for (const path of files) {
    if (include && !include.test(path) && !include.test(basename(path))) continue;
    const text = await safeText(repoDir, path);
    if (text === null) continue;
    const lines = splitLinesInclusive(text);
    for (let index = 0; index < lines.length; index += 1) {
      regex.lastIndex = 0;
      if (!regex.test(lines[index])) continue;
      const line = lines[index];
      hits.push(`${path}:${index + 1}:${line.replace(/\r?\n$/, "")}`);
      provenance.push({ path, startLine: index + 1, endLine: index + 1, text: line });
      if (hits.length >= RESULT_LIMIT) break;
    }
    if (hits.length >= RESULT_LIMIT) break;
  }
  return { ok: true, output: hits.join("\n") || "No matches", provenance };
}

async function searchTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const query = String(input.query ?? "").trim();
  const hint = String(input.hint ?? "auto");
  if (hint === "regex") return grepTool(repoDir, { pattern: query, path: input.path });
  const terms = query.toLowerCase().match(/[a-z0-9_.:-]+/g) ?? [];
  if (terms.length === 0) throw new Error("search query has no searchable terms");
  const files = await walkFiles(repoDir, String(input.path ?? "."));
  const scored: Array<{ score: number; path: string; line: number; text: string; raw: string }> = [];
  for (const path of files) {
    const text = await safeText(repoDir, path);
    if (text === null) continue;
    const lines = splitLinesInclusive(text);
    for (let index = 0; index < lines.length; index += 1) {
      const lower = lines[index].toLowerCase();
      const matches = terms.filter((term) => lower.includes(term)).length;
      if (matches === 0) continue;
      const exactBonus = lower.includes(query.toLowerCase()) ? terms.length : 0;
      scored.push({ score: matches + exactBonus, path, line: index + 1, text: lines[index].trim(), raw: lines[index] });
    }
  }
  scored.sort((a, b) => b.score - a.score || a.path.localeCompare(b.path) || a.line - b.line);
  const selected = scored.slice(0, RESULT_LIMIT);
  return {
    ok: true,
    output:
      selected.map((hit) => `${hit.path}:${hit.line} [score ${hit.score}] ${hit.text}`).join("\n") ||
      "No matches",
    provenance: selected.map((hit) => ({
      path: hit.path,
      startLine: hit.line,
      endLine: hit.line,
      text: hit.raw,
    })),
  };
}

const DECLARATION_PATTERNS = [
  /^\s*(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|enum|struct|trait|fn|def)\s+([A-Za-z_$][\w$]*)/,
  /^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][\w]*)/,
  /^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=/,
];

function declarations(text: string): Array<{ symbol: string; line: number; text: string; raw: string }> {
  const found: Array<{ symbol: string; line: number; text: string; raw: string }> = [];
  for (const [index, raw] of splitLinesInclusive(text).entries()) {
    for (const pattern of DECLARATION_PATTERNS) {
      const match = raw.match(pattern);
      if (match) {
        found.push({ symbol: match[1], line: index + 1, text: raw.trim(), raw });
        break;
      }
    }
  }
  return found;
}

async function outlineTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const target = String(input.target ?? ".");
  try {
    await resolveRepoDirectory(repoDir, target);
    const files = await walkFiles(repoDir, target, 8);
    return { ok: true, output: files.slice(0, 500).join("\n") };
  } catch (error) {
    if (input.files === true) throw error;
  }
  const text = await safeText(repoDir, target);
  if (text === null) throw new Error("file is binary or too large to outline");
  const found = declarations(text);
  return {
    ok: true,
    output: found.map((item) => `${target}:${item.line} ${item.text}`).join("\n") || "No declarations found",
    provenance: found.map((item) => ({ path: target, startLine: item.line, endLine: item.line, text: item.raw })),
  };
}

async function zoomTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const filePath = String(input.filePath ?? "");
  const requested = input.symbols ?? input.symbol;
  const symbol = Array.isArray(requested) ? String(requested[0] ?? "") : String(requested ?? "");
  const text = await safeText(repoDir, filePath);
  if (text === null) throw new Error("file is binary or too large to zoom");
  const declaration = declarations(text).find((item) => item.symbol === symbol);
  if (!declaration) throw new Error(`symbol ${symbol} not found in ${filePath}`);
  const lineCount = splitLinesInclusive(text).length;
  const startLine = Math.max(1, declaration.line - 3);
  const endLine = Math.min(lineCount, declaration.line + 30);
  const range = await readLineRange(repoDir, filePath, startLine, endLine);
  return {
    ok: true,
    output: `${filePath}:${startLine}-${endLine}\n${numbered(range.text, startLine)}`,
    provenance: [{ path: filePath, startLine, endLine, text: range.text }],
  };
}

async function callgraphTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const symbol = String(input.symbol ?? "");
  if (!/^[A-Za-z_$][\w$]*$/.test(symbol)) throw new Error("symbol must be an identifier");
  const result = await grepTool(repoDir, {
    pattern: `\\b${symbol.replace(/[$]/g, "\\$")}\\b`,
    path: input.path,
  });
  result.output = `Lexical definitions and references for ${symbol}:\n${result.output}`;
  return result;
}

async function globTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const pattern = globRegex(String(input.pattern ?? ""));
  const files = await walkFiles(repoDir, String(input.path ?? "."));
  return { ok: true, output: files.filter((path) => pattern.test(path)).slice(0, 1000).join("\n") };
}

async function treeTool(repoDir: string, input: Record<string, unknown>): Promise<LocalToolResult> {
  const path = String(input.path ?? ".");
  const depth = Number(input.depth ?? 4);
  const files = await walkFiles(repoDir, path, depth);
  return { ok: true, output: files.slice(0, 1000).join("\n") };
}

export async function executeTool(repoDir: string, name: string, rawInput: unknown): Promise<LocalToolResult> {
  try {
    if (!isRecord(rawInput)) throw new Error("tool input must be an object");
    switch (name) {
      case "read":
        return await readTool(repoDir, rawInput);
      case "grep":
        return await grepTool(repoDir, rawInput);
      case "search":
        return await searchTool(repoDir, rawInput);
      case "glob":
        return await globTool(repoDir, rawInput);
      case "tree":
        return await treeTool(repoDir, rawInput);
      case "outline":
        return await outlineTool(repoDir, rawInput);
      case "zoom":
        return await zoomTool(repoDir, rawInput);
      case "callgraph":
        return await callgraphTool(repoDir, rawInput);
      default:
        throw new Error(`unknown read-only tool: ${name}`);
    }
  } catch (error) {
    return { ok: false, output: "", error: error instanceof Error ? error.message : String(error) };
  }
}

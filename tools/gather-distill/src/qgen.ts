import { AccountPool, type AccountLease } from "./auth.ts";
import { AccountRejectedError, assistantText, sendMessage } from "./anthropic.ts";
import { QGEN_SYSTEM_PROMPT } from "../prompts/qgen.ts";
import { loadManifest } from "./repo.ts";
import { validateQuestion } from "./schema.ts";
import { AftClient, executeTool } from "./tools.ts";
import type { GatherJob } from "./types.ts";
import { parseJsonText } from "./utils.ts";

export interface QgenOptions {
  model?: string;
  count?: number;
  maxTokens?: number;
  /** Send output_config.effort. Off by default: only newer models accept it;
   * models without it reject the request with HTTP 400. */
  effort?: "low" | "medium" | "high";
  pool?: AccountPool;
  aftClient?: AftClient;
}

async function repoGrounding(repoDir: string, aftClient: AftClient): Promise<string> {
  const manifest = await loadManifest(repoDir);
  const files = await executeTool(repoDir, "glob", { pattern: "**/*" }, aftClient);
  const paths = files.output
    .split("\n")
    .filter((path) => path.length > 0 && !path.startsWith("[AFT ") && !path.startsWith("Search status:"));
  const readme = paths.find((path) => /^readme(?:\.[^/]*)?$/i.test(path));
  const entryPatterns = [
    /^(?:package\.json|Cargo\.toml|pyproject\.toml|go\.mod)$/,
    /^(?:src\/)?(?:main|lib|index)\.(?:rs|ts|tsx|js|py|go|java|kt|swift|c|cpp)$/,
  ];
  const selected = [readme, ...paths.filter((path) => entryPatterns.some((pattern) => pattern.test(path)))]
    .filter((path): path is string => Boolean(path))
    .filter((path, index, all) => all.indexOf(path) === index)
    .slice(0, 4);
  const excerpts: string[] = [];
  for (const path of selected) {
    const result = await executeTool(repoDir, "read", { filePath: path, startLine: 1, endLine: 200 }, aftClient);
    if (result.ok) excerpts.push(result.output.slice(0, 12_000));
  }
  return [
    `Repository: ${manifest.fullName}`,
    `Pinned SHA: ${manifest.sha}`,
    `Primary language: ${manifest.language}`,
    "",
    "Files:",
    files.output.slice(0, 20_000),
    "",
    "Grounding excerpts:",
    excerpts.join("\n\n---\n\n"),
  ].join("\n");
}

export async function generateQuestions(repoDir: string, options: QgenOptions = {}): Promise<GatherJob[]> {
  const pool = options.pool ?? new AccountPool();
  const model = options.model ?? "claude-sonnet-5-0";
  const count = options.count ?? 20;
  const ownsAftClient = options.aftClient === undefined;
  const aftClient = options.aftClient ?? new AftClient();
  try {
    const grounding = await repoGrounding(repoDir, aftClient);
    let lease: AccountLease = await pool.acquire();
    try {
      for (;;) {
        try {
          const response = await sendMessage(lease.credential, {
            model,
            max_tokens: options.maxTokens ?? 6_000,
            ...(options.effort ? { effort: options.effort } : {}),
            system: QGEN_SYSTEM_PROMPT,
            messages: [
              {
                role: "user",
                content: `${grounding}\n\nGenerate ${count} questions. The array must be balanced across request_class values.`,
              },
            ],
          });
          const parsed = parseJsonText(assistantText(response.content));
          if (!Array.isArray(parsed)) throw new Error("qgen response must be a JSON array");
          return parsed.map((question) => ({ dir: repoDir, ...validateQuestion(question) }));
        } catch (error) {
          if (!(error instanceof AccountRejectedError)) throw error;
          await pool.coolDown(lease);
          lease = await pool.acquire();
        }
      }
    } finally {
      pool.release(lease);
    }
  } finally {
    if (ownsAftClient) await aftClient.close();
  }
}

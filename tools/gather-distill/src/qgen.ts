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
  /** Questions from earlier passes over the same repo; the prompt instructs
   * the model not to duplicate them (deeper-coverage reruns). */
  avoid?: string[];
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
  const avoid = options.avoid ?? [];
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
                content: [
                  grounding,
                  ...(avoid.length > 0
                    ? [
                        "",
                        "Questions ALREADY generated for this repository in earlier passes (do NOT duplicate or trivially rephrase any of them; cover different files, subsystems, and behaviors):",
                        ...avoid.map((q) => `- ${q}`),
                      ]
                    : []),
                  "",
                  `Generate ${count} questions. The array must be balanced across request_class values.`,
                ].join("\n"),
              },
            ],
          });
          const text = assistantText(response.content);
          let parsed: unknown;
          try {
            parsed = parseJsonText(text);
          } catch (parseError) {
            // Dump the raw model text so a parse failure is diagnosable
            // (a terse "Unable to parse" with no artifact cost us a full pass).
            const dump = `/tmp/qgen-parse-fail-${Date.now()}.txt`;
            await Bun.write(dump, `repo: ${repoDir}\nstop_reason: ${response.stop_reason}\n---\n${text}`);
            throw new Error(`qgen parse failure (raw response dumped to ${dump}, stop_reason=${response.stop_reason}): ${parseError}`);
          }
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

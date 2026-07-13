import type { BankedRow } from "./types.ts";
import { readJsonl, writeJsonAtomic } from "./utils.ts";

export async function updateBurnRate(rowsPath: string, statusPath: string): Promise<void> {
  const rows = await readJsonl<BankedRow>(rowsPath);
  if (rows.length === 0) return;
  const first = Date.parse(rows[0].ts);
  const elapsedHours = Math.max((Date.now() - first) / 3_600_000, 1 / 3_600);
  const accounts: Record<string, { trajectories: number; input_tokens: number; output_tokens: number }> = {};
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;
  for (const row of rows) {
    inputTokens += row.input_tokens;
    outputTokens += row.output_tokens;
    cacheCreationTokens += row.cache_creation_input_tokens;
    cacheReadTokens += row.cache_read_input_tokens;
    const account = (accounts[row.account] ??= { trajectories: 0, input_tokens: 0, output_tokens: 0 });
    account.trajectories += 1;
    account.input_tokens += row.input_tokens;
    account.output_tokens += row.output_tokens;
  }
  // Cache-read tokens bill at ~10% of fresh input, so the cache hit ratio is
  // the headline efficiency number for an expiring subscription quota.
  const cacheableInput = inputTokens + cacheReadTokens;
  const cacheHitRatio = cacheableInput > 0 ? cacheReadTokens / cacheableInput : 0;
  await writeJsonAtomic(statusPath, {
    started_at: rows[0].ts,
    updated_at: new Date().toISOString(),
    trajectories: rows.length,
    valid_trajectories: rows.filter((row) => row.valid).length,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    cache_creation_input_tokens: cacheCreationTokens,
    cache_read_input_tokens: cacheReadTokens,
    cache_hit_ratio: cacheHitRatio,
    rolling_tokens_per_hour: (inputTokens + outputTokens) / elapsedHours,
    rolling_trajectories_per_hour: rows.length / elapsedHours,
    per_account: accounts,
  });
}

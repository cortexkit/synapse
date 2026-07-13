import { AccountPool, type AccountLease } from "./auth.ts";
import { AccountRejectedError, assistantText, sendMessage, type MessageResponse } from "./anthropic.ts";
import { sendOpenAiMessage } from "./openai.ts";
import {
  assertProductionFinalizeMode,
  gatherToolCallBudget,
  loadGatherSystemPrompt,
  type FinalizeMode,
} from "../prompts/gather-system.ts";
import { loadManifest, verifyPinnedHead } from "./repo.ts";
import { validateFinalJson } from "./schema.ts";
import { type AftClient, executeTool, GATHER_TOOLS } from "./tools.ts";
import type {
  AnthropicContentBlock,
  BankedRow,
  GatherFinalJson,
  GatherJob,
  TrajectoryMessage,
} from "./types.ts";
import { parseJsonText, stableJobId } from "./utils.ts";
import { validateBankedRow } from "./validate.ts";

export type GatherBackend = "anthropic" | "openai";

export interface GatherOptions {
  backend?: GatherBackend;
  baseUrl?: string;
  requestTimeoutMs?: number;
  model?: string;
  maxSteps?: number;
  maxPackageTokens?: number;
  tokenCeiling?: number;
  maxResponseTokens?: number;
  finalizeMode?: FinalizeMode;
  inlineValidate?: boolean;
  pool?: AccountPool;
  aftClient?: AftClient;
}

function userPrompt(job: GatherJob, sha: string, maxSteps: number, maxPackageTokens: number): string {
  return `Request:\n${job.request}\n\nScope:\n(caller did not restrict scope)\n\nRepository git HEAD: ${sha}\n\nBudgets:\n- maxSteps: ${maxSteps}\n- maxPackageTokens: ${maxPackageTokens}\n\nExplore with the attached read tools, then finish with the single JSON object described in the system instructions. Snippets should be path and line pointers only; the harness reads the bytes from disk.`;
}

function toolUses(content: AnthropicContentBlock[]): Array<Extract<AnthropicContentBlock, { type: "tool_use" }>> {
  return content.filter(
    (block): block is Extract<AnthropicContentBlock, { type: "tool_use" }> => block.type === "tool_use",
  );
}

function ignoredOpenAiFinalizeToolCall(content: AnthropicContentBlock[]): boolean {
  return toolUses(content).length > 0 || /<(?:function|tool_call)\b/i.test(assistantText(content));
}

function parseFinal(response: MessageResponse): { finalJson: GatherFinalJson | null; error?: string } {
  try {
    const parsed = parseJsonText(assistantText(response.content));
    const schema = validateFinalJson(parsed);
    return schema.value ? { finalJson: schema.value } : { finalJson: null, error: schema.errors.join("; ") };
  } catch (error) {
    return { finalJson: null, error: error instanceof Error ? error.message : String(error) };
  }
}

function modelForBackend(backend: GatherBackend, requested: string | undefined): string {
  return requested ?? (backend === "openai" ? "local-model" : "claude-opus-4-8");
}

export async function runGatherJob(job: GatherJob, options: GatherOptions = {}): Promise<BankedRow> {
  const backend = options.backend ?? "anthropic";
  const maxSteps = options.maxSteps ?? 40;
  const maxPackageTokens = options.maxPackageTokens ?? 40_000;
  const tokenCeiling = options.tokenCeiling ?? 200_000;
  const maxResponseTokens = options.maxResponseTokens ?? 8_000;
  const model = modelForBackend(backend, options.model);
  const finalizeMode = options.finalizeMode ?? "tool_choice_none_full_toolset";
  assertProductionFinalizeMode(finalizeMode);
  const budget = gatherToolCallBudget(maxSteps);
  const manifest = await loadManifest(job.dir);
  await verifyPinnedHead(job.dir, manifest.sha);
  const trajectory: TrajectoryMessage[] = [
    { role: "user", content: userPrompt(job, manifest.sha, maxSteps, maxPackageTokens) },
  ];
  const pool = backend === "anthropic" ? options.pool ?? new AccountPool() : undefined;
  let lease: AccountLease | undefined = pool ? await pool.acquire() : undefined;
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheCreationTokens = 0;
  let cacheReadTokens = 0;
  let thinkingTokens = 0;
  let toolCallCount = 0;
  const firedNudges = new Set<number>();
  let finalJson: GatherFinalJson | null = null;
  let reason: string | undefined;
  let budgetOutcome: BankedRow["budget_outcome"] = "natural";

  const callModel = async (finalize = false): Promise<MessageResponse> => {
    const request = {
      model,
      max_tokens: maxResponseTokens,
      system: loadGatherSystemPrompt(),
      messages: trajectory,
      tools: GATHER_TOOLS,
      tool_choice: finalize ? ({ type: "none" } as const) : undefined,
    };
    const recordUsage = (response: MessageResponse): MessageResponse => {
      inputTokens += response.usage.input_tokens;
      outputTokens += response.usage.output_tokens;
      cacheCreationTokens += response.usage.cache_creation_input_tokens;
      cacheReadTokens += response.usage.cache_read_input_tokens;
      thinkingTokens += response.usage.thinking_tokens;
      return response;
    };

    if (backend === "openai") {
      return recordUsage(
        await sendOpenAiMessage(request, {
          baseUrl: options.baseUrl,
          requestTimeoutMs: options.requestTimeoutMs,
        }),
      );
    }

    for (;;) {
      try {
        return recordUsage(await sendMessage(lease!.credential, request));
      } catch (error) {
        if (!(error instanceof AccountRejectedError)) throw error;
        await pool!.coolDown(lease!);
        lease = await pool!.acquire();
      }
    }
  };

  try {
    let mustFinalize = false;
    while (!mustFinalize) {
      const response = await callModel(false);
      trajectory.push({ role: "assistant", content: response.content });
      const calls = toolUses(response.content);
      if (calls.length === 0) {
        const parsed = parseFinal(response);
        finalJson = parsed.finalJson;
        reason = parsed.error;
        if (!finalJson) budgetOutcome = "invalid_final";
        break;
      }
      const toolResults: AnthropicContentBlock[] = [];
      for (const call of calls) {
        const result = await executeTool(job.dir, call.name, call.input, options.aftClient);
        toolResults.push({
          type: "tool_result",
          tool_use_id: call.id,
          // AFT formats tool output server-side; do not wrap or rewrite it.
          content: result.output,
          is_error: !result.ok,
        });
        toolCallCount += 1;
      }
      trajectory.push({ role: "user", content: toolResults });
      for (const nudge of budget.nudges) {
        if (toolCallCount >= nudge.at_tool_calls && !firedNudges.has(nudge.at_tool_calls)) {
          trajectory.push({ role: "user", content: nudge.text, synthetic: "budget_nudge" });
          firedNudges.add(nudge.at_tool_calls);
        }
      }
      mustFinalize =
        toolCallCount >= budget.finalize_at_tool_calls || inputTokens + outputTokens >= tokenCeiling;
    }

    if (mustFinalize) {
      budgetOutcome = "budget_finalize";
      trajectory.push({ role: "user", content: budget.finalize_text, synthetic: "budget_finalize" });
      const response = await callModel(true);
      trajectory.push({ role: "assistant", content: response.content });
      if (backend === "openai" && ignoredOpenAiFinalizeToolCall(response.content)) {
        // A local server ignored tool_choice:none. Do not execute another tool turn after the budget cap.
        reason = "budget_finalize: OpenAI-compatible server returned tool calls despite tool_choice:none";
      } else {
        const parsed = parseFinal(response);
        finalJson = parsed.finalJson;
        reason = parsed.error;
      }
    }
  } catch (error) {
    budgetOutcome = "api_error";
    reason = error instanceof Error ? error.message : String(error);
  } finally {
    if (pool && lease) pool.release(lease);
  }

  const row: BankedRow = {
    job_id: stableJobId(job.dir, job.request),
    request: job.request,
    repo_full: manifest.fullName,
    repo_sha: manifest.sha,
    tags: { ...job.tags, language: job.tags.language ?? manifest.language },
    full_trajectory: trajectory,
    final_json: finalJson,
    budget_outcome: budgetOutcome,
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    cache_creation_input_tokens: cacheCreationTokens,
    cache_read_input_tokens: cacheReadTokens,
    thinking_tokens: thinkingTokens,
    model,
    account: backend === "openai" ? "local" : lease!.credential.name,
    ts: new Date().toISOString(),
    valid: false,
    reason,
  };
  if (options.inlineValidate && row.final_json) {
    const validation = await validateBankedRow(row, job.dir);
    row.valid = validation.valid;
    row.reason = validation.errors.join("; ") || undefined;
  } else if (row.final_json) {
    row.valid = validateFinalJson(row.final_json).valid;
  }
  return row;
}

export async function failedGatherJob(
  job: GatherJob,
  reason: string,
  options: Pick<GatherOptions, "backend" | "model"> = {},
): Promise<BankedRow> {
  const backend = options.backend ?? "anthropic";
  const manifest = await loadManifest(job.dir);
  return {
    job_id: stableJobId(job.dir, job.request),
    request: job.request,
    repo_full: manifest.fullName,
    repo_sha: manifest.sha,
    tags: { ...job.tags, language: job.tags.language ?? manifest.language },
    full_trajectory: [],
    final_json: null,
    budget_outcome: "api_error",
    input_tokens: 0,
    output_tokens: 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 0,
    model: modelForBackend(backend, options.model),
    account: backend === "openai" ? "local" : "aft-warmup",
    ts: new Date().toISOString(),
    valid: false,
    reason,
  };
}

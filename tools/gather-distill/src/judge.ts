import { createHash } from "node:crypto";
import { assistantText, type MessageResponse } from "./anthropic.ts";
import { sendOpenAiMessage } from "./openai.ts";
import { readLineRange } from "./repo.ts";
import { validateFinalJson } from "./schema.ts";
import { executeTool, GATHER_TOOLS, type AftClient } from "./tools.ts";
import type { BankedRow, GatherFinalJson, GatherJob, AnthropicContentBlock, TrajectoryMessage } from "./types.ts";
import { isRecord, parseJsonText, stableJobId } from "./utils.ts";

export const JUDGE_MODEL_DEFAULT = "gpt-5.6";
export const JUDGE_TEMPERATURE = 0;
export const JUDGE_TOPUP_BUDGET = 15;

const JUDGE_PROMPT_BASE = [
  "You are a blind downstream consumer evaluating a repository evidence package.",
  "The user question and package below are the only evidence available in phase 1.",
  "Never infer or mention which system produced the package. Do not use repository tools in phase 1.",
  "",
  "Phase 1: answer the question using only the package. Return exactly one JSON object with:",
  '- phase1_sufficiency: "answerable_fully", "answerable_partially", or "not_answerable"',
  "- answer_draft: your best answer using only the package",
  "",
  "If phase 1 is not fully answerable, phase 2 begins in a later turn. Use only the attached read-only repository tools to fill the missing evidence.",
  `You have at most ${JUDGE_TOPUP_BUDGET} top-up tool calls. Count each tool call, including calls in a batch, and do not exceed that cap.`,
  "The tools are connected to the repository for the question. Read only what is needed; do not browse unrelated repositories.",
  "",
  "Phase 3: after phase 1, or after any needed top-up exploration, return exactly one JSON object with:",
  '- sufficiency: "full", "partial", or "none"',
  "- topup_tool_calls: the number of repository tool calls actually made",
  "- topup_tokens: the number of model tokens spent after phase 1",
  "- missing_evidence: a brief array describing what the package lacked",
  "- package_score: an integer from 1 through 10",
  "- score_rationale: a brief explanation of the score",
  "- answer: the finalized answer to the original question",
  "The harness replaces the two top-up counts with measured values, so report your best estimate.",
].join("\n");

const JUDGE_PROMPT_ADDENDA = [
  "Calibration discipline: classify a package as full only when its supplied bytes support a reliable answer, not merely because its interpretation sounds plausible. Be explicit about unanswered parts.",
  "Calibration discipline: paths and summaries are not evidence by themselves. Treat absent snippet bytes as missing evidence, and use top-up tools only when the package cannot support a reliable answer.",
] as const;

export interface JudgePrompt {
  iteration: number;
  text: string;
  sha: string;
  change: string;
}

export function judgePromptForIteration(iteration = 1): JudgePrompt {
  if (!Number.isInteger(iteration) || iteration < 1 || iteration > 3) {
    throw new Error("judge prompt iteration must be 1, 2, or 3");
  }
  const addendum = iteration === 1 ? "" : `\n\n${JUDGE_PROMPT_ADDENDA[iteration - 2]}`;
  const text = `${JUDGE_PROMPT_BASE}${addendum}`;
  return {
    iteration,
    text,
    sha: createHash("sha256").update(text).digest("hex"),
    change: iteration === 1 ? "base blind two-phase protocol" : JUDGE_PROMPT_ADDENDA[iteration - 2]!,
  };
}

export const JUDGE_SYSTEM_PROMPT = judgePromptForIteration(1).text;
export const JUDGE_PROMPT_SHA256 = judgePromptForIteration(1).sha;

export type Phase1Sufficiency = "answerable_fully" | "answerable_partially" | "not_answerable";
export type JudgeSufficiency = "full" | "partial" | "none";
export type JudgePackageKind = "candidate" | "gold" | "empty" | "mismatched";

export interface HydratedJudgeSnippet {
  path: string;
  startLine: number;
  endLine: number;
  why: string;
  text: string;
}

/** The only package bytes sent to the judge; it intentionally has no producer metadata. */
export interface HydratedJudgePackage {
  interpretation: string;
  scope: string[];
  snippets: HydratedJudgeSnippet[];
  omissions: GatherFinalJson["omissions"];
}

export interface Phase1Result {
  phase1_sufficiency: Phase1Sufficiency;
  answer_draft: string;
}

export interface ParsedJudgeVerdict {
  sufficiency: JudgeSufficiency;
  topup_tool_calls: number;
  topup_tokens: number;
  missing_evidence: string[];
  package_score: number;
  score_rationale: string;
  answer: string;
}

export interface JudgeToolCallRecord {
  index: number;
  id: string;
  name: string;
  input: unknown;
  ok: boolean;
  output: string;
}

export interface JudgeEvaluation {
  status: "completed";
  phase1: Phase1Result;
  verdict: ParsedJudgeVerdict;
  topup_calls: JudgeToolCallRecord[];
  judge_input_tokens: number;
  judge_output_tokens: number;
  phase1_tokens: number;
  topup_tokens_measured: number;
}

export interface JudgeRequestOptions {
  baseUrl: string;
  apiKey: string;
  apiKeyHeader?: string;
  model?: string;
  temperature?: number;
  topupBudget?: number;
  maxResponseTokens?: number;
  requestTimeoutMs?: number;
  prompt?: JudgePrompt;
}

export interface ControlPackageSpec {
  kind: Exclude<JudgePackageKind, "candidate">;
  question_job_id: string;
  question_request: string;
  package_job_id?: string;
  package_request?: string;
  package: GatherFinalJson;
}

export interface JudgeVerdictRow {
  job_id: string;
  request: string;
  repo_full: string;
  repo_sha: string;
  label: string;
  package_kind: JudgePackageKind;
  status: "completed" | "skipped_invalid" | "error";
  phase1_sufficiency: Phase1Sufficiency | "not_run" | null;
  answer_draft: string | null;
  sufficiency: JudgeSufficiency | null;
  topup_tool_calls: number;
  topup_tokens: number;
  missing_evidence: string[];
  package_score: number | null;
  score_rationale: string;
  answer: string;
  topup_trace: JudgeToolCallRecord[];
  judge_model: string;
  judge_prompt_sha: string;
  judge_temperature: number;
  judge_topup_budget: number;
  judge_input_tokens: number;
  judge_output_tokens: number;
  phase1_tokens: number;
  error?: string;
}

const EMPTY_PACKAGE: GatherFinalJson = {
  interpretation: "No repository evidence was supplied.",
  scope: [],
  snippets: [],
  omissions: [],
};

export function emptyJudgePackage(): GatherFinalJson {
  return structuredClone(EMPTY_PACKAGE);
}

function jobIdFor(job: GatherJob): string {
  return stableJobId(job.dir, job.request);
}

function rowMatchesJob(row: BankedRow, job: GatherJob): boolean {
  return (
    (row.job_id !== undefined && row.job_id === jobIdFor(job)) ||
    (row.request.trim() === job.request.trim() && row.repo_full.length > 0)
  );
}

function finalPackage(row: BankedRow | undefined): GatherFinalJson | null {
  if (!row?.final_json) return null;
  const validation = validateFinalJson(row.final_json);
  return validation.valid ? row.final_json : null;
}

function packageRowForJob(job: GatherJob, rows: BankedRow[]): BankedRow | undefined {
  return rows.find((row) => rowMatchesJob(row, job));
}

/**
 * Build the three blind calibration pairings. Mismatch selection prefers a
 * different question in the same repository so the judge sees a realistic
 * package shape while still receiving evidence for another question.
 */
export function buildControlPackages(question: GatherJob, goldRows: BankedRow[], jobs: GatherJob[]): ControlPackageSpec[] {
  const questionJobId = jobIdFor(question);
  const goldRow = packageRowForJob(question, goldRows);
  if (!goldRow || !finalPackage(goldRow)) {
    throw new Error(`gold package is missing or invalid for ${questionJobId}`);
  }

  const sameRepo = goldRows.filter(
    (row) => row.repo_full === goldRow.repo_full && row.request.trim() !== question.request.trim() && finalPackage(row),
  );
  const differentQuestion = sameRepo.length > 0
    ? sameRepo
    : goldRows.filter((row) => row.request.trim() !== question.request.trim() && finalPackage(row));
  const mismatchRow = [...differentQuestion].sort((left, right) => {
    const leftId = left.job_id ?? `${left.repo_full}\0${left.request}`;
    const rightId = right.job_id ?? `${right.repo_full}\0${right.request}`;
    return leftId.localeCompare(rightId);
  })[0];
  if (!mismatchRow || !finalPackage(mismatchRow)) {
    throw new Error(`cannot construct a mismatched package for ${questionJobId}; gold has no other question`);
  }

  const mismatchJob = jobs.find((job) => mismatchRow.job_id === jobIdFor(job))
    ?? jobs.find((job) => job.request.trim() === mismatchRow.request.trim());
  const packageJobId = mismatchJob ? jobIdFor(mismatchJob) : mismatchRow.job_id;
  if (packageJobId === questionJobId) {
    throw new Error(`mismatched package accidentally reused question ${questionJobId}`);
  }

  return [
    {
      kind: "gold",
      question_job_id: questionJobId,
      question_request: question.request,
      package_job_id: questionJobId,
      package_request: question.request,
      package: finalPackage(goldRow)!,
    },
    {
      kind: "empty",
      question_job_id: questionJobId,
      question_request: question.request,
      package: emptyJudgePackage(),
    },
    {
      kind: "mismatched",
      question_job_id: questionJobId,
      question_request: question.request,
      package_job_id: packageJobId,
      package_request: mismatchRow.request,
      package: finalPackage(mismatchRow)!,
    },
  ];
}

export async function hydrateJudgePackage(repoDir: string, value: GatherFinalJson): Promise<HydratedJudgePackage> {
  const validation = validateFinalJson(value);
  if (!validation.valid || !validation.value) {
    throw new Error(`cannot hydrate invalid package: ${validation.errors.join("; ")}`);
  }
  const snippets: HydratedJudgeSnippet[] = [];
  for (const snippet of validation.value.snippets) {
    const bytes = await readLineRange(repoDir, snippet.path, snippet.startLine, snippet.endLine);
    snippets.push({ ...snippet, text: bytes.text });
  }
  return {
    interpretation: validation.value.interpretation,
    scope: [...validation.value.scope],
    snippets,
    omissions: [...validation.value.omissions],
  };
}

export function isJudgeableRow(row: BankedRow | undefined): row is BankedRow & { final_json: GatherFinalJson } {
  return Boolean(
    row &&
      row.budget_outcome === "natural" &&
      row.valid &&
      row.final_json &&
      validateFinalJson(row.final_json).valid,
  );
}

function parseRecord(text: string, description: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = parseJsonText(text);
  } catch (error) {
    throw new Error(`${description} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
  }
  if (!isRecord(parsed)) throw new Error(`${description} must be a JSON object`);
  return parsed;
}

export function parsePhase1Response(text: string): Phase1Result {
  const value = parseRecord(text, "judge phase 1 response");
  const sufficiency = value.phase1_sufficiency;
  if (sufficiency !== "answerable_fully" && sufficiency !== "answerable_partially" && sufficiency !== "not_answerable") {
    throw new Error("judge phase 1 response has invalid phase1_sufficiency");
  }
  if (typeof value.answer_draft !== "string") throw new Error("judge phase 1 response is missing answer_draft");
  return { phase1_sufficiency: sufficiency, answer_draft: value.answer_draft };
}

function nonNegativeInteger(value: unknown, field: string): number {
  if (!Number.isInteger(value) || Number(value) < 0) throw new Error(`judge verdict ${field} must be a non-negative integer`);
  return Number(value);
}

export function parseJudgeVerdict(text: string): ParsedJudgeVerdict {
  const value = parseRecord(text, "judge verdict");
  if (value.sufficiency !== "full" && value.sufficiency !== "partial" && value.sufficiency !== "none") {
    throw new Error("judge verdict has invalid sufficiency");
  }
  const topupToolCalls = nonNegativeInteger(value.topup_tool_calls, "topup_tool_calls");
  const topupTokens = nonNegativeInteger(value.topup_tokens, "topup_tokens");
  if (!Array.isArray(value.missing_evidence) || value.missing_evidence.some((item) => typeof item !== "string")) {
    throw new Error("judge verdict missing_evidence must be an array of strings");
  }
  if (!Number.isInteger(value.package_score) || Number(value.package_score) < 1 || Number(value.package_score) > 10) {
    throw new Error("judge verdict package_score must be an integer from 1 through 10");
  }
  if (typeof value.score_rationale !== "string" || value.score_rationale.trim().length === 0) {
    throw new Error("judge verdict score_rationale must be a non-empty string");
  }
  return {
    sufficiency: value.sufficiency,
    topup_tool_calls: topupToolCalls,
    topup_tokens: topupTokens,
    missing_evidence: value.missing_evidence,
    package_score: Number(value.package_score),
    score_rationale: value.score_rationale,
    answer: typeof value.answer === "string" ? value.answer : "",
  };
}

function toolUses(content: AnthropicContentBlock[]): Array<Extract<AnthropicContentBlock, { type: "tool_use" }>> {
  return content.filter(
    (block): block is Extract<AnthropicContentBlock, { type: "tool_use" }> => block.type === "tool_use",
  );
}

function tokenTotal(response: MessageResponse): number {
  return Math.max(0, response.usage.input_tokens) + Math.max(0, response.usage.output_tokens);
}

function phase1User(question: string, package_: HydratedJudgePackage): string {
  return [
    "Original question:",
    question,
    "",
    "Candidate evidence package (the snippet text is the exact repository text supplied to the caller):",
    JSON.stringify(package_, null, 2),
    "",
    "Phase 1 only: do not use tools. Return the phase1 JSON object now.",
  ].join("\n");
}

function topupUser(remaining: number): string {
  return [
    "The package was not fully sufficient. You may now use the attached repository read tools to fill the gaps.",
    `There are ${remaining} top-up tool calls remaining. Count every requested tool call and stop before exceeding the cap.`,
    "After exploration, return the final verdict JSON object and the finalized answer. Do not mention package producers or labels.",
  ].join("\n");
}

function finalUser(): string {
  return "Phase 3: return the final verdict JSON object now. Use the measured exploration state, and include the finalized answer. Do not call tools.";
}

function budgetUser(): string {
  return "The top-up tool-call budget is exhausted. Do not call another tool. Return the final verdict JSON object now.";
}

function skippedEvaluation(
  label: string,
  kind: JudgePackageKind,
  row: BankedRow,
  options: Required<Pick<JudgeRequestOptions, "model" | "temperature" | "topupBudget">> & { prompt: JudgePrompt },
  reason: string,
): JudgeVerdictRow {
  return {
    job_id: row.job_id ?? stableJobId(row.repo_full, row.request),
    request: row.request,
    repo_full: row.repo_full,
    repo_sha: row.repo_sha,
    label,
    package_kind: kind,
    status: "skipped_invalid",
    phase1_sufficiency: "not_run",
    answer_draft: null,
    sufficiency: "none",
    topup_tool_calls: 0,
    topup_tokens: 0,
    missing_evidence: [reason],
    package_score: 1,
    score_rationale: reason,
    answer: "",
    topup_trace: [],
    judge_model: options.model,
    judge_prompt_sha: options.prompt.sha,
    judge_temperature: options.temperature,
    judge_topup_budget: options.topupBudget,
    judge_input_tokens: 0,
    judge_output_tokens: 0,
    phase1_tokens: 0,
  };
}

export async function runJudgeEvaluation(
  question: string,
  repoDir: string,
  package_: HydratedJudgePackage,
  options: JudgeRequestOptions,
  aftClient?: AftClient,
): Promise<JudgeEvaluation> {
  if (!options.baseUrl.trim()) throw new Error("judge base URL is required");
  if (!options.apiKey.trim()) throw new Error("judge API key is required");
  const model = options.model ?? JUDGE_MODEL_DEFAULT;
  const temperature = options.temperature ?? JUDGE_TEMPERATURE;
  const topupBudget = options.topupBudget ?? JUDGE_TOPUP_BUDGET;
  const maxResponseTokens = options.maxResponseTokens ?? 4_000;
  if (!Number.isInteger(topupBudget) || topupBudget <= 0) throw new Error("judge top-up budget must be a positive integer");
  if (!Number.isFinite(temperature) || temperature < 0) throw new Error("judge temperature must be non-negative");
  const prompt = options.prompt ?? judgePromptForIteration(1);
  const trajectory: TrajectoryMessage[] = [{ role: "user", content: phase1User(question, package_) }];
  let inputTokens = 0;
  let outputTokens = 0;
  let phase1Tokens = 0;
  let topupTokens = 0;
  const topupCalls: JudgeToolCallRecord[] = [];

  const callModel = async (withTools: boolean, finalize: boolean): Promise<MessageResponse> => {
    const response = await sendOpenAiMessage(
      {
        model,
        max_tokens: maxResponseTokens,
        system: prompt.text,
        messages: trajectory,
        ...(withTools ? { tools: GATHER_TOOLS } : {}),
        ...(finalize ? { tool_choice: { type: "none" } as const } : {}),
        temperature,
      },
      {
        baseUrl: options.baseUrl,
        apiKey: options.apiKey,
        apiKeyHeader: options.apiKeyHeader,
        requestTimeoutMs: options.requestTimeoutMs,
      },
    );
    inputTokens += response.usage.input_tokens;
    outputTokens += response.usage.output_tokens;
    return response;
  };

  const phase1Response = await callModel(false, true);
  phase1Tokens = tokenTotal(phase1Response);
  trajectory.push({ role: "assistant", content: phase1Response.content });
  const phase1 = parsePhase1Response(assistantText(phase1Response.content));

  const finalResponse = async (): Promise<ParsedJudgeVerdict> => {
    const response = await callModel(true, true);
    topupTokens += tokenTotal(response);
    trajectory.push({ role: "assistant", content: response.content });
    return parseJudgeVerdict(assistantText(response.content));
  };

  if (phase1.phase1_sufficiency === "answerable_fully") {
    trajectory.push({ role: "user", content: finalUser() });
    const verdict = await finalResponse();
    return {
      status: "completed",
      phase1,
      verdict: {
        ...verdict,
        topup_tool_calls: 0,
        topup_tokens: topupTokens,
      },
      topup_calls: [],
      judge_input_tokens: inputTokens,
      judge_output_tokens: outputTokens,
      phase1_tokens: phase1Tokens,
      topup_tokens_measured: topupTokens,
    };
  }

  trajectory.push({ role: "user", content: topupUser(topupBudget) });
  let verdict: ParsedJudgeVerdict | undefined;
  while (verdict === undefined) {
    const forceFinal = topupCalls.length >= topupBudget;
    const response = await callModel(true, forceFinal);
    topupTokens += tokenTotal(response);
    trajectory.push({ role: "assistant", content: response.content });
    const calls = toolUses(response.content);
    if (calls.length === 0) {
      verdict = parseJudgeVerdict(assistantText(response.content));
      break;
    }
    if (forceFinal) {
      throw new Error("judge endpoint returned tool calls after the top-up budget was exhausted");
    }

    const remaining = topupBudget - topupCalls.length;
    const allowedCalls = calls.slice(0, remaining);
    const results: AnthropicContentBlock[] = [];
    for (const call of allowedCalls) {
      let result: { ok: boolean; output: string };
      try {
        result = await executeTool(repoDir, call.name, call.input, aftClient);
      } catch (error) {
        result = { ok: false, output: error instanceof Error ? error.message : String(error) };
      }
      const record: JudgeToolCallRecord = {
        index: topupCalls.length,
        id: call.id,
        name: call.name,
        input: call.input,
        ok: result.ok,
        output: result.output,
      };
      topupCalls.push(record);
      results.push({ type: "tool_result", tool_use_id: call.id, content: result.output, is_error: !result.ok });
    }
    if (results.length > 0) trajectory.push({ role: "user", content: results });
    if (topupCalls.length >= topupBudget) {
      trajectory.push({ role: "user", content: budgetUser(), synthetic: "budget_finalize" });
    }
  }

  return {
    status: "completed",
    phase1,
    verdict: {
      ...verdict,
      topup_tool_calls: topupCalls.length,
      topup_tokens: topupTokens,
    },
    topup_calls: topupCalls,
    judge_input_tokens: inputTokens,
    judge_output_tokens: outputTokens,
    phase1_tokens: phase1Tokens,
    topup_tokens_measured: topupTokens,
  };
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2;
}

function mean(values: number[]): number | null {
  return values.length === 0 ? null : values.reduce((total, value) => total + value, 0) / values.length;
}

export interface JudgeSummary {
  rows: number;
  completed: number;
  skipped_invalid: number;
  errors: number;
  sufficiency: Record<JudgeSufficiency, number>;
  topup_calls_mean: number | null;
  topup_calls_median: number | null;
  topup_tokens_mean: number | null;
  package_score_mean: number | null;
}

export function summarizeJudgeRows(rows: JudgeVerdictRow[]): JudgeSummary {
  const validRows = rows.filter((row) => row.status !== "error" && row.sufficiency !== null);
  const calls = validRows.map((row) => row.topup_tool_calls);
  const tokens = validRows.map((row) => row.topup_tokens);
  const scores = validRows.flatMap((row) => (row.package_score === null ? [] : [row.package_score]));
  return {
    rows: rows.length,
    completed: rows.filter((row) => row.status === "completed").length,
    skipped_invalid: rows.filter((row) => row.status === "skipped_invalid").length,
    errors: rows.filter((row) => row.status === "error").length,
    sufficiency: {
      full: validRows.filter((row) => row.sufficiency === "full").length,
      partial: validRows.filter((row) => row.sufficiency === "partial").length,
      none: validRows.filter((row) => row.sufficiency === "none").length,
    },
    topup_calls_mean: mean(calls),
    topup_calls_median: median(calls),
    topup_tokens_mean: mean(tokens),
    package_score_mean: mean(scores),
  };
}

export interface CalibrationGate {
  pass: boolean;
  gold_mean_topup_calls: number | null;
  empty_mean_topup_calls: number | null;
  mismatch_none: number;
  mismatch_rows: number;
  reasons: string[];
}

export function evaluateCalibrationGate(rows: JudgeVerdictRow[]): CalibrationGate {
  const control = (kind: Exclude<JudgePackageKind, "candidate">) => rows.filter((row) => row.package_kind === kind && row.status !== "error");
  const gold = control("gold");
  const empty = control("empty");
  const mismatch = control("mismatched");
  const goldMean = mean(gold.map((row) => row.topup_tool_calls));
  const emptyMean = mean(empty.map((row) => row.topup_tool_calls));
  const mismatchNone = mismatch.filter((row) => row.sufficiency === "none").length;
  const reasons: string[] = [];
  if (gold.length === 0 || goldMean === null || goldMean >= 2) reasons.push("gold mean top-up must be below 2 calls");
  if (empty.length === 0 || emptyMean === null || emptyMean <= 8) reasons.push("empty mean top-up must be above 8 calls");
  if (mismatch.length === 0 || mismatchNone < Math.ceil(mismatch.length * 0.6)) reasons.push("mismatched packages must be flagged as none on at least 60% of jobs");
  return {
    pass: reasons.length === 0,
    gold_mean_topup_calls: goldMean,
    empty_mean_topup_calls: emptyMean,
    mismatch_none: mismatchNone,
    mismatch_rows: mismatch.length,
    reasons,
  };
}

export interface JudgeCostEstimate {
  sample_rows: number;
  projected_packages: number;
  mean_input_tokens: number | null;
  mean_output_tokens: number | null;
  projected_input_tokens: number | null;
  projected_output_tokens: number | null;
  projected_usd: number | null;
}

export function estimateJudgeCost(
  calibrationRows: JudgeVerdictRow[],
  projectedPackages: number,
  inputUsdPerMillion?: number,
  outputUsdPerMillion?: number,
): JudgeCostEstimate {
  const completed = calibrationRows.filter((row) => row.status === "completed");
  const input = mean(completed.map((row) => row.judge_input_tokens));
  const output = mean(completed.map((row) => row.judge_output_tokens));
  const projectedInput = input === null ? null : input * projectedPackages;
  const projectedOutput = output === null ? null : output * projectedPackages;
  const projectedUsd =
    projectedInput !== null &&
    projectedOutput !== null &&
    inputUsdPerMillion !== undefined &&
    outputUsdPerMillion !== undefined
      ? (projectedInput * inputUsdPerMillion + projectedOutput * outputUsdPerMillion) / 1_000_000
      : null;
  return {
    sample_rows: completed.length,
    projected_packages: projectedPackages,
    mean_input_tokens: input,
    mean_output_tokens: output,
    projected_input_tokens: projectedInput,
    projected_output_tokens: projectedOutput,
    projected_usd: projectedUsd,
  };
}

export function skippedJudgeRow(
  label: string,
  kind: JudgePackageKind,
  row: BankedRow,
  options: JudgeRequestOptions,
  reason: string,
): JudgeVerdictRow {
  const prompt = options.prompt ?? judgePromptForIteration(1);
  return skippedEvaluation(
    label,
    kind,
    row,
    {
      model: options.model ?? JUDGE_MODEL_DEFAULT,
      temperature: options.temperature ?? JUDGE_TEMPERATURE,
      topupBudget: options.topupBudget ?? JUDGE_TOPUP_BUDGET,
      prompt,
    },
    reason,
  );
}

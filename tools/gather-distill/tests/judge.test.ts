import { expect, test } from "bun:test";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildControlPackages,
  hydrateJudgePackage,
  parseJudgeVerdict,
  parsePhase1Response,
  type HydratedJudgePackage,
} from "../src/judge.ts";
import type { BankedRow, GatherJob, GatherFinalJson } from "../src/types.ts";
import { stableJobId } from "../src/utils.ts";

function job(dir: string, request: string): GatherJob {
  return {
    dir,
    request,
    tags: { request_class: "feature_orientation", expected_difficulty: 2, specificity: "high" },
  };
}

function package_(path: string): GatherFinalJson {
  return {
    interpretation: `Evidence for ${path}`,
    scope: [path],
    snippets: [{ path, startLine: 1, endLine: 2, why: "implementation" }],
    omissions: [],
  };
}

function goldRow(source: GatherJob, fullName: string, value: GatherFinalJson): BankedRow {
  return {
    job_id: stableJobId(source.dir, source.request),
    request: source.request,
    repo_full: fullName,
    repo_sha: "a".repeat(40),
    tags: source.tags,
    full_trajectory: [],
    final_json: value,
    budget_outcome: "natural",
    input_tokens: 1,
    output_tokens: 1,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 0,
    model: "gold",
    account: "test",
    ts: new Date(0).toISOString(),
    valid: true,
  };
}

test("parses the phase 1 and final judge JSON contracts", () => {
  expect(parsePhase1Response('{"phase1_sufficiency":"answerable_partially","answer_draft":"The package identifies the entry point."}')).toEqual({
    phase1_sufficiency: "answerable_partially",
    answer_draft: "The package identifies the entry point.",
  });
  expect(parseJudgeVerdict('{"sufficiency":"partial","topup_tool_calls":3,"topup_tokens":222,"missing_evidence":["the caller"],"package_score":7,"score_rationale":"The main path is present but one dependency is absent.","answer":"The feature starts here."}')).toEqual({
    sufficiency: "partial",
    topup_tool_calls: 3,
    topup_tokens: 222,
    missing_evidence: ["the caller"],
    package_score: 7,
    score_rationale: "The main path is present but one dependency is absent.",
    answer: "The feature starts here.",
  });
});

test("mismatched control pairs question A with package B", () => {
  const questionA = job("/tmp/repo", "Where is feature A implemented?");
  const questionB = job("/tmp/repo", "Where is feature B implemented?");
  const controls = buildControlPackages(
    questionA,
    [goldRow(questionA, "owner/repo", package_("src/feature-a.ts")), goldRow(questionB, "owner/repo", package_("src/feature-b.ts"))],
    [questionA, questionB],
  );
  const mismatch = controls.find((control) => control.kind === "mismatched")!;
  expect(mismatch.question_job_id).toBe(stableJobId(questionA.dir, questionA.request));
  expect(mismatch.package_job_id).toBe(stableJobId(questionB.dir, questionB.request));
  expect(mismatch.package_job_id).not.toBe(mismatch.question_job_id);
  expect(mismatch.package_request).toBe(questionB.request);
  expect(mismatch.package.snippets[0]?.path).toBe("src/feature-b.ts");
  expect(controls.find((control) => control.kind === "empty")?.package.snippets).toEqual([]);
});

test("hydrates snippet bytes from the pinned repository before judging", async () => {
  const repo = await mkdtemp(join(tmpdir(), "judge-hydration-"));
  await writeFile(join(repo, "feature.ts"), "export function feature() {\n  return true;\n}\n");
  const hydrated: HydratedJudgePackage = await hydrateJudgePackage(repo, package_("feature.ts"));
  expect(hydrated.snippets).toEqual([
    { path: "feature.ts", startLine: 1, endLine: 2, why: "implementation", text: "export function feature() {\n  return true;\n" },
  ]);
});

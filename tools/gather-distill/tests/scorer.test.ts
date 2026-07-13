import { expect, test } from "bun:test";
import { scoreRows } from "../src/scorer.ts";
import type { BankedRow } from "../src/types.ts";

function row(overrides: Partial<BankedRow>): BankedRow {
  return {
    job_id: "job-1",
    request: "Where is the answer assembled?",
    repo_full: "owner/repo",
    repo_sha: "a".repeat(40),
    tags: { request_class: "feature_orientation", expected_difficulty: 2, specificity: "high" },
    full_trajectory: [],
    final_json: {
      interpretation: "Find the answer assembly.",
      scope: [],
      snippets: [],
      omissions: [],
    },
    budget_outcome: "natural",
    input_tokens: 20,
    output_tokens: 10,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
    thinking_tokens: 3,
    model: "candidate-model",
    account: "local",
    ts: new Date(0).toISOString(),
    valid: true,
    ...overrides,
  };
}

test("scores non-vacuous file, line, and tool overlap against gold", () => {
  const candidate = row({
    final_json: {
      interpretation: "Candidate answer.",
      scope: ["context-only.ts"],
      snippets: [
        { path: "./src/a.ts", startLine: 1, endLine: 4, why: "Candidate range." },
        { path: "src/b.ts", startLine: 1, endLine: 1, why: "Candidate-only file." },
      ],
      omissions: [],
    },
    full_trajectory: [
      {
        role: "assistant",
        content: [
          { type: "tool_use", id: "candidate-1", name: "read", input: {} },
          { type: "tool_use", id: "candidate-2", name: "search", input: {} },
        ],
      },
    ],
  });
  const gold = row({
    model: "claude-opus-4-8",
    output_tokens: 40,
    thinking_tokens: 0,
    final_json: {
      interpretation: "Gold answer.",
      scope: [],
      snippets: [
        { path: "src/a.ts", startLine: 3, endLine: 6, why: "Overlapping gold range." },
        { path: "src/c.ts", startLine: 9, endLine: 10, why: "Gold-only file." },
      ],
      omissions: [],
    },
    full_trajectory: [
      {
        role: "assistant",
        content: Array.from({ length: 4 }, (_, index) => ({ type: "tool_use" as const, id: `gold-${index}`, name: "read", input: {} })),
      },
    ],
  });

  const report = scoreRows([candidate], [gold]);
  expect(report.jobs).toHaveLength(1);
  expect(report.jobs[0]).toMatchObject({
    job_id: "job-1",
    contract_valid: true,
    file_f1: 0.5,
    line_overlap: 1 / 3,
    tool_efficiency: 0.5,
    candidate_tool_calls: 2,
    gold_tool_calls: 4,
    output_tokens: 10,
    thinking_tokens: 3,
  });
  expect(report.aggregate.file_f1).toEqual({ count: 1, mean: 0.5, median: 0.5 });
  expect(report.aggregate.line_overlap).toEqual({ count: 1, mean: 1 / 3, median: 1 / 3 });
  expect(report.summary_row).toMatchObject({
    model: "candidate-model",
    jobs: 1,
    budget_natural: 1,
    output_tokens_mean: 10,
    thinking_tokens_mean: 3,
  });
});


test("clamps cited spans to the pinned file length before line overlap", () => {
  const candidate = row({
    final_json: {
      interpretation: "Candidate answer.",
      scope: [],
      snippets: [{ path: "src/a.ts", startLine: 1, endLine: 99, why: "Whole file." }],
      omissions: [],
    },
  });
  const gold = row({
    final_json: {
      interpretation: "Gold answer.",
      scope: [],
      snippets: [{ path: "src/a.ts", startLine: 1, endLine: 3, why: "Whole file." }],
      omissions: [],
    },
  });
  const key = `${candidate.repo_full}\0${candidate.repo_sha}\0${candidate.request}`;
  const caps = new Map([[key, new Map([["src/a.ts", 3]])]]);

  expect(scoreRows([candidate], [gold], new Map(), caps).jobs[0]!.line_overlap).toBe(1);
});

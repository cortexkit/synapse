import { describe, expect, test } from "bun:test";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { GatherFinalJson, TrajectoryMessage } from "../src/types.ts";
import { validateCitations } from "../src/validate.ts";

async function fixtureRepo(): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), "gather-validate-"));
  await mkdir(join(root, "src"));
  await writeFile(join(root, "src", "lib.ts"), "export const one = 1;\nexport const two = 2;\n");
  return root;
}

function finalJson(startLine: number, endLine: number): GatherFinalJson {
  return {
    interpretation: "Find the exported constants.",
    scope: ["src/lib.ts"],
    snippets: [{ path: "src/lib.ts", startLine, endLine, why: "Defines the constants." }],
    omissions: [],
  };
}

describe("citation validation", () => {
  test("known-bad citation (startLine past EOF) must fail", async () => {
    const errors = await validateCitations(await fixtureRepo(), finalJson(90, 99));
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.join(" ")).toContain("past the end");
  });

  test("endLine past EOF clamps and passes (production parity)", async () => {
    const errors = await validateCitations(await fixtureRepo(), finalJson(1, 99));
    expect(errors).toEqual([]);
  });

  test("rejects trajectory snippet bytes that differ from the pinned file", async () => {
    const trajectory: TrajectoryMessage[] = [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "tool-1",
            content: JSON.stringify({
              ok: true,
              output: "",
              provenance: [{ path: "src/lib.ts", startLine: 1, endLine: 2, text: "tampered\nbytes\n" }],
            }),
          },
        ],
      },
    ];
    const errors = await validateCitations(await fixtureRepo(), finalJson(1, 2), trajectory);
    expect(errors).toEqual(["final_json.snippets[0]: trajectory snippet bytes do not match the pinned clone"]);
  });
});

import { parseJsonText } from "../src/utils.ts";

test("parseJsonText extracts the fenced block after a prose preamble", () => {
  const withPreamble = 'I have the complete picture across the client and server.\n\n```json\n{\n  "interpretation": "x",\n  "scope": ["a"],\n  "snippets": [],\n  "omissions": []\n}\n```';
  const parsed = parseJsonText(withPreamble) as Record<string, unknown>;
  expect(parsed.interpretation).toBe("x");
  expect(parsed.scope).toEqual(["a"]);
});

test("parseJsonText prefers the last fenced block and tolerates a trailing sign-off", () => {
  const multi = 'thinking:\n```json\n{"draft":true}\n```\nfinal:\n```json\n{"interpretation":"final","scope":[],"snippets":[],"omissions":[]}\n```\nDone.';
  const parsed = parseJsonText(multi) as Record<string, unknown>;
  expect(parsed.interpretation).toBe("final");
});

test("parseJsonText falls back to the outermost brace span when unfenced", () => {
  const bare = 'Here it is: {"interpretation":"bare","scope":[],"snippets":[],"omissions":[]} thanks';
  const parsed = parseJsonText(bare) as Record<string, unknown>;
  expect(parsed.interpretation).toBe("bare");
});

import { readLineRange } from "../src/repo.ts";
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execSync } from "node:child_process";

test("readLineRange clamps endLine past EOF (production parity) but rejects startLine past EOF", async () => {
  const dir = mkdtempSync(join(tmpdir(), "clamp-test-"));
  execSync("git init -q", { cwd: dir });
  writeFileSync(join(dir, "f.txt"), "a\nb\nc\n");
  const sha = "0000000000000000000000000000000000000000";
  writeFileSync(join(dir, ".gather-corpus-manifest.json"), JSON.stringify({ fullName: "t/t", sha, language: "x", size_mb: 0 }));
  // endLine 90 on a 3-line file: clamped, accepted
  const clamped = await readLineRange(dir, "f.txt", 1, 90);
  expect(clamped.text).toBe("a\nb\nc\n");
  expect(clamped.lineCount).toBe(3);
  // startLine past EOF: genuine error, rejected
  await expect(readLineRange(dir, "f.txt", 10, 12)).rejects.toThrow(/past the end/);
});

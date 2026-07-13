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

function finalJson(endLine: number): GatherFinalJson {
  return {
    interpretation: "Find the exported constants.",
    scope: ["src/lib.ts"],
    snippets: [{ path: "src/lib.ts", startLine: 1, endLine, why: "Defines the constants." }],
    omissions: [],
  };
}

describe("citation validation", () => {
  test("known-bad out-of-range citation must fail", async () => {
    const errors = await validateCitations(await fixtureRepo(), finalJson(99));
    expect(errors.length).toBeGreaterThan(0);
    expect(errors.join(" ")).toContain("exceeds file length");
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
    const errors = await validateCitations(await fixtureRepo(), finalJson(2), trajectory);
    expect(errors).toEqual(["final_json.snippets[0]: trajectory snippet bytes do not match the pinned clone"]);
  });
});

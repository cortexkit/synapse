import { describe, expect, test } from "bun:test";
import { GATHER_TOOLS } from "../src/tools.ts";
import {
  assertProductionFinalizeMode,
  GATHER_BUDGET_FINALIZE_TEXT,
  gatherToolCallBudget,
  loadGatherSystemPrompt,
} from "../prompts/gather-system.ts";

describe("production gather contract", () => {
  test("default 40-step thresholds and texts match the production builder", () => {
    expect(gatherToolCallBudget(40)).toEqual({
      nudges: [
        {
          at_tool_calls: 20,
          text: "you have used 20 tool calls; steer toward wrapping up — record what you have",
        },
        {
          at_tool_calls: 25,
          text: "5 calls left in your comfortable budget; finish current thread and prepare your final JSON",
        },
      ],
      finalize_at_tool_calls: 30,
      finalize_text: GATHER_BUDGET_FINALIZE_TEXT,
    });
  });

  test("system prompt retains exact trailing newline and active contract markers", () => {
    const prompt = loadGatherSystemPrompt();
    expect(prompt.endsWith("depth_limit.\n")).toBeTrue();
    expect(prompt).toContain("Start with search —");
    expect(prompt).not.toContain("record_evidence");
  });

  test("tools-empty finalization is forbidden", () => {
    expect(() => assertProductionFinalizeMode("tools_empty")).toThrow("byte-identical toolset");
    expect(() => assertProductionFinalizeMode("tool_choice_none_full_toolset")).not.toThrow();
  });

  test("declares the bare production AFT catalog without the retired tree tool", () => {
    expect(GATHER_TOOLS.map((tool) => tool.name)).toEqual([
      "search",
      "outline",
      "zoom",
      "callgraph",
      "read",
      "grep",
      "glob",
      "inspect",
      "conflicts",
    ]);
    expect(GATHER_TOOLS.find((tool) => tool.name === "search")?.input_schema).toMatchObject({
      required: ["query"],
      properties: { hint: { enum: ["regex", "literal", "semantic", "auto"] } },
    });
    expect(GATHER_TOOLS.find((tool) => tool.name === "callgraph")?.input_schema).toMatchObject({
      required: ["op", "filePath", "symbol"],
    });
  });
});

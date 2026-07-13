import { GATHER_SYSTEM_PROMPT_V10 } from "./gather-system-v10.ts";

export type FinalizeMode = "tool_choice_none_full_toolset" | "tools_empty";

export const GATHER_BUDGET_FINALIZE_TEXT =
  'You are out of exploration budget. Do not call any tools. Reply now with the single JSON object described in the system instructions (interpretation, scope, snippets as path+startLine+endLine pointers, omissions) and nothing else. Point at the code you found; for leads you did not finish, add omissions with why "budget".';

export interface GatherBudget {
  nudges: Array<{ at_tool_calls: number; text: string }>;
  finalize_at_tool_calls: number;
  finalize_text: string;
}

export function loadGatherSystemPrompt(): string {
  return GATHER_SYSTEM_PROMPT_V10;
}

// Byte-for-byte equivalent to gather_tool_call_budget_thresholds and its payload text at
// CortexKit/alfonso commit 3ff7970e723e3c228c6efa4c61d27092db42d078.
export function gatherToolCallBudget(maxSteps: number): GatherBudget {
  const steps = Math.max(0, Math.floor(maxSteps));
  const firstNudge = Math.max(Math.floor(steps / 2), 1);
  const secondNudge = Math.max(Math.floor((steps * 5) / 8), firstNudge + 1);
  const finalize = Math.max(Math.floor((steps * 3) / 4), secondNudge + 1);
  return {
    nudges: [
      {
        at_tool_calls: firstNudge,
        text: `you have used ${firstNudge} tool calls; steer toward wrapping up — record what you have`,
      },
      {
        at_tool_calls: secondNudge,
        text: `${Math.max(0, finalize - secondNudge)} calls left in your comfortable budget; finish current thread and prepare your final JSON`,
      },
    ],
    finalize_at_tool_calls: finalize,
    finalize_text: GATHER_BUDGET_FINALIZE_TEXT,
  };
}

export function assertProductionFinalizeMode(mode: FinalizeMode): void {
  if (mode !== "tool_choice_none_full_toolset") {
    throw new Error(
      "tools_empty forks from the production student contract; finalization must retain the byte-identical toolset with tool_choice:none",
    );
  }
}

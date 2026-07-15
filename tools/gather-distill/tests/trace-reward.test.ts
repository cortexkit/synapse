import { expect, test } from "bun:test";
import { join } from "node:path";

const root = join(import.meta.dir, "..");
const fixtures = join(import.meta.dir, "fixtures");
const gold = join(fixtures, "trace-gold-rows.jsonl");
const job = "ca7f354653c24facc5e8921c86d72263546ed221c389f8ff081aa3ec9188f67b";

async function scoreCandidate(candidate: string): Promise<{ reward: number; diagnostics: Record<string, unknown> }> {
  const process = Bun.spawn(
    ["python3", "train_reward/reward.py", "--job", job, "--candidate-file", join(fixtures, candidate), "--gold", gold],
    { cwd: root, stdout: "pipe", stderr: "pipe" },
  );
  const [status, stdout, stderr] = await Promise.all([
    process.exited,
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
  ]);
  expect(status, stderr).toBe(0);
  return JSON.parse(stdout) as { reward: number; diagnostics: Record<string, unknown> };
}

test("TRACE reward gives an Opus gold package full credit", async () => {
  const verdict = await scoreCandidate("trace-candidate-gold.json");
  expect(verdict.reward).toBeCloseTo(1, 8);
  expect(verdict.diagnostics).toMatchObject({ contract_valid: true, line_jaccard: 1, tool_calls: 2 });
});

test("TRACE reward rejects a malformed final package", async () => {
  const verdict = await scoreCandidate("trace-candidate-malformed.json");
  expect(verdict.reward).toBe(0);
  expect(verdict.diagnostics).toMatchObject({ contract_valid: false });
});

test("TRACE reward gives partial credit to a valid wrong-files package", async () => {
  const verdict = await scoreCandidate("trace-candidate-wrong-files.json");
  expect(verdict.diagnostics).toMatchObject({ contract_valid: true, tool_calls: 2 });
  expect(verdict.reward).toBeGreaterThan(0);
  expect(verdict.reward).toBeLessThan(1);
});

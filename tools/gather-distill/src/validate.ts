import { join } from "node:path";
import { loadManifest, readLineRange, splitLinesInclusive, verifyPinnedHead } from "./repo.ts";
import { validateFinalJson } from "./schema.ts";
import type { BankedRow, GatherFinalJson, LocalToolResult, ToolProvenance, TrajectoryMessage } from "./types.ts";
import { isRecord } from "./utils.ts";

function trajectoryProvenance(trajectory: TrajectoryMessage[]): ToolProvenance[] {
  const provenance: ToolProvenance[] = [];
  for (const message of trajectory) {
    if (!Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (block.type !== "tool_result") continue;
      try {
        const parsed: unknown = JSON.parse(block.content);
        if (!isRecord(parsed) || !Array.isArray(parsed.provenance)) continue;
        for (const item of parsed.provenance as LocalToolResult["provenance"]) {
          if (
            item &&
            typeof item.path === "string" &&
            Number.isInteger(item.startLine) &&
            Number.isInteger(item.endLine) &&
            typeof item.text === "string"
          ) {
            provenance.push(item);
          }
        }
      } catch {
        // Non-JSON tool errors carry no source bytes to compare.
      }
    }
  }
  return provenance;
}

function sliceClaim(claim: ToolProvenance, startLine: number, endLine: number): string | null {
  if (claim.startLine > startLine || claim.endLine < endLine) return null;
  const lines = splitLinesInclusive(claim.text);
  const offset = startLine - claim.startLine;
  const count = endLine - startLine + 1;
  if (offset + count > lines.length) return null;
  return lines.slice(offset, offset + count).join("");
}

export async function validateCitations(
  repoDir: string,
  finalJson: GatherFinalJson,
  trajectory: TrajectoryMessage[] = [],
): Promise<string[]> {
  const errors: string[] = [];
  const claims = trajectoryProvenance(trajectory);
  for (const [index, snippet] of finalJson.snippets.entries()) {
    let diskText: string;
    try {
      diskText = (await readLineRange(repoDir, snippet.path, snippet.startLine, snippet.endLine)).text;
    } catch (error) {
      errors.push(`final_json.snippets[${index}]: ${error instanceof Error ? error.message : String(error)}`);
      continue;
    }
    const matchingClaims = claims
      .filter((claim) => claim.path === snippet.path)
      .map((claim) => sliceClaim(claim, snippet.startLine, snippet.endLine))
      .filter((text): text is string => text !== null);
    if (matchingClaims.length > 0 && !matchingClaims.some((text) => text === diskText)) {
      errors.push(`final_json.snippets[${index}]: trajectory snippet bytes do not match the pinned clone`);
    }
  }
  return errors;
}

export async function validateBankedRow(row: BankedRow, repoDir: string): Promise<{ valid: boolean; errors: string[] }> {
  const errors: string[] = [];
  let manifest;
  try {
    manifest = await loadManifest(repoDir);
    if (row.repo_full !== manifest.fullName) errors.push("repo_full does not match the pinned clone manifest");
    if (row.repo_sha !== manifest.sha) errors.push("repo_sha does not match the pinned clone manifest");
    await verifyPinnedHead(repoDir, manifest.sha);
  } catch (error) {
    errors.push(error instanceof Error ? error.message : String(error));
  }
  const schema = validateFinalJson(row.final_json);
  errors.push(...schema.errors);
  if (schema.value) errors.push(...(await validateCitations(repoDir, schema.value, row.full_trajectory)));
  return { valid: errors.length === 0, errors };
}

export function repoDirForRow(corpusRoot: string, row: BankedRow): string {
  const parts = row.repo_full.split("/");
  if (parts.length !== 2 || parts.some((part) => part.length === 0)) {
    throw new Error(`cannot map repo_full to corpus directory: ${row.repo_full}`);
  }
  return join(corpusRoot, `${parts[0]}__${parts[1]}`);
}

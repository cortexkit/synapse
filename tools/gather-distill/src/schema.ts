import type { GatherFinalJson, JobTags } from "./types.ts";
import { REQUEST_CLASSES } from "./types.ts";
import { isRecord } from "./utils.ts";

const OMISSION_REASONS = new Set(["budget", "empty_result", "skipped_candidate", "depth_limit"]);
const SPECIFICITIES = new Set(["low", "med", "high"]);

function exactKeys(value: Record<string, unknown>, keys: string[], at: string, errors: string[]): void {
  const expected = new Set(keys);
  for (const key of Object.keys(value)) if (!expected.has(key)) errors.push(`${at}.${key}: unexpected field`);
  for (const key of keys) if (!(key in value)) errors.push(`${at}.${key}: missing field`);
}

function nonEmptyString(value: unknown, at: string, errors: string[]): value is string {
  if (typeof value !== "string" || value.trim().length === 0) {
    errors.push(`${at}: expected a non-empty string`);
    return false;
  }
  return true;
}

export function validateFinalJson(value: unknown): { valid: boolean; errors: string[]; value?: GatherFinalJson } {
  const errors: string[] = [];
  if (!isRecord(value)) return { valid: false, errors: ["final_json: expected an object"] };
  exactKeys(value, ["interpretation", "scope", "snippets", "omissions"], "final_json", errors);
  nonEmptyString(value.interpretation, "final_json.interpretation", errors);
  if (!Array.isArray(value.scope)) errors.push("final_json.scope: expected an array");
  else value.scope.forEach((item, index) => nonEmptyString(item, `final_json.scope[${index}]`, errors));
  if (!Array.isArray(value.snippets)) errors.push("final_json.snippets: expected an array");
  else {
    value.snippets.forEach((item, index) => {
      const at = `final_json.snippets[${index}]`;
      if (!isRecord(item)) return errors.push(`${at}: expected an object`);
      exactKeys(item, ["path", "startLine", "endLine", "why"], at, errors);
      if (nonEmptyString(item.path, `${at}.path`, errors)) {
        if (item.path.startsWith("/") || item.path.split(/[\\/]/).includes("..")) {
          errors.push(`${at}.path: expected a repository-relative path without '..'`);
        }
      }
      if (!Number.isInteger(item.startLine) || Number(item.startLine) < 1) {
        errors.push(`${at}.startLine: expected a positive integer`);
      }
      if (!Number.isInteger(item.endLine) || Number(item.endLine) < Number(item.startLine)) {
        errors.push(`${at}.endLine: expected an integer at least startLine`);
      }
      nonEmptyString(item.why, `${at}.why`, errors);
    });
  }
  if (!Array.isArray(value.omissions)) errors.push("final_json.omissions: expected an array");
  else {
    value.omissions.forEach((item, index) => {
      const at = `final_json.omissions[${index}]`;
      if (!isRecord(item)) return errors.push(`${at}: expected an object`);
      exactKeys(item, ["what", "why", "detail"], at, errors);
      nonEmptyString(item.what, `${at}.what`, errors);
      if (!OMISSION_REASONS.has(String(item.why))) errors.push(`${at}.why: invalid omission reason`);
      nonEmptyString(item.detail, `${at}.detail`, errors);
    });
  }
  return errors.length === 0
    ? { valid: true, errors, value: value as unknown as GatherFinalJson }
    : { valid: false, errors };
}

export function validateQuestion(value: unknown): { request: string; tags: JobTags } {
  if (!isRecord(value)) throw new Error("question must be an object");
  exactQuestionKeys(value, ["request", "tags"], "question");
  if (typeof value.request !== "string" || value.request.trim().length < 12) {
    throw new Error("question.request must be a concrete non-empty question");
  }
  if (!isRecord(value.tags)) throw new Error("question.tags must be an object");
  exactQuestionKeys(value.tags, ["request_class", "expected_difficulty", "specificity"], "question.tags");
  if (!REQUEST_CLASSES.includes(value.tags.request_class as never)) throw new Error("invalid request_class");
  if (!Number.isInteger(value.tags.expected_difficulty) || Number(value.tags.expected_difficulty) < 1 || Number(value.tags.expected_difficulty) > 5) {
    throw new Error("expected_difficulty must be an integer from 1 through 5");
  }
  if (!SPECIFICITIES.has(String(value.tags.specificity))) throw new Error("invalid specificity");
  return { request: value.request.trim(), tags: value.tags as unknown as JobTags };
}

function exactQuestionKeys(value: Record<string, unknown>, keys: string[], at: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.join("\0") !== expected.join("\0")) {
    throw new Error(`${at} must contain exactly ${keys.join(", ")}`);
  }
}

export const REQUEST_CLASSES = [
  "bug_investigation",
  "feature_orientation",
  "api_usage",
  "refactor_prep",
  "cross_module_trace",
] as const;
export type RequestClass = (typeof REQUEST_CLASSES)[number];
export type Specificity = "low" | "med" | "high";

export interface JobTags {
  request_class: RequestClass;
  expected_difficulty: number;
  specificity: Specificity;
  language?: string;
  [key: string]: unknown;
}

export interface GatherJob {
  dir: string;
  request: string;
  tags: JobTags;
}

export interface RepoManifest {
  fullName: string;
  sha: string;
  language: string;
  size_mb: number;
}

export interface FinalSnippet {
  path: string;
  startLine: number;
  endLine: number;
  why: string;
}

export interface FinalOmission {
  what: string;
  why: "budget" | "empty_result" | "skipped_candidate" | "depth_limit";
  detail: string;
}

export interface GatherFinalJson {
  interpretation: string;
  scope: string[];
  snippets: FinalSnippet[];
  omissions: FinalOmission[];
}

export type AnthropicContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: unknown }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean };

export interface TrajectoryMessage {
  role: "user" | "assistant";
  content: string | AnthropicContentBlock[];
  synthetic?: "budget_nudge" | "budget_finalize";
}

export interface BankedRow {
  request: string;
  repo_full: string;
  repo_sha: string;
  tags: JobTags;
  full_trajectory: TrajectoryMessage[];
  final_json: GatherFinalJson | null;
  budget_outcome: "natural" | "budget_finalize" | "api_error" | "invalid_final";
  input_tokens: number;
  output_tokens: number;
  model: string;
  account: string;
  ts: string;
  valid: boolean;
  reason?: string;
}

export interface LedgerEntry {
  job_id: string;
  dir: string;
  request: string;
  tags: JobTags;
  outcome: "banked" | "rejected" | "failed";
  duration_ms: number;
  input_tokens: number;
  output_tokens: number;
  account: string;
  valid: boolean;
  reason?: string;
  ts: string;
}

export interface ToolProvenance {
  path: string;
  startLine: number;
  endLine: number;
  text: string;
}

export interface ToolResult {
  ok: boolean;
  output: string;
  provenance?: ToolProvenance[];
  error?: string;
}

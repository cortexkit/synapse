import {
  applyClaudeCodeHeaders,
  applyClaudeCodeMetadata,
  buildBillingHeaderValue,
  CLAUDE_CODE_ENTRYPOINT,
  CLAUDE_CODE_IDENTITY,
  orderClaudeCodeBody,
  resolveClaudeCodeIdentity,
  signRequestBody,
} from "@cortexkit/anthropic-auth-core";
import type { Credential } from "./auth.ts";
import type { AnthropicContentBlock, TrajectoryMessage } from "./types.ts";
import type { ToolDeclaration } from "./tools.ts";
import { isRecord } from "./utils.ts";

const ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages";

export interface MessageRequest {
  model: string;
  max_tokens: number;
  system: string;
  messages: TrajectoryMessage[];
  tools?: ToolDeclaration[];
  tool_choice?: { type: "none" };
  effort?: "low" | "medium" | "high";
}

export interface MessageResponse {
  content: AnthropicContentBlock[];
  stop_reason: string | null;
  usage: { input_tokens: number; output_tokens: number };
}

export class AccountRejectedError extends Error {
  readonly status: number;

  constructor(status: number) {
    super(status === 401 ? "Anthropic rejected the account credentials" : "Anthropic account is temporarily unavailable");
    this.name = "AccountRejectedError";
    this.status = status;
  }
}

function responseShape(value: unknown): MessageResponse {
  if (!isRecord(value) || !Array.isArray(value.content) || !isRecord(value.usage)) {
    throw new Error("Anthropic returned an unexpected response shape");
  }
  const input = value.usage.input_tokens;
  const output = value.usage.output_tokens;
  if (typeof input !== "number" || typeof output !== "number") {
    throw new Error("Anthropic response is missing token usage");
  }
  return {
    content: value.content as AnthropicContentBlock[],
    stop_reason: typeof value.stop_reason === "string" ? value.stop_reason : null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

export async function sendMessage(credential: Credential, request: MessageRequest): Promise<MessageResponse> {
  const headers = new Headers({
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
  });
  const messages = request.messages.map(({ role, content }) => ({ role, content }));
  const body: Record<string, unknown> = {
    model: request.model,
    max_tokens: request.max_tokens,
    system: request.system,
    messages,
  };
  if (request.tools) body.tools = request.tools;
  if (request.tool_choice) body.tool_choice = request.tool_choice;
  if (request.effort) body.output_config = { effort: request.effort };

  let bodyText: string;
  if (credential.kind === "oauth") {
    const identity = await resolveClaudeCodeIdentity(credential.secret, request.model);
    const billingHeader = buildBillingHeaderValue(messages, undefined, CLAUDE_CODE_ENTRYPOINT);
    body.system = [
      { type: "text", text: billingHeader },
      { type: "text", text: CLAUDE_CODE_IDENTITY },
      { type: "text", text: request.system },
    ];
    applyClaudeCodeMetadata(body, identity);
    applyClaudeCodeHeaders(headers, credential.secret, { body, identity });
    bodyText = await signRequestBody(JSON.stringify(orderClaudeCodeBody(body)));
  } else {
    headers.set("x-api-key", credential.secret);
    bodyText = JSON.stringify(body);
  }

  const response = await fetch(ANTHROPIC_MESSAGES_URL, {
    method: "POST",
    headers,
    body: bodyText,
  });
  if (response.status === 401 || response.status === 429) throw new AccountRejectedError(response.status);
  if (!response.ok) {
    const requestId = response.headers.get("request-id");
    throw new Error(`Anthropic messages request failed (${response.status}${requestId ? `, request ${requestId}` : ""})`);
  }
  return responseShape(await response.json());
}

export function assistantText(content: AnthropicContentBlock[]): string {
  return content
    .filter((block): block is Extract<AnthropicContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("\n");
}
